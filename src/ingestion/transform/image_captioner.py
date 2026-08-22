"""Image Captioner transform for enriching chunks with image descriptions.

Performance Optimizations:
1. Only processes images that are actually referenced in chunk text (via [IMAGE: id] placeholder)
2. Uses caption cache to avoid redundant Vision API calls for the same image
3. Skips chunks without image references entirely
4. Parallel processing of unique images with thread-safe caching
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Optional, Dict

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Regex to find image placeholders: [IMAGE: some_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# Default max parallel workers for Vision API calls
DEFAULT_MAX_WORKERS = 3  # Lower than text LLM due to higher cost/latency

# Cap caption generation length. The prompt asks for a comprehensive summary
# and llm.max_tokens (4096) would otherwise be inherited; on a local Ollama
# vision model (~3 token/s solo, ~1 token/s per stream with 3 workers) an
# unbounded caption can exceed even a 300s timeout. ~200 tokens is ample
# for retrieval-oriented captions.
DEFAULT_CAPTION_MAX_TOKENS = 200


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.
    
    This transform identifies chunks containing image references, uses a Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with
    these captions to improve retrieval for visual content.
    
    Key Features:
    - Only processes images actually referenced in chunk text (not all images in metadata)
    - Caches captions to avoid redundant Vision API calls
    - Thread-safe caption cache for potential future parallelization
    """
    
    def __init__(
        self,
        settings: Settings,
        llm: Optional[BaseVisionLLM] = None,
        image_storage: Optional[Any] = None,
    ):
        self.settings = settings
        self.image_storage = image_storage
        self.llm = None
        # Caption cache: image_id -> caption string (thread-safe with lock)
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        # D-031: persistent caption cache keyed by (model, sha256(image bytes)).
        # image_id is NOT stable across re-parses (docling re-renders bbox
        # regions each run), but the rendered bytes of an identical page are —
        # making re-distributed copies skip the Vision LLM almost entirely.
        from src.core.settings import resolve_path
        self._caption_db = str(resolve_path("data/db/caption_cache.db"))
        self._init_caption_db()

        # Check if vision LLM is enabled in settings
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # We don't raise here to allow pipeline to continue without captioning
                 # effectively falling back to no-op for this transform
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")

        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the image captioning prompt from configuration."""
        # Assuming standard relative path. In production, logic might be robust.
        from src.core.settings import resolve_path
        prompt_path = resolve_path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """Extract image IDs actually referenced in the chunk text.
        
        Args:
            text: Chunk text content
            
        Returns:
            List of image IDs found in [IMAGE: id] placeholders
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self, 
        img_id: str, 
        img_path: str, 
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """Get caption for an image, using cache if available. Thread-safe.
        
        Args:
            img_id: Image identifier
            img_path: Path to image file
            trace: Optional trace context
            
        Returns:
            Caption string or None if failed
        """
        # Check cache first (thread-safe read)
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]

        # D-031 persistent layer: (model, file-bytes hash) survives process
        # restarts AND re-parses of identical content.
        file_hash = self._image_file_hash(img_path)
        if file_hash is not None:
            cached_caption = self._persistent_caption_get(file_hash)
            if cached_caption is not None:
                with self._cache_lock:
                    self._caption_cache[img_id] = cached_caption
                return cached_caption

        # Validate path
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None
        
        try:
            image_input = ImageInput(path=img_path)
            response = self.llm.chat_with_image(
                text=self.prompt,
                image=image_input,
                trace=trace,
                max_tokens=DEFAULT_CAPTION_MAX_TOKENS,
            )
            caption = response.content
            
            # Cache the result (thread-safe write)
            with self._cache_lock:
                self._caption_cache[img_id] = caption
            if file_hash is not None:
                self._persistent_caption_put(file_hash, caption)
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    # ── D-031 persistent caption cache helpers ────────────────────────────

    def _init_caption_db(self) -> None:
        """Create the caption cache table if needed (best-effort)."""
        try:
            import sqlite3

            conn = sqlite3.connect(self._caption_db)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS caption_cache (
                        model TEXT NOT NULL,
                        image_hash TEXT NOT NULL,
                        caption TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (model, image_hash)
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Caption cache init failed (disabled): {e}")
            self._caption_db = None

    def _caption_model(self) -> str:
        return getattr(self.llm, "model", type(self.llm).__name__) if self.llm else "unknown"

    @staticmethod
    def _image_file_hash(img_path: str) -> Optional[str]:
        """SHA256 of image bytes — None if unreadable."""
        try:
            import hashlib

            h = hashlib.sha256()
            with open(img_path, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
            return h.hexdigest()
        except Exception:
            return None

    def _persistent_caption_get(self, image_hash: str) -> Optional[str]:
        if not self._caption_db:
            return None
        try:
            import sqlite3

            conn = sqlite3.connect(self._caption_db)
            try:
                cursor = conn.execute(
                    "SELECT caption FROM caption_cache "
                    "WHERE model = ? AND image_hash = ?",
                    (self._caption_model(), image_hash),
                )
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"Caption cache read failed (ignored): {e}")
            return None

    def _persistent_caption_put(self, image_hash: str, caption: str) -> None:
        if not self._caption_db:
            return
        try:
            import sqlite3
            from datetime import datetime, timezone

            conn = sqlite3.connect(self._caption_db)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO caption_cache "
                    "(model, image_hash, caption, created_at) VALUES (?, ?, ?, ?)",
                    (
                        self._caption_model(),
                        image_hash,
                        caption,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"Caption cache write failed (ignored): {e}")

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Generate captions: resolve image paths via ImageStorage, call vision
        LLM, write captions via set_caption, and stitch caption into chunk.text
        so it stays searchable.

        No longer writes chunk.metadata["image_captions"]. When image_storage
        is not injected, paths cannot be resolved and captioning is skipped.
        """
        if not self.llm:
            return chunks

        with self._cache_lock:
            self._caption_cache.clear()

        # Collect image_id -> file_path for all referenced images.
        # Images that already carry a persisted caption (written by a previous
        # run via set_caption) are pre-seeded into the in-memory cache instead
        # of being re-captioned — re-ingest then skips the local Vision LLM
        # (~1min/image) for already-captioned images.
        images_to_caption: Dict[str, str] = {}
        reused_captions = 0
        seen_ids: set = set()
        for chunk in chunks:
            for img_id in self._find_referenced_image_ids(chunk.text):
                img_id_s = img_id.strip()
                if img_id_s in seen_ids:
                    continue
                seen_ids.add(img_id_s)
                meta = self._resolve_image_meta(img_id_s)
                if not meta:
                    continue
                if meta.get("caption"):
                    with self._cache_lock:
                        self._caption_cache[img_id_s] = meta["caption"]
                    reused_captions += 1
                else:
                    images_to_caption[img_id_s] = meta["file_path"]

        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)

        # Stitch captions into text + persist via set_caption
        total_captions_added = 0
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            if not referenced_ids:
                continue
            new_text = chunk.text
            for img_id in referenced_ids:
                img_id_s = img_id.strip()
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_s)
                if not caption:
                    continue
                new_text = new_text.replace(
                    f"[IMAGE: {img_id}]",
                    f"[IMAGE: {img_id}]\n(Description: {caption})",
                )
                total_captions_added += 1
                if self.image_storage is not None:
                    self.image_storage.set_caption(img_id_s, caption)
            chunk.text = new_text

        api_calls = len(images_to_caption)
        logger.info(
            f"Added {total_captions_added} captions "
            f"(API calls: {api_calls}, reused from storage: {reused_captions})"
        )
        return chunks

    def _resolve_image_meta(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Resolve an image's metadata (file_path, caption, ...) via ImageStorage.

        Returns None when image_storage is not injected, when the image is
        not registered, or when the file does not exist on disk.
        """
        if self.image_storage is None:
            return None
        try:
            meta = self.image_storage.get_image_meta(image_id)
        except Exception as e:
            logger.warning(f"get_image_meta failed for {image_id}: {e}")
            return None
        if meta and meta.get("file_path") and Path(meta["file_path"]).exists():
            return meta
        return None
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Generate captions for multiple images in parallel.
        
        Args:
            images_to_caption: Dict of img_id -> img_path
            trace: Optional trace context
        """
        if not images_to_caption:
            return
        
        max_workers = min(DEFAULT_MAX_WORKERS, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    logger.error(f"Failed to generate caption for {img_id}: {e}")
