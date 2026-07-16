"""Core data types and contracts for the entire pipeline.

This module defines the fundamental data structures used across all pipeline stages:
- ingestion (loaders, transforms, embedding, storage)
- retrieval (query engine, search, reranking)
- mcp_server (tools, response formatting)

Design Principles:
- Centralized contracts: All stages use these types to avoid coupling
- Serializable: All types support dict/JSON conversion
- Extensible metadata: Minimum required fields with flexible extension
- Type-safe: Full type hints for static analysis
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """Represents a raw document loaded from source.
    
    This is the output of Loaders (e.g., PDF Loader) before splitting.
    
    Attributes:
        id: Unique identifier for the document (e.g., file hash or path-based ID)
        text: Document content in standardized Markdown format.
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Document-level metadata including:
            - source_path (required): Original file path
            - doc_type: Document type (e.g., 'pdf', 'markdown')
            - title: Document title extracted or inferred
            - page_count: Total pages (if applicable)
            - images: Intermediate list of image references produced by parsers
              (see Images Field Specification below). Consumed by the ingestion
              pipeline's image-registration step (stage 2.5 ``register_image``)
              and **dropped before chunk metadata is written** — downstream
              Chunk / ChunkRecord metadata never carries ``images``.
            - Any other custom metadata

    Images Field Specification (metadata.images):
        Structure: List[{"id": str, "path": str, "page": int, "text_offset": int,
                        "text_length": int, "position": dict}]
        Fields:
            - id: Unique image identifier (format: {doc_hash}_{page}_{seq})
            - path: Image file storage path (convention: data/images/{collection}/{image_id}.png)
            - page: Page number in original document (optional, for paginated docs like PDF)
            - text_offset: Starting character position of placeholder in Document.text (0-based)
            - text_length: Length of placeholder string (typically len("[IMAGE: {image_id}]"))
            - position: Physical position info in original doc (optional, e.g., PDF coords, pixel position)
        Note: text_offset and text_length enable precise placeholder location,
              supporting scenarios where the same image appears multiple times.
              This list is a parser-to-pipeline intermediate — it is not persisted
              into chunk metadata; image data lives in ImageStorage SQLite.
    
    Example:
        >>> doc = Document(
        ...     id="doc_abc123",
        ...     text="# Title\\n\\nContent...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "doc_type": "pdf",
        ...         "title": "Annual Report 2025"
        ...     }
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate required metadata fields."""
        if "source_path" not in self.metadata:
            raise ValueError("Document metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        return cls(**data)


@dataclass
class Chunk:
    """Represents a text chunk after splitting a Document.
    
    This is the output of Splitters and input to Transform pipeline.
    Each chunk maintains traceability to its source document.
    
    Attributes:
        id: Unique chunk identifier (e.g., hash-based or sequential)
        text: Chunk content (subset of original document text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Chunk-level metadata inherited and extended from Document:
            - source_path (required): Original file path
            - chunk_index: Sequential position in document (0-based)
            - start_offset: Character offset in original document (optional)
            - end_offset: Character offset in original document (optional)
            - source_ref: Reference to parent document ID (optional)
            - Any document-level metadata propagated from Document
        start_offset: Starting character position in original document (optional)
        end_offset: Ending character position in original document (optional)
        source_ref: Reference to parent Document.id (optional)

    Note: ``images`` / ``image_captions`` / ``image_refs`` are **no longer
          written** to chunk metadata. Image data lives in ImageStorage SQLite
          and is recovered at query time via ``[IMAGE: id]`` placeholders in
          ``chunk.text`` (see ``ImageStorage.get_image_meta``).
    
    Example:
        >>> chunk = Chunk(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "page": 1
        ...     },
        ...     start_offset=0,
        ...     end_offset=150
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    source_ref: Optional[str] = None
    
    def __post_init__(self):
        """Validate required metadata fields."""
        if "source_path" not in self.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        return cls(**data)


@dataclass
class Section:
    """A typed structural segment produced by a Parser (K2).

    Layout-aware parsers (e.g. pdf_deep via the RAGFlow deepdoc sidecar)
    populate ``Document.metadata["sections"]`` with Section-like dicts. The
    DocumentChunker respects type boundaries when splitting — tables are kept
    whole, titles merge into adjacent text, headers/footers are dropped, etc.

    Note: when stored in ``Document.metadata["sections"]`` these are plain dicts
    (JSON-serializable); this dataclass exists for type clarity and test
    construction (``Section(...).to_dict()``).

    Chunk-level optional metadata keys produced from sections:
        - section_type: the source Section.type inherited by the chunk
        - bbox: the source Section.bbox (tables / figures)
        - table_html: original HTML for table chunks (display-grade); the chunk's
          ``text`` holds cleaned plain text for embedding

    Attributes:
        type: Segment category — title | text | table | figure | figure_caption |
              list | header | footer | equation.
        text: Segment content. For ``table`` this is the HTML; for others, plain
              text (figure segments carry ``[IMAGE:{id}]`` placeholders).
        page: Source page number (1-based for paginated docs; 1 otherwise).
        bbox: Optional bounding box ``{x0, top, x1, bottom}`` in source coords.
        images: Image refs associated with this segment (figures / captions),
                following the Document.images contract.
        html: Optional display-grade HTML (tables; redundant with ``text``).
    """
    type: str
    text: str
    page: int = 1
    bbox: Optional[Dict[str, Any]] = None
    images: List[Dict[str, Any]] = field(default_factory=list)
    html: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Section":
        """Create Section from dictionary."""
        return cls(**data)


@dataclass
class ChunkRecord:
    """Represents a fully processed chunk ready for storage and retrieval.
    
    This is the output of the embedding pipeline and the data structure
    stored in vector databases. It extends Chunk with vector representations.
    
    Attributes:
        id: Unique chunk identifier (must be stable for idempotent upsert)
        text: Chunk content (same as Chunk.text).
              Images are represented as placeholders: [IMAGE: {image_id}]
        metadata: Extended metadata including:
            - source_path (required): Original file path
            - chunk_index: Sequential position
            - All metadata from Chunk
            - Any enrichment from Transform pipeline (title, summary, tags)
        dense_vector: Dense embedding vector (e.g., from OpenAI, BGE)
        sparse_vector: Sparse vector for BM25/keyword matching (optional)

    Note: ``metadata.image_captions`` is obsolete — captions now live in
          ImageStorage SQLite (``image_index.caption`` column, read/written via
          ``set_caption`` / ``get_image_meta``) and are also stitched into chunk
          text for searchability. ``metadata.images`` / ``metadata.image_refs``
          are likewise not written; image data is recovered at query time via
          ``[IMAGE: id]`` placeholders in ``text``.
    
    Example:
        >>> record = ChunkRecord(
        ...     id="chunk_abc123_001",
        ...     text="## Section 1\\n\\nFirst paragraph...",
        ...     metadata={
        ...         "source_path": "data/documents/report.pdf",
        ...         "chunk_index": 0,
        ...         "title": "Introduction",
        ...         "summary": "Overview of project goals"
        ...     },
        ...     dense_vector=[0.1, 0.2, ..., 0.3],
        ...     sparse_vector={"word1": 0.5, "word2": 0.3}
        ... )
    """
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate required metadata fields."""
        if "source_path" not in self.metadata:
            raise ValueError("ChunkRecord metadata must contain 'source_path'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """Create ChunkRecord from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_chunk(cls, chunk: Chunk, dense_vector: Optional[List[float]] = None,
                   sparse_vector: Optional[Dict[str, float]] = None) -> "ChunkRecord":
        """Create ChunkRecord from a Chunk with vectors.
        
        Args:
            chunk: Source Chunk object
            dense_vector: Dense embedding vector
            sparse_vector: Sparse vector representation
            
        Returns:
            ChunkRecord with all fields populated from chunk
        """
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )


# Type aliases for convenience
Metadata = Dict[str, Any]
Vector = List[float]
SparseVector = Dict[str, float]


@dataclass
class ProcessedQuery:
    """Represents a processed query ready for retrieval.
    
    This is the output of QueryProcessor, containing extracted keywords
    and parsed filters for downstream Dense/Sparse retrievers.
    
    Attributes:
        original_query: The raw user query string
        keywords: List of extracted keywords after stopword removal
        filters: Dictionary of filter conditions (e.g., {"collection": "api-docs"})
        expanded_terms: Optional list of synonyms/expanded terms (for future use)
    
    Example:
        >>> pq = ProcessedQuery(
        ...     original_query="如何配置 Azure OpenAI？",
        ...     keywords=["配置", "Azure", "OpenAI"],
        ...     filters={"collection": "docs"}
        ... )
    """
    
    original_query: str
    keywords: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    expanded_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedQuery":
        """Create ProcessedQuery from dictionary."""
        return cls(**data)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result from Dense/Sparse retrievers.
    
    This is the output of DenseRetriever, SparseRetriever, and HybridSearch,
    providing a unified contract for retrieval results across all search methods.
    
    Attributes:
        chunk_id: Unique identifier for the retrieved chunk
        score: Relevance score (higher = more relevant, normalized to [0, 1])
        text: The actual text content of the retrieved chunk
        metadata: Associated metadata (source_path, chunk_index, title, etc.)
    
    Example:
        >>> result = RetrievalResult(
        ...     chunk_id="doc1_chunk_003",
        ...     score=0.85,
        ...     text="Azure OpenAI 配置步骤如下...",
        ...     metadata={
        ...         "source_path": "docs/azure-guide.pdf",
        ...         "chunk_index": 3,
        ...         "title": "Azure Configuration"
        ...     }
        ... )
    """
    
    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be numeric, got {type(self.score).__name__}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Create RetrievalResult from dictionary."""
        return cls(**data)
