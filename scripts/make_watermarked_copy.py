"""Create a watermarked re-distribution copy of a research report PDF.

Simulates the real-world scenario from D-031: the SAME report re-saved by a
distribution platform (different bytes / different file SHA256, same content)
with a download watermark stamped on the first page footer.

Usage:
    python scripts/make_watermarked_copy.py <source.pdf> <target.pdf>
"""

import sys
from pathlib import Path

import pymupdf


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    doc = pymupdf.open(src)
    page = doc[0]
    rect = page.rect
    # Stamp a platform-style watermark line at the bottom of page 1.
    page.insert_text(
        (36, rect.height - 18),
        "用户691141754于2026-08-22日下载，仅供本人内部使用，不可传播与转载",
        fontsize=7,
        color=(0.5, 0.5, 0.5),
    )
    # Re-save with full garbage collection + deflate: guarantees different
    # file bytes (like a platform re-encode) even apart from the stamp.
    doc.save(dst, garbage=4, deflate=True)
    doc.close()

    import hashlib

    h_src = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    h_dst = hashlib.sha256(Path(dst).read_bytes()).hexdigest()[:16]
    print(f"source sha256[:16] = {h_src}")
    print(f"target sha256[:16] = {h_dst}")
    print(f"bytes differ: {h_src != h_dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
