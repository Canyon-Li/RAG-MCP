"""Extract text from all PDFs in eval_docs directory for QA generation."""
import os
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("pdfplumber not available, trying pypdf...", file=sys.stderr)
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

EVAL_DIR = Path(r"D:\Desktop\Code\RAG-MCP\MODULAR-RAG-MCP-SERVER\tests\fixtures\eval_docs")
OUT_DIR = Path(r"D:\Desktop\Code\RAG-MCP\MODULAR-RAG-MCP-SERVER\scripts\_pdf_text")
OUT_DIR.mkdir(exist_ok=True)


def extract_with_pdfplumber(pdf_path: Path) -> str:
    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text() or ""
            text_parts.append(f"--- PAGE {i+1} ---\n{t}")
    return "\n\n".join(text_parts)


def extract_with_pypdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text_parts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        text_parts.append(f"--- PAGE {i+1} ---\n{t}")
    return "\n\n".join(text_parts)


def main():
    pdfs = sorted([f for f in EVAL_DIR.iterdir() if f.suffix.lower() == ".pdf"])
    print(f"Found {len(pdfs)} PDFs")
    for pdf in pdfs:
        out_file = OUT_DIR / (pdf.stem + ".txt")
        print(f"\n=== Extracting: {pdf.name} ===")
        try:
            if pdfplumber:
                text = extract_with_pdfplumber(pdf)
            else:
                text = extract_with_pypdf(pdf)
            out_file.write_text(text, encoding="utf-8")
            print(f"  -> {out_file} ({len(text)} chars)")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
