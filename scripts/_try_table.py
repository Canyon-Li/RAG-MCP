"""Quick demo: run PdfTableParser on a PDF and print the extracted sections.

Usage:
    conda activate langchain-test
    python scripts/_try_table.py <pdf-path>
    python scripts/_try_table.py   # defaults to chinese_table_chart_doc.pdf
"""
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.libs.parser.pdf_table_parser import PdfTableParser

default_pdf = _REPO / "tests" / "fixtures" / "sample_documents" / "chinese_table_chart_doc.pdf"
pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(default_pdf)

print(f"解析: {pdf_path}")
parser = PdfTableParser(image_storage_dir="data/images/try", extract_images=False)
doc = parser.parse(pdf_path)

secs = doc.metadata.get("sections", [])
print(f"degraded: {doc.metadata.get('degraded', False)}")
print(f"sections 总数: {len(secs)}")
print(f"类型分布: {dict(Counter(s['type'] for s in secs))}")

tables = [s for s in secs if s["type"] == "table"]
print(f"\n表格段: {len(tables)} 个")
for i, s in enumerate(tables[:]):
    print(f"\n--- table #{i} (page {s['page']}) ---")
    print("纯文本(chunk.text):", repr(s["text"][:200]))
    print("HTML(table_html)  :", repr(s["html"][:200]))

if not tables:
    print("\n（没提取到表格——该 PDF 可能是图片型/扫描件，pdf_table 已降级或仅产纯文本）")
