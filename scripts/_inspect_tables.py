"""Inspect table-extraction quality: structural metrics + HTML render.

Usage:
    python scripts/_inspect_tables.py [pdf-path]
"""
import html
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.libs.parser.pdf_table_parser import PdfTableParser

default_pdf = _REPO / "tests/fixtures/sample_documents/chinese_table_chart_doc.pdf"
pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(default_pdf)
out_html = _REPO / "scripts/_table_inspect.html"

parser = PdfTableParser(image_storage_dir="data/images/try", extract_images=False)
doc = parser.parse(pdf_path)
tables = [s for s in doc.metadata.get("sections", []) if s["type"] == "table"]

print(f"PDF: {pdf_path}")
print(f"degraded: {doc.metadata.get('degraded', False)}")
print(f"表格数: {len(tables)}\n")

print("=== 结构健康指标 ===")
_TR = re.compile(r"<tr>(.*?)</tr>", re.S)
_TD = re.compile(r"<td>(.*?)</td>", re.S)
for i, s in enumerate(tables):
    rows = _TR.findall(s["html"])
    row_cells = [len(_TD.findall(r)) for r in rows]
    cols_dist = Counter(row_cells)
    empty = s["html"].count("<td></td>")
    total = sum(row_cells)
    consistency = "✅" if len(cols_dist) == 1 else "⚠️列错位"
    empty_rate = f"{empty}/{total}={empty/total*100:.0f}%" if total else "n/a"
    print(f"表#{i} p{s['page']}: {len(rows)}行 列分布{dict(cols_dist)} {consistency} | 煤{empty_rate}")

parts = [
    "<html><head><meta charset='utf-8'><style>",
    "body{font-family:sans-serif;padding:20px} table{border-collapse:collapse;margin:8px 0} ",
    "td{border:1px solid #999;padding:4px 8px} .meta{color:#666;font-size:12px} ",
    ".plain{background:#f5f5f5;padding:6px;font-size:12px;white-space:pre-wrap;margin-bottom:16px}",
    "</style></head><body>",
    f"<h2>{Path(pdf_path).name} — 表格提取质量检查</h2>",
]
for i, s in enumerate(tables):
    parts.append(
        f"<div><div class='meta'>表#{i} (page {s['page']})</div>"
        f"{s['html']}"
        f"<div class='plain'>纯文本: {html.escape(s['text'])[:400]}</div></div>"
    )
parts.append("</body></html>")
out_html.write_text("\n".join(parts), encoding="utf-8")
print(f"\nHTML 渲染: {out_html}")
print("→ 浏览器打开，和原 PDF 并排对比看内容/结构是否正确")
