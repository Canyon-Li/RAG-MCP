"""Unit tests for the shared chunk_list rendering component."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


def _chunk(chunk_id: str, score: float, text: str = "body", source: str = "a.pdf") -> Dict[str, Any]:
    return {"chunk_id": chunk_id, "score": score, "text": text, "source": source, "title": "T"}


class TestRenderChunkList:
    """AppTest 冒烟 + highlight 逻辑(通过 expander 标题断言)。

    AppTest 的 expander 元素在不同 streamlit 版本上暴露 label/title 属性
    的方式不同——统一用 ``_expander_titles`` 收集,兼容两者。
    """

    def _run(self, chunks: List[Dict[str, Any]], highlight: Optional[set] = None) -> Any:
        from streamlit.testing.v1 import AppTest

        def page_script(chunks, highlight):
            from src.observability.dashboard.components.chunk_list import render_chunk_list
            render_chunk_list(chunks, prefix="t", highlight=highlight)

        at = AppTest.from_function(
            page_script,
            default_timeout=10,
            args=(chunks,),
            kwargs={"highlight": highlight},
        )
        at.run()
        return at

    @staticmethod
    def _expander_titles(at: Any) -> List[str]:
        titles = []
        for e in at.expander:
            for attr in ("label", "title"):
                val = getattr(e, attr, None)
                if val is not None:
                    titles.append(str(val))
                    break
        return titles

    def test_renders_without_exception(self) -> None:
        at = self._run([_chunk("c1", 0.9)])
        assert not at.exception

    def test_empty_list_no_exception(self) -> None:
        at = self._run([])
        assert not at.exception

    def test_highlight_hit_and_miss(self) -> None:
        at = self._run([_chunk("c1", 0.9), _chunk("c2", 0.1)], highlight={"c1"})
        assert not at.exception
        titles = self._expander_titles(at)
        assert any("🟢" in t for t in titles), f"hit chunk 应有 🟢 前缀: {titles}"
        assert any("⚪" in t for t in titles), f"miss chunk 应有 ⚪ 前缀: {titles}"

    def test_highlight_none_keeps_score_colours(self) -> None:
        at = self._run([_chunk("c1", 0.9), _chunk("c2", 0.1)], highlight=None)
        assert not at.exception
        titles = self._expander_titles(at)
        assert any("🟢" in t for t in titles)   # score>=0.8
        assert any("🔴" in t for t in titles)   # score<0.5
