"""Shared chunk-list rendering component.

Extracted from ``query_traces._render_chunk_list`` so Playground and
Query Traces render chunks identically. ``highlight`` (a set of chunk_ids)
switches the header indicator from score-colours to intersection marking:
🟢 = in the highlight set, ⚪ = not. ``highlight=None`` keeps the original
score-based colouring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def render_chunk_list(
    chunks: List[Dict[str, Any]],
    *,
    prefix: str = "chunk",
    highlight: Optional[set] = None,
) -> None:
    """Render a list of chunk dicts as expandable cards.

    Args:
        chunks: List of ``{chunk_id, score, text, source, title}`` dicts.
        prefix: Unique Streamlit widget-key prefix per call site.
        highlight: Optional set of chunk_ids. When not None, header shows
            🟢 for members and ⚪ for non-members (intersection marking);
            when None, header shows score-based colours (🟢>=0.8 🟡>=0.5 🔴).
    """
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        if highlight is not None:
            indicator = "🟢" if chunk_id in highlight else "⚪"
        elif score >= 0.8:
            indicator = "🟢"
        elif score >= 0.5:
            indicator = "🟡"
        else:
            indicator = "🔴"

        header = f"{indicator} **#{ci + 1}** — Score: `{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")
            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption("_No text available_")
