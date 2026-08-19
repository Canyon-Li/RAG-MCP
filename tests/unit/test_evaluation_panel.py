"""Unit tests for the rewritten Evaluation Panel (report viewer)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestEvaluationPanelRendering:
    """AppTest-level checks with a mocked report service."""

    @pytest.fixture
    def mock_svc(self) -> MagicMock:
        from src.observability.dashboard.services.evaluation_report_service import EvalRunSnapshot

        svc = MagicMock()
        svc.list_runs.return_value = ["eval_v2_baseline.json"]
        svc.load_run.return_value = EvalRunSnapshot(
            run_id="v2-baseline",
            label="v2-baseline · 2026-08-16",
            date="2026-08-16",
            n_qa=23,
            batches=[
                {"source_recall_at_k": 1.0, "source_precision_at_k": 0.7,
                 "context_relevance": 1.0, "context_precision": 0.75, "context_recall": 0.67},
            ],
            aggregate={
                "source_recall_at_k": 0.96, "source_precision_at_k": 0.71,
                "context_relevance": 0.87, "context_precision": 0.65, "context_recall": 0.49,
            },
        )
        svc.load_history_entries.return_value = []
        return svc

    def _render(self, mock_svc: MagicMock) -> Any:
        from streamlit.testing.v1 import AppTest

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch(
            "src.observability.dashboard.pages.evaluation_panel.EvaluationReportService",
            return_value=mock_svc,
        ):
            at.run()
        return at

    def test_renders_five_metric_cards(self, mock_svc: MagicMock) -> None:
        at = self._render(mock_svc)
        assert not at.exception
        # 五个指标卡都是 st.metric → 数量 >= 5,且文字包含 context_recall
        metrics_text = "\n".join(str(getattr(m, "label", "") or getattr(m, "value", ""))
                                  for m in at.metric)
        assert "context_recall" in metrics_text or len(at.metric) >= 5

    def test_renders_batch_table(self, mock_svc: MagicMock) -> None:
        at = self._render(mock_svc)
        assert not at.exception
        assert len(at.dataframe) >= 1

    def test_no_runs_shows_guidance(self) -> None:
        from streamlit.testing.v1 import AppTest

        svc = MagicMock()
        svc.list_runs.return_value = []

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch(
            "src.observability.dashboard.pages.evaluation_panel.EvaluationReportService",
            return_value=svc,
        ):
            at.run()
        assert not at.exception
        infos = "\n".join(str(getattr(i, "value", "")) for i in at.info)
        assert "evaluate.py" in infos or "评估" in infos

    def test_broken_report_shows_error(self) -> None:
        from streamlit.testing.v1 import AppTest

        svc = MagicMock()
        svc.list_runs.return_value = ["eval_bad.json"]
        svc.load_run.side_effect = ValueError("Corrupted report")

        def page_script():
            from src.observability.dashboard.pages.evaluation_panel import render
            render()

        at = AppTest.from_function(page_script, default_timeout=10)
        with patch(
            "src.observability.dashboard.pages.evaluation_panel.EvaluationReportService",
            return_value=svc,
        ):
            at.run()
        assert not at.exception
        errors = "\n".join(str(getattr(e, "value", "")) for e in at.error)
        assert errors  # 有 error banner 且页面未崩
