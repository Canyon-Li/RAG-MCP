"""Unit tests for EvaluationReportService (read-only report parsing)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest


def _write_v2(logs: Path) -> None:
    logs.joinpath("eval_v2_baseline.json").write_text(json.dumps({
        "run": "v2-baseline",
        "date": "2026-08-16",
        "n_qa": 23,
        "batches": [
            {"context_precision": 0.75, "context_recall": 0.67, "context_relevance": 1.0,
             "source_precision_at_k": 0.7, "source_recall_at_k": 1.0},
        ],
        "weighted_aggregate": {
            "context_precision": 0.65, "context_recall": 0.49, "context_relevance": 0.87,
            "source_precision_at_k": 0.71, "source_recall_at_k": 0.96,
        },
    }, ensure_ascii=False), encoding="utf-8")


def _write_v1(logs: Path) -> None:
    """v1 report: aggregate only, missing context_recall key."""
    logs.joinpath("eval_v1_old.json").write_text(json.dumps({
        "run": "v1-old",
        "date": "2026-07-01",
        "n_qa": 10,
        "batches": [],
        "weighted_aggregate": {
            "context_precision": 0.6, "context_relevance": 0.9,
            "source_precision_at_k": 0.7, "source_recall_at_k": 1.0,
        },
    }, ensure_ascii=False), encoding="utf-8")


class TestListRuns:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        svc = EvaluationReportService(logs_dir=tmp_path)
        assert svc.list_runs() == []

    def test_v2_files_sorted_latest_first(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        old = tmp_path / "eval_a.json"
        new = tmp_path / "eval_b.json"
        old.write_text("{}", encoding="utf-8")
        new.write_text("{}", encoding="utf-8")
        import os
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        svc = EvaluationReportService(logs_dir=tmp_path)
        assert svc.list_runs() == ["eval_b.json", "eval_a.json"]

    def test_history_appended_last(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        (tmp_path / "eval_x.json").write_text("{}", encoding="utf-8")
        (tmp_path / "eval_history.jsonl").write_text("\n", encoding="utf-8")
        svc = EvaluationReportService(logs_dir=tmp_path)
        assert svc.list_runs()[-1] == "__history__"


class TestLoadRun:
    def test_v2_full(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        _write_v2(tmp_path)
        svc = EvaluationReportService(logs_dir=tmp_path)
        snap = svc.load_run("eval_v2_baseline.json")
        assert snap.run_id == "v2-baseline"
        assert snap.n_qa == 23
        assert len(snap.batches) == 1
        assert snap.aggregate["context_recall"] == pytest.approx(0.49)
        # 五指标键全在
        for key in ("source_recall_at_k", "source_precision_at_k",
                    "context_relevance", "context_precision", "context_recall"):
            assert key in snap.aggregate

    def test_v1_missing_metric_is_none(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        _write_v1(tmp_path)
        svc = EvaluationReportService(logs_dir=tmp_path)
        snap = svc.load_run("eval_v1_old.json")
        assert snap.aggregate["context_recall"] is None
        assert snap.aggregate["context_precision"] == pytest.approx(0.6)

    def test_broken_json_raises_valueerror(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        (tmp_path / "eval_bad.json").write_text("{not json", encoding="utf-8")
        svc = EvaluationReportService(logs_dir=tmp_path)
        with pytest.raises(ValueError):
            svc.load_run("eval_bad.json")

    def test_unknown_run_raises_keyerror(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        svc = EvaluationReportService(logs_dir=tmp_path)
        with pytest.raises(KeyError):
            svc.load_run("nope.json")


class TestHistoryEntries:
    def test_history_line_to_snapshot(self, tmp_path: Path) -> None:
        from src.observability.dashboard.services.evaluation_report_service import (
            EvaluationReportService,
        )
        entry = {
            "timestamp": "2026-08-16 20:40:44",
            "aggregate_metrics": {
                "context_precision": 0.5, "context_recall": 0.25, "context_relevance": 1.0,
                "source_precision_at_k": 1.0, "source_recall_at_k": 1.0,
            },
        }
        (tmp_path / "eval_history.jsonl").write_text(
            json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        svc = EvaluationReportService(logs_dir=tmp_path)
        runs = svc.load_history_entries()
        assert len(runs) == 1
        assert runs[0].aggregate["context_recall"] == pytest.approx(0.25)
        assert runs[0].batches == []          # history 行无 batches
