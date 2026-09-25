"""EvaluationReportService – read-only parsing of eval report files.

Reads v2 standalone reports (``logs/eval_*.json``: run/date/n_qa/batches/
weighted_aggregate) and CLI history (``logs/eval_history.jsonl``, one
report dict per line). Produces ``EvalRunSnapshot`` dataclasses; pages do
zero file IO. v1 reports missing ``context_recall`` surface as None.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

HISTORY_RUN_ID = "__history__"
# T20: standard ragas 4-metric era. The v4.0-era keys stay registered so
# historical runs still parse; runs simply surface None for absent metrics.
METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "source_recall_at_k",
    "source_precision_at_k",
    "context_relevance",
)


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    description: str


METRIC_SPECS: List[MetricSpec] = [
    MetricSpec("faithfulness", "faithfulness", "回答是否忠于检索到的上下文（评分模型）"),
    MetricSpec("answer_relevancy", "answer_relevancy", "回答是否切题（评分模型 + 本地 embedding）"),
    MetricSpec("context_precision", "context_precision", "相关 chunk 是否排在前面（评分模型）"),
    MetricSpec("context_recall", "context_recall", "该召回的信息,检索结果覆盖了吗（评分模型）"),
    MetricSpec("source_recall_at_k", "source_recall@5", "top-5 里有没有命中正确论文（v4.0 退役）"),
    MetricSpec("source_precision_at_k", "source_precision@5", "top-5 里命中正确论文的比例（v4.0 退役）"),
    MetricSpec("context_relevance", "context_relevance", "检索回的上下文和问题相关吗（v4.0 退役）"),
]


@dataclass
class EvalRunSnapshot:
    run_id: str
    label: str
    date: str = ""
    n_qa: Optional[int] = None
    batches: List[Dict[str, Optional[float]]] = field(default_factory=list)
    aggregate: Dict[str, Optional[float]] = field(default_factory=dict)


def _fill_metrics(raw: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Map a metrics dict onto the five fixed keys; missing → None."""
    raw = raw or {}
    return {k: (float(raw[k]) if raw.get(k) is not None else None) for k in METRIC_KEYS}


class EvaluationReportService:
    """Read-only accessor for evaluation reports under ``logs/``."""

    def __init__(self, logs_dir: Optional[Path] = None) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir else resolve_path("logs")

    def list_runs(self) -> List[str]:
        """Standalone reports newest-first; history file (if any) last."""
        runs: List[str] = [
            p.name for p in sorted(
                self._logs_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ]
        if (self._logs_dir / "eval_history.jsonl").exists():
            runs.append(HISTORY_RUN_ID)
        return runs

    def load_run(self, run_id: str) -> EvalRunSnapshot:
        """Load one standalone report by filename.

        Raises:
            KeyError: Unknown run_id.
            ValueError: File is not valid JSON.
        """
        path = self._logs_dir / run_id
        if run_id == HISTORY_RUN_ID or not path.exists():
            raise KeyError(f"Unknown run: {run_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted report {run_id}: {exc}") from exc

        return EvalRunSnapshot(
            run_id=str(data.get("run", run_id)),
            label=f"{data.get('run', run_id)} · {data.get('date', '')}",
            date=str(data.get("date", "")),
            n_qa=data.get("n_qa"),
            batches=[_fill_metrics(b) for b in data.get("batches") or []],
            aggregate=_fill_metrics(data.get("weighted_aggregate")),
        )

    def load_history_entries(self) -> List[EvalRunSnapshot]:
        """One snapshot per jsonl line; empty list when file absent."""
        path = self._logs_dir / "eval_history.jsonl"
        if not path.exists():
            return []
        snapshots: List[EvalRunSnapshot] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupted history line: %.60s", line)
                continue
            snapshots.append(EvalRunSnapshot(
                run_id=str(data.get("timestamp", "history")),
                label=str(data.get("timestamp", "history")),
                date=str(data.get("timestamp", ""))[:10],
                n_qa=data.get("query_count"),
                batches=[_fill_metrics(qr.get("metrics")) for qr in data.get("query_results") or []],
                aggregate=_fill_metrics(data.get("aggregate_metrics")),
            ))
        return snapshots
