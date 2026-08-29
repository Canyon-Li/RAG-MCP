"""Unit tests for reranker wiring in scripts/evaluate.py.

T15: the eval path reranks independently of the MCP query path —
EvalRunner._retrieve takes 2x candidates and truncates back to top_k, but
only when a CoreReranker is injected. These tests verify main() builds
that reranker from settings (when rerank.enabled) and passes it to
EvalRunner, and skips it cleanly when disabled.
"""

from __future__ import annotations

from unittest.mock import Mock, patch


def _run_main(settings: Mock) -> tuple:
    """Run evaluate.main() with heavy deps mocked.

    Stops at runner.run() via SystemExit. Returns
    (create_core_reranker_mock, EvalRunner_class_mock).
    """
    # evaluate.py swaps sys.stdout/stderr for UTF-8 wrappers at import time
    # on win32, clobbering pytest's capture — fake a non-win32 platform so
    # the module loads inertly (same trick as test_evaluate_dotenv.py).
    with patch("sys.platform", "linux"):
        import scripts.evaluate as evaluate_mod

    mock_reranker = Mock()
    mock_reranker.reranker_type = "cross_encoder"
    # __name__ patched so the module's `if __name__ == "__main__"` guard
    # (if any gets added) doesn't fire under re-import.
    with patch.object(evaluate_mod, "__name__", "scripts.evaluate"), \
         patch("sys.argv", ["evaluate.py", "--no-search"]), \
         patch("dotenv.load_dotenv"), \
         patch("src.core.settings.load_settings", return_value=settings), \
         patch("src.libs.evaluator.evaluator_factory.EvaluatorFactory.create") as mock_eval, \
         patch(
             "src.core.query_engine.reranker.create_core_reranker",
             return_value=mock_reranker,
         ) as mock_create, \
         patch("src.observability.evaluation.eval_runner.EvalRunner") as mock_runner_cls:
        mock_eval.return_value = Mock()
        mock_runner_cls.return_value.run.side_effect = SystemExit(0)
        try:
            evaluate_mod.main()
        except SystemExit:
            pass

    return mock_create, mock_runner_cls


def _make_settings(enabled: bool) -> Mock:
    settings = Mock()
    settings.rerank.enabled = enabled
    settings.rerank.provider = "cross_encoder" if enabled else "none"
    settings.rerank.model = "D:/models/bge-reranker-v2-m3" if enabled else "none"
    return settings


class TestEvaluateRerankerWiring:
    def test_enabled_builds_reranker_and_passes_to_eval_runner(self) -> None:
        """rerank.enabled=true → CoreReranker built from settings and injected."""
        settings = _make_settings(enabled=True)

        mock_create, mock_runner_cls = _run_main(settings)

        mock_create.assert_called_once_with(settings=settings)
        kwargs = mock_runner_cls.call_args.kwargs
        assert kwargs.get("reranker") is mock_create.return_value, (
            "EvalRunner must receive the CoreReranker built from settings"
        )

    def test_disabled_passes_none(self) -> None:
        """rerank.enabled=false → no CoreReranker built, reranker=None."""
        settings = _make_settings(enabled=False)

        mock_create, mock_runner_cls = _run_main(settings)

        mock_create.assert_not_called()
        kwargs = mock_runner_cls.call_args.kwargs
        assert kwargs.get("reranker") is None
