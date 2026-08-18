"""Unit tests for dotenv loading in scripts/evaluate.py.

Verifies main() calls load_dotenv BEFORE constructing the evaluator, so
RAGAS_JUDGE_* / DEEPSEEK_API_KEY written in .env take effect at call time.
"""

from __future__ import annotations

from unittest.mock import patch


class TestEvaluateDotenv:
    def test_main_calls_load_dotenv_before_settings(self) -> None:
        """load_dotenv must run before load_settings (call order guard)."""
        calls = []

        def fake_dotenv() -> None:
            calls.append("dotenv")

        def fake_settings(*a, **kw):
            calls.append("settings")
            raise SystemExit(0)  # stop main() right after settings load

        # evaluate.py swaps sys.stdout/stderr for UTF-8 wrappers at import time
        # when sys.platform == "win32"; that clobbers pytest's capture pipes and
        # breaks teardown ("I/O operation on closed file"). Fake a non-win32
        # platform around the import so the module loads inertly under pytest.
        with patch("sys.platform", "linux"):
            import scripts.evaluate as evaluate_mod

        with patch.object(evaluate_mod, "__name__", "scripts.evaluate"), \
             patch("sys.argv", ["evaluate.py"]), \
             patch("dotenv.load_dotenv", side_effect=fake_dotenv) as mock_ld, \
             patch("src.core.settings.load_settings", side_effect=fake_settings):
            try:
                evaluate_mod.main()
            except SystemExit:
                pass

        mock_ld.assert_called_once()
        assert calls == ["dotenv", "settings"], (
            f"dotenv must load before settings, got {calls}"
        )
