"""Tests for Data Quality Engine."""

import pytest
from unittest.mock import MagicMock, patch
from src.quality.data_quality_engine import DataQualityEngine, Severity, CheckResult


class TestDataQualityEngine:

    def test_check_result_defaults(self):
        r = CheckResult("test", "table", True, Severity.INFO, "ok")
        assert r.value == 0.0
        assert r.threshold == 0.0
        assert r.checked_at is not None

    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    @patch("src.quality.data_quality_engine.bigquery.Client")
    def test_run_suite_passes_with_no_failures(self, mock_bq):
        dq = DataQualityEngine("test-project")
        dq.results = [
            CheckResult("check1", "t", True, Severity.CRITICAL, "ok"),
            CheckResult("check2", "t", True, Severity.WARNING, "ok"),
        ]
        dq._log_results = MagicMock()
        dq._send_alert = MagicMock()

        dq.run_suite("t")  # Should not raise

        dq._send_alert.assert_not_called()

    @patch("src.quality.data_quality_engine.bigquery.Client")
    def test_run_suite_raises_on_critical_failure(self, mock_bq):
        dq = DataQualityEngine("test-project")
        dq.results = [
            CheckResult("check1", "t", False, Severity.CRITICAL, "schema drift"),
        ]
        dq._log_results = MagicMock()
        dq._send_alert = MagicMock()

        with pytest.raises(Exception, match="DATA QUALITY BLOCKED"):
            dq.run_suite("t")

    @patch("src.quality.data_quality_engine.bigquery.Client")
    def test_run_suite_warns_on_non_critical(self, mock_bq):
        dq = DataQualityEngine("test-project")
        dq.results = [
            CheckResult("check1", "t", True, Severity.CRITICAL, "ok"),
            CheckResult("vol", "t", False, Severity.WARNING, "z=3.5"),
        ]
        dq._log_results = MagicMock()
        dq._send_alert = MagicMock()

        dq.run_suite("t")  # Should not raise (warning only)
        dq._send_alert.assert_called_once()
