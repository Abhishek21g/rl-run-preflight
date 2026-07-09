import json
import math
from pathlib import Path

import pytest

from rl_preflight.checks import overall_pass, run_checks
from rl_preflight.config import LogprobScenario, RunManifest, load_manifest
from rl_preflight.numerics import (
    FLOAT32_LOG_RATIO_OVERFLOW,
    importance_ratio,
    masked_pg_term_is_nan,
    ratio_overflows_float32,
)
from rl_preflight.plan import build_plan
from rl_preflight.run import execute_run


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_log_ratio_overflow_threshold():
    assert ratio_overflows_float32(-2.0, -91.0)
    assert not ratio_overflows_float32(-2.0, -4.0)


def test_importance_ratio_finite_mild_off_policy():
    ratio = importance_ratio(-2.0, -4.0)
    assert math.isfinite(ratio)


def test_importance_ratio_overflows_at_issue_example():
    ratio = importance_ratio(-2.0, -91.0)
    assert math.isinf(ratio)


def test_masked_token_nan_on_overflow():
    assert masked_pg_term_is_nan(-2.0, -91.0, in_loss_mask=False, advantage=1.0)


def test_masked_token_no_nan_when_in_loss_mask():
    assert not masked_pg_term_is_nan(-2.0, -91.0, in_loss_mask=True, advantage=1.0)


def test_float32_overflow_constant():
    assert FLOAT32_LOG_RATIO_OVERFLOW == 88.0


def test_load_intellect_manifest():
    manifest = load_manifest(EXAMPLES / "intellect-style-grpo.yaml")
    assert manifest.name == "intellect-style-grpo"
    assert manifest.importance_ratio_max == 20.0
    assert len(manifest.logprob_scenarios) == 3


def test_plan_lists_core_checks():
    manifest = load_manifest(EXAMPLES / "intellect-style-grpo.yaml")
    plan = build_plan(manifest)
    assert "importance_ratio_overflow" in plan["checks"]
    assert "async_off_policy_lag" in plan["checks"]


def test_safe_manifest_passes():
    manifest = load_manifest(EXAMPLES / "intellect-style-grpo.yaml")
    gates, _ = run_checks(manifest)
    assert overall_pass(gates)


def test_overflow_manifest_fails():
    manifest = load_manifest(EXAMPLES / "overflow-risk.yaml")
    gates, _ = run_checks(manifest)
    assert not overall_pass(gates)


def test_async_lag_misconfig_fails():
    manifest = RunManifest(
        name="bad-lag",
        async_policy_lag=12,
        max_off_policy_lag=8,
        importance_ratio_max=20.0,
        logprob_scenarios=[LogprobScenario("ok", -1.0, -1.1, True)],
    )
    gates, _ = run_checks(manifest)
    assert not overall_pass(gates)


def test_run_writes_receipt_bundle(tmp_path):
    manifest = load_manifest(EXAMPLES / "intellect-style-grpo.yaml")
    receipt_dir = execute_run(manifest, out_dir=tmp_path, run_id="testrun01")
    assert (receipt_dir / "summary.json").exists()
    assert (receipt_dir / "manifest.json").exists()
    assert (receipt_dir / "report.md").exists()
    latest = tmp_path / "receipts" / "latest"
    assert latest.is_symlink()


def test_summary_json_schema(tmp_path):
    manifest = load_manifest(EXAMPLES / "intellect-style-grpo.yaml")
    receipt_dir = execute_run(manifest, out_dir=tmp_path, run_id="schema01")
    data = json.loads((receipt_dir / "summary.json").read_text())
    assert data["run_id"] == "schema01"
    assert "gates" in data
    assert data["passed"] is True


def test_default_mock_scenarios_used_when_empty(tmp_path):
    manifest = RunManifest(name="empty-mock", importance_ratio_max=20.0)
    receipt_dir = execute_run(manifest, out_dir=tmp_path, run_id="mockdefault")
    summary = json.loads((receipt_dir / "summary.json").read_text())
    assert len(summary["scenario_results"]) >= 4


def test_overflow_run_exits_nonzero_via_cli(tmp_path):
    from rl_preflight.cli import main

    code = main(
        [
            "run",
            str(EXAMPLES / "overflow-risk.yaml"),
            "--out",
            str(tmp_path),
            "--run-id",
            "fail01",
        ]
    )
    assert code == 1


def test_plan_json_output(capsys):
    from rl_preflight.cli import main

    code = main(["plan", str(EXAMPLES / "intellect-style-grpo.yaml"), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_name"] == "intellect-style-grpo"


def test_doctor_command(capsys, tmp_path):
    from rl_preflight.cli import main

    manifest = load_manifest(EXAMPLES / "intellect-style-grpo.yaml")
    receipt_dir = execute_run(manifest, out_dir=tmp_path, run_id="doc01")
    code = main(["doctor", str(receipt_dir)])
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_report_markdown(capsys, tmp_path):
    from rl_preflight.cli import main

    manifest = load_manifest(EXAMPLES / "intellect-style-grpo.yaml")
    receipt_dir = execute_run(manifest, out_dir=tmp_path, run_id="rep01")
    code = main(["report", str(receipt_dir), "--format", "md"])
    assert code == 0
    assert "Gates" in capsys.readouterr().out
