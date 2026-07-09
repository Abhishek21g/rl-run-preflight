"""Execute mock preflight run."""

from __future__ import annotations

import uuid
from pathlib import Path

from rl_preflight.checks import run_checks
from rl_preflight.config import RunManifest, default_mock_scenarios
from rl_preflight.receipt import write_receipt_bundle
from rl_preflight.report import write_markdown_report


def execute_run(
    manifest: RunManifest,
    *,
    out_dir: Path,
    mock: bool = True,
    seed: int | None = None,
    run_id: str | None = None,
) -> Path:
    _ = seed  # reserved for future stochastic rollout fixtures
    if mock and not manifest.logprob_scenarios:
        manifest.logprob_scenarios = default_mock_scenarios()

    rid = run_id or uuid.uuid4().hex[:12]
    gates, scenario_results = run_checks(manifest)
    receipt_dir = write_receipt_bundle(
        out_dir,
        rid,
        manifest,
        gates,
        scenario_results,
        mode="mock" if mock else "live",
    )
    write_markdown_report(receipt_dir / "summary.json", receipt_dir / "report.md")
    return receipt_dir
