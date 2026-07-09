"""Plan preflight checks for a manifest."""

from __future__ import annotations

from rl_preflight.checks import planned_checks
from rl_preflight.config import RunManifest


def build_plan(manifest: RunManifest) -> dict:
    scenarios = manifest.logprob_scenarios
    return {
        "run_name": manifest.name,
        "framework": manifest.framework,
        "checks": planned_checks(manifest),
        "mock_scenario_count": len(scenarios),
        "training": {
            "dtype": manifest.dtype,
            "async_policy_lag": manifest.async_policy_lag,
            "max_off_policy_lag": manifest.max_off_policy_lag,
        },
        "loss": {
            "type": manifest.loss_type,
            "importance_ratio_max": manifest.importance_ratio_max,
        },
    }
