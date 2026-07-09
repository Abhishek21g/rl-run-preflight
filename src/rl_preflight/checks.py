"""Preflight gate checks."""

from __future__ import annotations

from dataclasses import dataclass

from rl_preflight.config import LogprobScenario, RunManifest
from rl_preflight.numerics import evaluate_scenario


@dataclass
class GateResult:
    name: str
    passed: bool
    severity: str  # error | warn | info
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
        }


def planned_checks(manifest: RunManifest) -> list[str]:
    checks = [
        "config_schema",
        "importance_ratio_overflow",
        "masked_overflow_nan",
        "importance_ratio_cap",
        "async_off_policy_lag",
    ]
    if manifest.dtype in {"bfloat16", "float16"} and manifest.framework == "prime-rl":
        checks.append("float32_matmul_precision")
    return checks


def run_checks(manifest: RunManifest) -> tuple[list[GateResult], list[dict]]:
    scenarios = manifest.logprob_scenarios or []
    scenario_results = [
        evaluate_scenario(
            s.name,
            s.trainer_logprob,
            s.inference_logprob,
            in_loss_mask=s.in_loss_mask,
            advantage=s.advantage,
        ).to_dict()
        for s in scenarios
    ]

    gates: list[GateResult] = []
    gates.append(_check_config_schema(manifest))
    gates.append(_check_overflow_unbounded(manifest, scenarios))
    gates.append(_check_masked_nan(scenarios))
    gates.append(_check_ratio_cap(manifest, scenarios))
    gates.append(_check_async_lag(manifest))
    if manifest.dtype in {"bfloat16", "float16"}:
        gates.append(_check_matmul_precision(manifest))

    return gates, scenario_results


def _check_config_schema(manifest: RunManifest) -> GateResult:
    ok = bool(manifest.name) and manifest.framework in {"prime-rl", "verifiers"}
    return GateResult(
        name="config_schema",
        passed=ok,
        severity="error",
        detail=f"framework={manifest.framework!r}, run={manifest.name!r}",
    )


def _check_overflow_unbounded(
    manifest: RunManifest, scenarios: list[LogprobScenario]
) -> GateResult:
    if manifest.importance_ratio_max is not None:
        return GateResult(
            name="importance_ratio_overflow",
            passed=True,
            severity="info",
            detail=f"importance_ratio_max={manifest.importance_ratio_max} configured — trainer should clip",
        )

    offenders = [
        s.name
        for s in scenarios
        if evaluate_scenario(
            s.name,
            s.trainer_logprob,
            s.inference_logprob,
            in_loss_mask=s.in_loss_mask,
            advantage=s.advantage,
        ).overflows
    ]
    if offenders:
        return GateResult(
            name="importance_ratio_overflow",
            passed=False,
            severity="error",
            detail=f"unbounded ratio overflows float32 for scenarios: {', '.join(offenders)}",
        )
    return GateResult(
        name="importance_ratio_overflow",
        passed=True,
        severity="info",
        detail="no overflow in mock scenarios (unbounded ratio still risky in production)",
    )


def _check_masked_nan(scenarios: list[LogprobScenario]) -> GateResult:
    offenders = [
        s.name
        for s in scenarios
        if evaluate_scenario(
            s.name,
            s.trainer_logprob,
            s.inference_logprob,
            in_loss_mask=s.in_loss_mask,
            advantage=s.advantage,
        ).masked_nan
    ]
    if offenders:
        return GateResult(
            name="masked_overflow_nan",
            passed=False,
            severity="error",
            detail=f"masked tokens still produce NaN for: {', '.join(offenders)}",
        )
    return GateResult(
        name="masked_overflow_nan",
        passed=True,
        severity="info",
        detail="no masked-token NaN in mock scenarios",
    )


def _check_ratio_cap(manifest: RunManifest, scenarios: list[LogprobScenario]) -> GateResult:
    cap = manifest.importance_ratio_max
    if cap is None:
        return GateResult(
            name="importance_ratio_cap",
            passed=True,
            severity="warn",
            detail="importance_ratio_max unset — stale async rollouts can hit unbounded exp()",
        )

    exceed = []
    for s in scenarios:
        result = evaluate_scenario(
            s.name,
            s.trainer_logprob,
            s.inference_logprob,
            in_loss_mask=s.in_loss_mask,
            advantage=s.advantage,
        )
        if math_isfinite(result.ratio) and result.ratio > cap:
            exceed.append(f"{s.name}({result.ratio:.1f}>{cap})")

    if exceed:
        return GateResult(
            name="importance_ratio_cap",
            passed=True,
            severity="warn",
            detail=f"scenarios exceed cap but should be clipped at train time: {', '.join(exceed)}",
        )
    return GateResult(
        name="importance_ratio_cap",
        passed=True,
        severity="info",
        detail=f"cap={cap} covers mock scenarios",
    )


def _check_async_lag(manifest: RunManifest) -> GateResult:
    lag = manifest.async_policy_lag
    max_lag = manifest.max_off_policy_lag
    if lag > max_lag:
        return GateResult(
            name="async_off_policy_lag",
            passed=False,
            severity="error",
            detail=f"async_policy_lag={lag} exceeds max_off_policy_lag={max_lag}",
        )
    if lag > 16:
        return GateResult(
            name="async_off_policy_lag",
            passed=True,
            severity="warn",
            detail=f"async_policy_lag={lag} > 16 — prime-rl notes training instability beyond ~16",
        )
    return GateResult(
        name="async_off_policy_lag",
        passed=True,
        severity="info",
        detail=f"async_policy_lag={lag}, max_off_policy_lag={max_lag}",
    )


def _check_matmul_precision(manifest: RunManifest) -> GateResult:
    prec = manifest.float32_matmul_precision
    if prec != "highest":
        return GateResult(
            name="float32_matmul_precision",
            passed=True,
            severity="warn",
            detail=f"dtype={manifest.dtype} with float32_matmul_precision={prec!r} — ROCm/large-vocab softmax may need 'highest'",
        )
    return GateResult(
        name="float32_matmul_precision",
        passed=True,
        severity="info",
        detail="float32_matmul_precision=highest",
    )


def math_isfinite(x: float) -> bool:
    return x == x and x not in {float("inf"), float("-inf")}


def overall_pass(gates: list[GateResult]) -> bool:
    return all(g.passed for g in gates if g.severity == "error")
