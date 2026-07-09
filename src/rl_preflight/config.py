"""Parse and validate preflight run manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LogprobScenario:
    name: str
    trainer_logprob: float
    inference_logprob: float
    in_loss_mask: bool = True
    advantage: float = 1.0


@dataclass
class RunManifest:
    name: str
    framework: str = "prime-rl"
    dtype: str = "bfloat16"
    async_policy_lag: int = 1
    max_off_policy_lag: int = 8
    loss_type: str = "default"
    importance_ratio_max: float | None = 20.0
    kl_tau: float = 0.01
    float32_matmul_precision: str = "high"
    logprob_scenarios: list[LogprobScenario] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        run = data.get("run", {})
        training = data.get("training", {})
        loss = data.get("loss", {})
        mock = data.get("mock", {})

        scenarios: list[LogprobScenario] = []
        for item in mock.get("logprob_scenarios", []):
            scenarios.append(
                LogprobScenario(
                    name=str(item["name"]),
                    trainer_logprob=float(item["trainer_logprob"]),
                    inference_logprob=float(item["inference_logprob"]),
                    in_loss_mask=bool(item.get("in_loss_mask", True)),
                    advantage=float(item.get("advantage", 1.0)),
                )
            )

        ratio_max = loss.get("importance_ratio_max", 20.0)
        if ratio_max is None:
            ratio_max_val = None
        else:
            ratio_max_val = float(ratio_max)

        return cls(
            name=str(run.get("name", "unnamed")),
            framework=str(run.get("framework", "prime-rl")),
            dtype=str(training.get("dtype", "bfloat16")),
            async_policy_lag=int(training.get("async_policy_lag", 1)),
            max_off_policy_lag=int(training.get("max_off_policy_lag", 8)),
            loss_type=str(loss.get("type", "default")),
            importance_ratio_max=ratio_max_val,
            kl_tau=float(loss.get("kl_tau", 0.01)),
            float32_matmul_precision=str(training.get("float32_matmul_precision", "high")),
            logprob_scenarios=scenarios,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": {"name": self.name, "framework": self.framework},
            "training": {
                "dtype": self.dtype,
                "async_policy_lag": self.async_policy_lag,
                "max_off_policy_lag": self.max_off_policy_lag,
                "float32_matmul_precision": self.float32_matmul_precision,
            },
            "loss": {
                "type": self.loss_type,
                "importance_ratio_max": self.importance_ratio_max,
                "kl_tau": self.kl_tau,
            },
            "mock": {
                "logprob_scenarios": [
                    {
                        "name": s.name,
                        "trainer_logprob": s.trainer_logprob,
                        "inference_logprob": s.inference_logprob,
                        "in_loss_mask": s.in_loss_mask,
                        "advantage": s.advantage,
                    }
                    for s in self.logprob_scenarios
                ]
            },
        }


def load_manifest(path: Path) -> RunManifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a YAML mapping: {path}")
    return RunManifest.from_dict(data)


def default_mock_scenarios() -> list[LogprobScenario]:
    return [
        LogprobScenario("in-policy-token", -0.5, -0.6, True, 1.0),
        LogprobScenario("mild-off-policy", -2.0, -4.0, True, 1.0),
        LogprobScenario(
            "single-token-blowup",
            trainer_logprob=-2.0,
            inference_logprob=-91.0,
            in_loss_mask=True,
            advantage=1.0,
        ),
        LogprobScenario(
            "masked-overflow-nan",
            trainer_logprob=-2.0,
            inference_logprob=-91.0,
            in_loss_mask=False,
            advantage=1.0,
        ),
    ]
