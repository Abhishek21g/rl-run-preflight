"""Importance-ratio numerics (prime-rl#2972 class failures)."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any


# exp(x) overflows IEEE float32 near ln(FLT_MAX) ≈ 88.72
FLOAT32_LOG_RATIO_OVERFLOW = 88.0


def _as_float32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def log_importance_ratio(trainer_logprob: float, inference_logprob: float) -> float:
    return trainer_logprob - inference_logprob


def importance_ratio(trainer_logprob: float, inference_logprob: float) -> float:
    log_ratio = log_importance_ratio(trainer_logprob, inference_logprob)
    return _as_float32(math.exp(log_ratio))


def ratio_overflows_float32(trainer_logprob: float, inference_logprob: float) -> bool:
    log_ratio = log_importance_ratio(trainer_logprob, inference_logprob)
    return log_ratio > FLOAT32_LOG_RATIO_OVERFLOW or not math.isfinite(
        importance_ratio(trainer_logprob, inference_logprob)
    )


def masked_pg_term_is_nan(
    trainer_logprob: float,
    inference_logprob: float,
    *,
    in_loss_mask: bool,
    advantage: float,
) -> bool:
    """0 * inf can still become nan when a masked token overflows."""
    if in_loss_mask:
        return False
    ratio = importance_ratio(trainer_logprob, inference_logprob)
    if not math.isinf(ratio):
        return False
    term = _as_float32(0.0 * ratio * advantage)
    return math.isnan(term)


@dataclass
class ScenarioResult:
    name: str
    log_ratio: float
    ratio: float
    overflows: bool
    masked_nan: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "log_ratio": json_safe_float(self.log_ratio),
            "ratio": json_safe_float(self.ratio),
            "overflows": self.overflows,
            "masked_nan": self.masked_nan,
        }


def json_safe_float(value: float) -> float | str:
    """RFC 8259 JSON cannot represent inf/nan — use strings in receipts."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def json_dumps(data: Any, *, indent: int | None = None) -> str:
    return json.dumps(sanitize_for_json(data), indent=indent)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return json_safe_float(value)
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    return value


def evaluate_scenario(
    name: str,
    trainer_logprob: float,
    inference_logprob: float,
    *,
    in_loss_mask: bool,
    advantage: float,
) -> ScenarioResult:
    log_ratio = log_importance_ratio(trainer_logprob, inference_logprob)
    ratio = importance_ratio(trainer_logprob, inference_logprob)
    return ScenarioResult(
        name=name,
        log_ratio=log_ratio,
        ratio=ratio,
        overflows=ratio_overflows_float32(trainer_logprob, inference_logprob),
        masked_nan=masked_pg_term_is_nan(
            trainer_logprob,
            inference_logprob,
            in_loss_mask=in_loss_mask,
            advantage=advantage,
        ),
    )
