"""Receipt artifacts under out/receipts/<run-id>/."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_preflight.checks import GateResult, overall_pass
from rl_preflight.config import RunManifest


@dataclass
class PreflightSummary:
    schema_version: str = "1.0"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    run_id: str = ""
    run_name: str = ""
    framework: str = "prime-rl"
    mode: str = "mock"
    passed: bool = False
    gates: list[GateResult] = field(default_factory=list)
    scenario_results: list[dict] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "framework": self.framework,
            "mode": self.mode,
            "passed": self.passed,
            "gates": [g.to_dict() for g in self.gates],
            "scenario_results": self.scenario_results,
            "manifest": self.manifest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreflightSummary:
        gates = [GateResult(**g) for g in data.get("gates", [])]
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            generated_at=data.get("generated_at", ""),
            run_id=data.get("run_id", ""),
            run_name=data.get("run_name", ""),
            framework=data.get("framework", "prime-rl"),
            mode=data.get("mode", "mock"),
            passed=bool(data.get("passed", False)),
            gates=gates,
            scenario_results=list(data.get("scenario_results", [])),
            manifest=dict(data.get("manifest", {})),
        )

    @classmethod
    def load(cls, path: Path) -> PreflightSummary:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def write_receipt_bundle(
    out_dir: Path,
    run_id: str,
    manifest: RunManifest,
    gates: list[GateResult],
    scenario_results: list[dict],
    *,
    mode: str = "mock",
) -> Path:
    receipt_dir = out_dir / "receipts" / run_id
    receipt_dir.mkdir(parents=True, exist_ok=True)

    summary = PreflightSummary(
        run_id=run_id,
        run_name=manifest.name,
        framework=manifest.framework,
        mode=mode,
        passed=overall_pass(gates),
        gates=gates,
        scenario_results=scenario_results,
        manifest=manifest.to_dict(),
    )
    summary_path = receipt_dir / "summary.json"
    summary.save(summary_path)

    manifest_path = receipt_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_path": manifest.name,
                "checks": [g.name for g in gates],
                "artifacts": {
                    "summary": "summary.json",
                    "report_md": "report.md",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latest = out_dir / "receipts" / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(receipt_dir.name)

    return receipt_dir
