"""Doctor and report rendering."""

from __future__ import annotations

from pathlib import Path

from rl_preflight.receipt import PreflightSummary


def render_doctor(summary: PreflightSummary) -> str:
    status = "PASS" if summary.passed else "FAIL"
    lines = [
        f"RL Run Preflight — {summary.run_name} [{status}]",
        "=" * 60,
        f"run_id: {summary.run_id}  mode: {summary.mode}",
        "",
        "## Gates",
    ]
    for gate in summary.gates:
        mark = "PASS" if gate.passed else "FAIL"
        lines.append(f"  [{mark}] ({gate.severity}) {gate.name}: {gate.detail}")

    if summary.scenario_results:
        lines += ["", "## Mock scenarios"]
        for row in summary.scenario_results:
            flag = []
            if row.get("overflows"):
                flag.append("overflow")
            if row.get("masked_nan"):
                flag.append("masked_nan")
            suffix = f" [{', '.join(flag)}]" if flag else ""
            lines.append(
                f"  {row['name']}: log_ratio={row['log_ratio']:.2f}, ratio={row['ratio']}{suffix}"
            )

    return "\n".join(lines)


def doctor_path(path: Path) -> PreflightSummary:
    if path.is_dir():
        path = path / "summary.json"
    return PreflightSummary.load(path)


def write_markdown_report(summary_path: Path, report_path: Path) -> None:
    summary = PreflightSummary.load(summary_path)
    report_path.write_text(render_doctor(summary) + "\n", encoding="utf-8")
