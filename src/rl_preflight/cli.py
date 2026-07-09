"""RL Run Preflight CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rl_preflight.config import load_manifest
from rl_preflight.plan import build_plan
from rl_preflight.report import doctor_path, render_doctor, write_markdown_report
from rl_preflight.run import execute_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rl-preflight",
        description="Preflight prime-rl / verifiers GRPO configs before GPU training.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan_p = sub.add_parser("plan", help="List checks that will run for a manifest")
    plan_p.add_argument("config", type=Path)
    plan_p.add_argument("--json", action="store_true")
    plan_p.set_defaults(handler=_cmd_plan)

    run_p = sub.add_parser("run", help="Run mock preflight and write receipts")
    run_p.add_argument("config", type=Path)
    run_p.add_argument("--mock", action="store_true", default=True)
    run_p.add_argument("--out", type=Path, default=Path("out"))
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--run-id", type=str, default=None)
    run_p.add_argument("--json", action="store_true")
    run_p.set_defaults(handler=_cmd_run)

    doc_p = sub.add_parser("doctor", help="Summarize a receipt directory or summary.json")
    doc_p.add_argument("target", type=Path)
    doc_p.add_argument("--json", action="store_true")
    doc_p.set_defaults(handler=_cmd_doctor)

    rep_p = sub.add_parser("report", help="Write or print Markdown report from a receipt")
    rep_p.add_argument("target", type=Path)
    rep_p.add_argument("--format", choices=["md", "json"], default="md")
    rep_p.add_argument("-o", "--output", type=Path, default=None)
    rep_p.set_defaults(handler=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_plan(args) -> int:
    manifest = load_manifest(args.config)
    plan = build_plan(manifest)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"Plan: {plan['run_name']} ({plan['framework']})")
        print("Checks:", ", ".join(plan["checks"]))
        print(f"Mock scenarios: {plan['mock_scenario_count']}")
    return 0


def _cmd_run(args) -> int:
    manifest = load_manifest(args.config)
    receipt_dir = execute_run(
        manifest,
        out_dir=args.out,
        mock=args.mock,
        seed=args.seed,
        run_id=args.run_id,
    )
    summary = doctor_path(receipt_dir)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(render_doctor(summary))
        print(f"\nWrote {receipt_dir}")
    return 0 if summary.passed else 1


def _cmd_doctor(args) -> int:
    summary = doctor_path(args.target)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(render_doctor(summary))
    return 0 if summary.passed else 1


def _cmd_report(args) -> int:
    summary_path = args.target / "summary.json" if args.target.is_dir() else args.target
    summary = doctor_path(summary_path)
    if args.format == "json":
        payload = json.dumps(summary.to_dict(), indent=2)
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload)
    else:
        md = render_doctor(summary) + "\n"
        if args.output:
            args.output.write_text(md, encoding="utf-8")
        else:
            print(md, end="")
        if args.target.is_dir():
            write_markdown_report(summary_path, args.target / "report.md")
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
