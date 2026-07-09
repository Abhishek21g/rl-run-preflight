# RL Run Preflight Workbench

Validate **prime-rl** / **verifiers** GRPO configs before a multi-hour GPU run.

Catches importance-ratio overflow (#2972-class), masked-token NaNs, and async off-policy lag misconfigs — in **mock mode**, no cluster required.

## Quick start

```bash
pip install -e ".[dev]"
rl-preflight plan examples/intellect-style-grpo.yaml
rl-preflight run examples/intellect-style-grpo.yaml --mock
rl-preflight doctor out/receipts/latest/summary.json
rl-preflight report out/receipts/latest/ --format md
```

## CLI

| Command | Purpose |
|---------|---------|
| `plan` | List checks for a manifest |
| `run` | Mock preflight → `out/receipts/<run-id>/` |
| `doctor` | Pass/fail summary from receipt |
| `report` | Markdown or JSON report |

## Receipt layout

```
out/receipts/<run-id>/
  manifest.json
  summary.json
  report.md
out/receipts/latest -> <run-id>
```

## Demo

Open `site/index.html` or deploy to `enaguthi.com/rl-run-preflight/site/`.

**Not affiliated with Prime Intellect.** Independent tool inspired by public prime-rl issues and INTELLECT-2 async RL docs.
