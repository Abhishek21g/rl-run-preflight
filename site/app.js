const BUNDLED = {
  pass: {
    schema_version: "1.0",
    run_id: "demo-pass",
    run_name: "intellect-style-grpo",
    framework: "prime-rl",
    mode: "mock",
    passed: true,
    gates: [
      { name: "config_schema", passed: true, severity: "error", detail: "framework='prime-rl', run='intellect-style-grpo'" },
      { name: "importance_ratio_overflow", passed: true, severity: "info", detail: "importance_ratio_max=20.0 configured — trainer should clip" },
      { name: "masked_overflow_nan", passed: true, severity: "info", detail: "no masked-token NaN in mock scenarios" },
      { name: "importance_ratio_cap", passed: true, severity: "info", detail: "cap=20.0 covers mock scenarios" },
      { name: "async_off_policy_lag", passed: true, severity: "info", detail: "async_policy_lag=4, max_off_policy_lag=8" },
      { name: "float32_matmul_precision", passed: true, severity: "info", detail: "float32_matmul_precision=highest" },
    ],
    scenario_results: [
      { name: "in-policy-token", log_ratio: 0.1, ratio: 1.105, overflows: false, masked_nan: false },
      { name: "mild-off-policy", log_ratio: 2.0, ratio: 7.389, overflows: false, masked_nan: false },
      { name: "stale-but-clipped", log_ratio: 89.0, ratio: "inf", overflows: true, masked_nan: false },
    ],
  },
  fail: {
    schema_version: "1.0",
    run_id: "demo-fail",
    run_name: "overflow-risk-unbounded",
    framework: "prime-rl",
    mode: "mock",
    passed: false,
    gates: [
      { name: "config_schema", passed: true, severity: "error", detail: "framework='prime-rl', run='overflow-risk-unbounded'" },
      { name: "importance_ratio_overflow", passed: false, severity: "error", detail: "unbounded ratio overflows float32 for scenarios: single-token-blowup, masked-overflow-nan" },
      { name: "masked_overflow_nan", passed: false, severity: "error", detail: "masked tokens still produce NaN for: masked-overflow-nan" },
      { name: "importance_ratio_cap", passed: true, severity: "warn", detail: "importance_ratio_max unset — stale async rollouts can hit unbounded exp()" },
      { name: "async_off_policy_lag", passed: true, severity: "info", detail: "async_policy_lag=4, max_off_policy_lag=8" },
      { name: "float32_matmul_precision", passed: true, severity: "warn", detail: "dtype=bfloat16 with float32_matmul_precision='high' — ROCm/large-vocab softmax may need 'highest'" },
    ],
    scenario_results: [
      { name: "single-token-blowup", log_ratio: 89.0, ratio: "inf", overflows: true, masked_nan: false },
      { name: "masked-overflow-nan", log_ratio: 89.0, ratio: "inf", overflows: true, masked_nan: true },
    ],
  },
};

const RUNS = {
  pass: {
    key: "pass",
    label: "intellect-style-grpo",
    receiptPath: "./assets/receipts/demo-pass/summary.json",
    command: "rl-preflight run examples/intellect-style-grpo.yaml --mock",
  },
  fail: {
    key: "fail",
    label: "overflow-risk-unbounded",
    receiptPath: "./assets/receipts/demo-fail/summary.json",
    command: "rl-preflight run examples/overflow-risk.yaml --mock",
  },
};

let current = null;

function formatRatio(ratio) {
  if (ratio === "inf" || ratio === "-inf") return ratio;
  const n = Number(ratio);
  return Number.isFinite(n) ? n.toFixed(2) : String(ratio);
}

function loadRun(key) {
  const meta = RUNS[key];
  document.getElementById("demo-error").textContent = "";
  document.getElementById("run-select").value = key;
  current = { meta, summary: structuredClone(BUNDLED[key]) };
  render();

  fetch(`${meta.receiptPath}?v=2`, { cache: "no-store" })
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((summary) => {
      current.summary = summary;
      render();
    })
    .catch(() => {
      /* bundled receipts are authoritative for the demo */
    });
}

function render() {
  if (!current) return;
  const { meta, summary } = current;
  const pass = summary.passed;

  document.getElementById("hero-run-label").textContent = summary.run_name;
  document.getElementById("hero-badge").textContent = pass ? "healthy" : "blocked";
  document.getElementById("hero-badge").className = `badge ${pass ? "pass" : "fail"}`;
  document.getElementById("hero-status-dot").className = `status-dot ${pass ? "pass" : "fail"}`;
  document.getElementById("hero-command").textContent = meta.command;

  const errors = summary.gates.filter((g) => g.severity === "error" && !g.passed).length;
  const scenarios = summary.scenario_results?.length ?? 0;

  document.getElementById("hero-metrics").innerHTML = `
    <article><span>gates</span><strong>${summary.gates.length}</strong></article>
    <article><span>errors</span><strong>${errors}</strong></article>
    <article><span>scenarios</span><strong>${scenarios}</strong></article>
  `;

  document.getElementById("hero-doctor").innerHTML = `
    <div class="row"><span>gate</span><span>severity</span><span>result</span></div>
    ${summary.gates
      .map((g) => {
        const cls = g.passed ? (g.severity === "warn" ? "warn" : "ok") : "bad";
        const mark = g.passed ? "PASS" : "FAIL";
        return `<div class="row"><span>${g.name}</span><span>${g.severity}</span><span class="${cls}">${mark} — ${escapeHtml(g.detail)}</span></div>`;
      })
      .join("")}
  `;

  const statusEl = document.getElementById("demo-status");
  statusEl.textContent = pass ? "PASS — safe to launch training" : "FAIL — fix config before GPU run";
  statusEl.className = pass ? "ok" : "bad";

  document.getElementById("scenario-grid").innerHTML = (summary.scenario_results || [])
    .map((s) => {
      const flags = [];
      if (s.overflows) flags.push("overflow");
      if (s.masked_nan) flags.push("masked_nan");
      return `<article class="scenario-card"><div><strong>${escapeHtml(s.name)}</strong><div>log_ratio=${Number(s.log_ratio).toFixed(2)} · ratio=${formatRatio(s.ratio)}</div></div><div class="flags">${flags.join(", ") || "ok"}</div></article>`;
    })
    .join("");

  document.getElementById("receipt-link").href = meta.receiptPath;
  document.getElementById("receipt-link").textContent = meta.receiptPath;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

document.getElementById("run-select").addEventListener("change", (e) => loadRun(e.target.value));
document.getElementById("run-btn").addEventListener("click", () => loadRun(document.getElementById("run-select").value));

loadRun("fail");
