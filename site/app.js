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

async function loadRun(key) {
  const meta = RUNS[key];
  const res = await fetch(meta.receiptPath);
  if (!res.ok) throw new Error(`Failed to load ${meta.receiptPath}`);
  const summary = await res.json();
  current = { meta, summary };
  render();
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
  const warns = summary.gates.filter((g) => g.severity === "warn").length;
  const scenarios = summary.scenario_results?.length ?? 0;

  document.getElementById("hero-metrics").innerHTML = `
    <article><span>gates</span><strong>${summary.gates.length}</strong></article>
    <article><span>errors</span><strong>${errors}</strong></article>
    <article><span>scenarios</span><strong>${scenarios}</strong></article>
  `;

  const doctor = document.getElementById("hero-doctor");
  doctor.innerHTML = `
    <div class="row"><span>gate</span><span>severity</span><span>result</span></div>
    ${summary.gates
      .map((g) => {
        const cls = g.passed ? (g.severity === "warn" ? "warn" : "ok") : "bad";
        const mark = g.passed ? "PASS" : "FAIL";
        return `<div class="row"><span>${g.name}</span><span>${g.severity}</span><span class="${cls}">${mark} — ${escapeHtml(g.detail)}</span></div>`;
      })
      .join("")}
  `;

  document.getElementById("demo-status").textContent = pass ? "PASS — safe to launch training" : "FAIL — fix config before GPU run";
  document.getElementById("demo-status").className = pass ? "ok" : "bad";

  const grid = document.getElementById("scenario-grid");
  grid.innerHTML = (summary.scenario_results || [])
    .map((s) => {
      const flags = [];
      if (s.overflows) flags.push("overflow");
      if (s.masked_nan) flags.push("masked_nan");
      const ratio = Number.isFinite(s.ratio) ? s.ratio.toFixed(2) : "inf";
      return `<article class="scenario-card"><div><strong>${escapeHtml(s.name)}</strong><div>log_ratio=${s.log_ratio.toFixed(2)} · ratio=${ratio}</div></div><div class="flags">${flags.join(", ") || "ok"}</div></article>`;
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

document.getElementById("run-select").addEventListener("change", (e) => {
  loadRun(e.target.value).catch((err) => {
    document.getElementById("demo-error").textContent = err.message;
  });
});

document.getElementById("run-btn").addEventListener("click", () => {
  const key = document.getElementById("run-select").value;
  loadRun(key).catch((err) => {
    document.getElementById("demo-error").textContent = err.message;
  });
});

loadRun("fail").catch((err) => {
  document.getElementById("demo-error").textContent = err.message;
});
