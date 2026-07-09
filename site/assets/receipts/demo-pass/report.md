RL Run Preflight — intellect-style-grpo [PASS]
============================================================
run_id: demo-pass  mode: mock

## Gates
  [PASS] (error) config_schema: framework='prime-rl', run='intellect-style-grpo'
  [PASS] (info) importance_ratio_overflow: importance_ratio_max=20.0 configured — trainer should clip
  [PASS] (info) masked_overflow_nan: no masked-token NaN in mock scenarios
  [PASS] (info) importance_ratio_cap: cap=20.0 covers mock scenarios
  [PASS] (info) async_off_policy_lag: async_policy_lag=4, max_off_policy_lag=8
  [PASS] (info) float32_matmul_precision: float32_matmul_precision=highest

## Mock scenarios
  in-policy-token: log_ratio=0.10, ratio=1.1051709651947021
  mild-off-policy: log_ratio=2.00, ratio=7.389056205749512
  stale-but-clipped: log_ratio=89.00, ratio=inf [overflow]
