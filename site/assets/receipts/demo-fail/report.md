RL Run Preflight — overflow-risk-unbounded [FAIL]
============================================================
run_id: demo-fail  mode: mock

## Gates
  [PASS] (error) config_schema: framework='prime-rl', run='overflow-risk-unbounded'
  [FAIL] (error) importance_ratio_overflow: unbounded ratio overflows float32 for scenarios: single-token-blowup, masked-overflow-nan
  [FAIL] (error) masked_overflow_nan: masked tokens still produce NaN for: masked-overflow-nan
  [PASS] (warn) importance_ratio_cap: importance_ratio_max unset — stale async rollouts can hit unbounded exp()
  [PASS] (info) async_off_policy_lag: async_policy_lag=4, max_off_policy_lag=8
  [PASS] (warn) float32_matmul_precision: dtype=bfloat16 with float32_matmul_precision='high' — ROCm/large-vocab softmax may need 'highest'

## Mock scenarios
  single-token-blowup: log_ratio=89.00, ratio=inf [overflow]
  masked-overflow-nan: log_ratio=89.00, ratio=inf [overflow, masked_nan]
