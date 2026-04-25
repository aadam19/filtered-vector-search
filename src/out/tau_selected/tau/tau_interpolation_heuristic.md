# Tau Interpolation Summary

- Metric: `heuristic`
- Polynomial degree: `3`
- Fitted on `4` selectivity points

## Fit Coefficients

`[-1.4254632513028434e-17, -7.149228635057322e-17, -8.872917584073394e-17, 0.10438473561764547]`

## Observed Points

| Selectivity | Tau | Accuracy | Balanced Accuracy | Macro F1 | Source |
|---:|---:|---:|---:|---:|---|
| 0.001000 | 0.104385 | 0.9667 | 0.9667 | 0.9666 | src/out/tau_selected/exp8/tau_estimation_0p001_summary.csv |
| 0.010000 | 0.104385 | 0.9667 | 0.9667 | 0.9666 | src/out/tau_selected/exp9/tau_estimation_0p01_summary.csv |
| 0.100000 | 0.104385 | 0.9667 | 0.9667 | 0.9666 | src/out/tau_selected/exp10/tau_estimation_0p1_summary.csv |
| 0.500000 | 0.104385 | 0.9667 | 0.9667 | 0.9666 | src/out/tau_selected/exp11/tau_estimation_0p5_summary.csv |

## Interpolated Table

| Selectivity | log10(Selectivity) | Interpolated Tau |
|---:|---:|---:|
| 0.001000 | -3.000000 | 0.104385 |
| 0.010000 | -2.000000 | 0.104385 |
| 0.100000 | -1.000000 | 0.104385 |
