# Tau Interpolation Summary

- Metric: `tanh`
- Polynomial degree: `2`
- Fitted on `4` selectivity points

## Fit Coefficients

`[-0.03652986870695888, -0.12008418389936595, -0.0020866309341351086]`

## Observed Points

| Selectivity | Tau | Accuracy | Balanced Accuracy | Macro F1 | Source |
|---:|---:|---:|---:|---:|---|
| 0.001000 | 0.031273 | 0.8333 | 0.8333 | 0.8151 | out/experiment8/tau/tau_estimation_0p001_summary.csv |
| 0.010000 | 0.086001 | 0.9667 | 0.9667 | 0.9666 | out/experiment7/tau/tau_estimation_0p01_summary.csv |
| 0.100000 | 0.088712 | 1.0000 | 1.0000 | 1.0000 | out/experiment5/tau/tau_estimation_0p1_summary.csv |
| 0.500000 | 0.027592 | 1.0000 | 1.0000 | 1.0000 | out/experiment6/tau/tau_estimation_0p5_summary.csv |

## Interpolated Table

| Selectivity | log10(Selectivity) | Interpolated Tau |
|---:|---:|---:|
| 0.001000 | -3.000000 | 0.029397 |
| 0.010000 | -2.000000 | 0.091962 |
| 0.100000 | -1.000000 | 0.081468 |
| 0.500000 | -0.301030 | 0.030752 |
