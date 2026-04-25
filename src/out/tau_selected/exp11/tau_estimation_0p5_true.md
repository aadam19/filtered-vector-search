# Tau Estimation Summary

- Input: `src/out/experiment11/correlation_comparison_0p5.csv`
- Metric: `true`
- Objective: `balanced_accuracy`
- Estimated tau: `0.019288`
- Accuracy at tau: `1.0000`
- Balanced accuracy at tau: `1.0000`
- Macro F1 at tau: `1.0000`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.401482 | -0.467550 | -0.538304 | -0.521023 | -0.308510 | -0.182634 | -0.549715 | -0.136813 |
| random | 10 | 0.001231 | -0.000023 | -0.008284 | -0.004352 | 0.006309 | 0.008669 | -0.010408 | 0.018852 |
| positive | 10 | 0.056206 | 0.058385 | 0.036860 | 0.043726 | 0.066516 | 0.073979 | 0.033198 | 0.077981 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 0 | 0 | 10 |
