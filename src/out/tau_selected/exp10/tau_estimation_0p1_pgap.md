# Tau Estimation Summary

- Input: `src/out/experiment10/correlation_comparison_0p1.csv`
- Metric: `pgap`
- Objective: `balanced_accuracy`
- Estimated tau: `0.004521`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.019409 | -0.019366 | -0.024573 | -0.020206 | -0.016655 | -0.016080 | -0.027983 | -0.013109 |
| random | 10 | 0.000427 | -0.000205 | -0.000854 | -0.000556 | 0.001146 | 0.002377 | -0.001511 | 0.004127 |
| positive | 10 | 0.068503 | 0.060996 | 0.011477 | 0.030088 | 0.103386 | 0.120327 | -0.014123 | 0.180379 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 1 | 0 | 9 |
