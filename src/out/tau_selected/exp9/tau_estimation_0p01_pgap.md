# Tau Estimation Summary

- Input: `src/out/experiment9/correlation_comparison_0p01.csv`
- Metric: `pgap`
- Objective: `balanced_accuracy`
- Estimated tau: `0.001352`
- Accuracy at tau: `0.9333`
- Balanced accuracy at tau: `0.9333`
- Macro F1 at tau: `0.9333`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.002170 | -0.002223 | -0.002440 | -0.002406 | -0.002005 | -0.001936 | -0.002473 | -0.001578 |
| random | 10 | 0.000282 | 0.000328 | -0.000658 | 0.000068 | 0.000591 | 0.001205 | -0.001123 | 0.001407 |
| positive | 10 | 0.039598 | 0.036175 | 0.004789 | 0.016550 | 0.051989 | 0.074938 | -0.000856 | 0.107895 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 9 | 1 |
| positive | 0 | 1 | 9 |
