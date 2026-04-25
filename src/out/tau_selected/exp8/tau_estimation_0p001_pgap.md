# Tau Estimation Summary

- Input: `src/out/experiment8/correlation_comparison_0p001.csv`
- Metric: `pgap`
- Objective: `balanced_accuracy`
- Estimated tau: `0.000000`
- Accuracy at tau: `0.6333`
- Balanced accuracy at tau: `0.6333`
- Macro F1 at tau: `0.5085`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.000186 | -0.000192 | -0.000208 | -0.000205 | -0.000171 | -0.000165 | -0.000217 | -0.000134 |
| random | 10 | 0.000226 | 0.000328 | -0.000658 | -0.000356 | 0.000591 | 0.001205 | -0.001123 | 0.001407 |
| positive | 10 | 0.035494 | 0.026205 | 0.004789 | 0.014966 | 0.044470 | 0.074938 | -0.000856 | 0.107895 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 3 | 0 | 7 |
| positive | 1 | 0 | 9 |
