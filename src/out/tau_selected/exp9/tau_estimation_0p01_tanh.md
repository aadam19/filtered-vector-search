# Tau Estimation Summary

- Input: `src/out/experiment9/correlation_comparison_0p01.csv`
- Metric: `tanh`
- Objective: `balanced_accuracy`
- Estimated tau: `0.028364`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.045813 | -0.044712 | -0.061448 | -0.046652 | -0.038453 | -0.037126 | -0.069977 | -0.030267 |
| random | 10 | 0.004561 | 0.006943 | -0.015850 | 0.000957 | 0.012038 | 0.022907 | -0.028213 | 0.026199 |
| positive | 10 | 0.219329 | 0.222173 | 0.062824 | 0.207293 | 0.280951 | 0.321011 | -0.014002 | 0.419150 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 0 | 1 | 9 |
