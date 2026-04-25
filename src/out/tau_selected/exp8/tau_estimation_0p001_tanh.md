# Tau Estimation Summary

- Input: `src/out/experiment8/correlation_comparison_0p001.csv`
- Metric: `tanh`
- Objective: `balanced_accuracy`
- Estimated tau: `0.007354`
- Accuracy at tau: `0.7000`
- Balanced accuracy at tau: `0.7000`
- Macro F1 at tau: `0.6389`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.011318 | -0.011698 | -0.012680 | -0.012458 | -0.010385 | -0.010027 | -0.013241 | -0.008174 |
| random | 10 | 0.003356 | 0.006943 | -0.015850 | -0.008080 | 0.012038 | 0.022907 | -0.028213 | 0.026199 |
| positive | 10 | 0.219228 | 0.228380 | 0.062824 | 0.213200 | 0.275626 | 0.321011 | -0.014002 | 0.419150 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 3 | 2 | 5 |
| positive | 1 | 0 | 9 |
