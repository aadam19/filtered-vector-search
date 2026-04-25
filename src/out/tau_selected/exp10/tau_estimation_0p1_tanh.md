# Tau Estimation Summary

- Input: `src/out/experiment10/correlation_comparison_0p1.csv`
- Metric: `tanh`
- Objective: `balanced_accuracy`
- Estimated tau: `0.026088`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.121282 | -0.099279 | -0.228552 | -0.103585 | -0.085382 | -0.082435 | -0.260275 | -0.067205 |
| random | 10 | 0.001822 | -0.001531 | -0.006444 | -0.004133 | 0.006161 | 0.012696 | -0.011745 | 0.025907 |
| positive | 10 | 0.203509 | 0.230731 | 0.065916 | 0.166245 | 0.287976 | 0.313181 | -0.132851 | 0.400354 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 1 | 0 | 9 |
