# Tau Estimation Summary

- Input: `src/out/experiment11/correlation_comparison_0p5.csv`
- Metric: `pgap`
- Objective: `balanced_accuracy`
- Estimated tau: `0.003754`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.062278 | -0.070697 | -0.088695 | -0.086250 | -0.045636 | -0.024659 | -0.100788 | 0.015019 |
| random | 10 | 0.000341 | 0.000315 | -0.002318 | -0.000123 | 0.001292 | 0.002159 | -0.002894 | 0.003557 |
| positive | 10 | 0.077188 | 0.076647 | 0.060299 | 0.072121 | 0.086064 | 0.099414 | 0.029054 | 0.115206 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 9 | 0 | 1 |
| random | 0 | 10 | 0 |
| positive | 0 | 0 | 10 |
