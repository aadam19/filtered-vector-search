# Tau Estimation Summary

- Input: `src/out/experiment11/correlation_comparison_0p5.csv`
- Metric: `tanh`
- Objective: `balanced_accuracy`
- Estimated tau: `0.006374`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.128415 | -0.137305 | -0.181087 | -0.167670 | -0.086314 | -0.042812 | -0.282563 | 0.033526 |
| random | 10 | 0.000570 | 0.000569 | -0.004304 | -0.000226 | 0.002352 | 0.003831 | -0.005388 | 0.006266 |
| positive | 10 | 0.141629 | 0.139963 | 0.100463 | 0.124526 | 0.157136 | 0.169511 | 0.066900 | 0.237023 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 9 | 0 | 1 |
| random | 0 | 10 | 0 |
| positive | 0 | 0 | 10 |
