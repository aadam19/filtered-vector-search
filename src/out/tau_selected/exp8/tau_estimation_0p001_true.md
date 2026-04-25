# Tau Estimation Summary

- Input: `src/out/experiment8/correlation_comparison_0p001.csv`
- Metric: `true`
- Objective: `balanced_accuracy`
- Estimated tau: `0.085904`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.208508 | -0.203148 | -0.276295 | -0.238287 | -0.189979 | -0.138344 | -0.284000 | -0.131290 |
| random | 10 | 0.022093 | 0.025278 | -0.020116 | -0.002059 | 0.043339 | 0.063601 | -0.041729 | 0.085383 |
| positive | 10 | 0.192049 | 0.173119 | 0.099568 | 0.145429 | 0.272205 | 0.288507 | 0.046054 | 0.295481 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 0 | 1 | 9 |
