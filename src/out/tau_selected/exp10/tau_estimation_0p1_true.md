# Tau Estimation Summary

- Input: `src/out/experiment10/correlation_comparison_0p1.csv`
- Metric: `true`
- Objective: `balanced_accuracy`
- Estimated tau: `0.045303`
- Accuracy at tau: `1.0000`
- Balanced accuracy at tau: `1.0000`
- Macro F1 at tau: `1.0000`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.628722 | -0.672448 | -0.859017 | -0.717272 | -0.491638 | -0.370478 | -0.951370 | -0.347780 |
| random | 10 | 0.005823 | 0.006731 | -0.012409 | -0.006859 | 0.009611 | 0.019040 | -0.013068 | 0.043130 |
| positive | 10 | 0.130095 | 0.120895 | 0.064185 | 0.095276 | 0.151444 | 0.208883 | 0.063583 | 0.247327 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 0 | 0 | 10 |
