# Tau Estimation Summary

- Input: `src/out/experiment9/correlation_comparison_0p01.csv`
- Metric: `true`
- Objective: `balanced_accuracy`
- Estimated tau: `0.038007`
- Accuracy at tau: `1.0000`
- Balanced accuracy at tau: `1.0000`
- Macro F1 at tau: `1.0000`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.226446 | -0.228185 | -0.291078 | -0.267352 | -0.194088 | -0.159453 | -0.291455 | -0.155814 |
| random | 10 | 0.016382 | 0.011852 | 0.005854 | 0.011143 | 0.019786 | 0.032992 | 0.003474 | 0.037714 |
| positive | 10 | 0.200579 | 0.209031 | 0.131644 | 0.142737 | 0.260774 | 0.273105 | 0.083205 | 0.291629 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 0 | 0 | 10 |
