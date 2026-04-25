# Tau Estimation Summary

- Input: `src/out/experiment8/correlation_comparison_0p001.csv`
- Metric: `heuristic`
- Objective: `balanced_accuracy`
- Estimated tau: `0.104385`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.323272 | -0.330754 | -0.380971 | -0.345100 | -0.284454 | -0.274636 | -0.433849 | -0.223896 |
| random | 10 | 0.032704 | 0.077585 | -0.090040 | -0.049865 | 0.095259 | 0.102269 | -0.103720 | 0.103720 |
| positive | 10 | 0.256426 | 0.305866 | 0.168083 | 0.281318 | 0.345100 | 0.380971 | -0.334231 | 0.433849 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 10 | 0 | 0 |
| random | 0 | 10 | 0 |
| positive | 1 | 0 | 9 |
