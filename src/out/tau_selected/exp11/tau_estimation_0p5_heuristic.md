# Tau Estimation Summary

- Input: `src/out/experiment11/correlation_comparison_0p5.csv`
- Metric: `heuristic`
- Objective: `balanced_accuracy`
- Estimated tau: `0.104385`
- Accuracy at tau: `0.9667`
- Balanced accuracy at tau: `0.9667`
- Macro F1 at tau: `0.9666`

## Class Distribution

| Class | Count | Mean | Median | Q10 | Q25 | Q75 | Q90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| negative | 10 | -0.236502 | -0.305866 | -0.348555 | -0.341246 | -0.281318 | -0.158121 | -0.375096 | 0.433849 |
| random | 10 | 0.013280 | 0.065408 | -0.097777 | -0.087758 | 0.087589 | 0.102269 | -0.103720 | 0.103720 |
| positive | 10 | 0.323272 | 0.330754 | 0.274636 | 0.284454 | 0.345100 | 0.380971 | 0.223896 | 0.433849 |

## Confusion Matrix at Estimated Tau

| Actual \ Predicted | Negative | Random | Positive |
|---|---:|---:|---:|
| negative | 9 | 0 | 1 |
| random | 0 | 10 | 0 |
| positive | 0 | 0 | 10 |
