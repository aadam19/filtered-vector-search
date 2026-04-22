# Strategy Comparison Report

- Queries: `10000`
- `k_eval`: `100`

- Comparison plot: `strategy_comparison_q10000_topk100.png`
- Planner routing plot: `planner_routing_q10000_topk100.png`

## Strategy Metrics

| Strategy | Total Time (s) | QPS | Recall@k | Routing Time (s) | Execution Time (s) |
| --- | --- | --- | --- | --- | --- |
| PRE | 33.834200 | 295.56 | 0.9999 |  |  |
| POST | 12.426200 | 804.75 | 0.6518 |  |  |
| ACORN | 15.885872 | 629.49 | 0.9055 |  |  |
| PLANNER | 11.231850 | 890.33 | 0.9070 | 1.048045 | 10.183805 |

## Planner Routing

| Route | Queries |
| --- | --- |
| PRE | 1232 |
| POST | 3831 |
| ACORN | 4937 |

## Planner Correlation Types

| Correlation | Queries |
| --- | --- |
| positive | 4148 |
| random | 0 |
| negative | 5852 |
