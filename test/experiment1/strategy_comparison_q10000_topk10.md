# Strategy Comparison Report

- Queries: `10000`
- `k_eval`: `10`

- Comparison plot: `strategy_comparison_q10000_topk10.png`
- Planner routing plot: `planner_routing_q10000_topk10.png`

## Strategy Metrics

| Strategy | Total Time (s) | QPS | Recall@k | Routing Time (s) | Execution Time (s) |
| --- | --- | --- | --- | --- | --- |
| PRE | 41.853400 | 238.93 | 0.9988 |  |  |
| POST | 6.627750 | 1508.81 | 0.6526 |  |  |
| ACORN | 14.941896 | 669.26 | 0.6691 |  |  |
| PLANNER | 9.362999 | 1068.03 | 0.7495 | 1.054765 | 8.308234 |

## Planner Routing

| Route | Queries |
| --- | --- |
| PRE | 997 |
| POST | 4110 |
| ACORN | 4893 |

## Planner Correlation Types

| Correlation | Queries |
| --- | --- |
| positive | 4346 |
| random | 0 |
| negative | 5654 |
