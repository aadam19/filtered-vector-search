# Strategy Comparison Report

- Queries: `10000`
- `k_eval`: `100`

- Comparison plot: `strategy_comparison_q10000_topk100.png`
- Planner routing plot: `planner_routing_q10000_topk100.png`

## Strategy Metrics

| Strategy | Total Time (s) | QPS | Recall@k | Routing Time (s) | Execution Time (s) |
| --- | --- | --- | --- | --- | --- |
| PRE | 42.175700 | 237.10 | 0.9999 |  |  |
| POST | 12.335800 | 810.65 | 0.6724 |  |  |
| ACORN | 16.331014 | 612.33 | 0.9013 |  |  |
| PLANNER | 11.107705 | 900.28 | 0.8980 | 1.055959 | 10.051747 |

## Planner Routing

| Route | Queries |
| --- | --- |
| PRE | 1010 |
| POST | 4011 |
| ACORN | 4979 |

## Planner Correlation Types

| Correlation | Queries |
| --- | --- |
| positive | 4275 |
| random | 0 |
| negative | 5725 |
