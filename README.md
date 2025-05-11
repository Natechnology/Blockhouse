# README.md

## Overview
This repository provides **`backtest.py`**, a standalone Python script to back-test and tune a Smart Order Router (SOR) using the static cost model from Cont & Kukanov. It benchmarks against three baselines (best-ask, TWAP, VWAP) and saves a cumulative-cost plot (`results.png`).

---

## Approach
- **Static Cost Allocator**: Implements the Cont & Kukanov pseudocode to minimize expected cost when splitting a parent order across venues.
- **Branch Pruning**: Prunes infeasible split branches based on remaining capacity, reducing combinatorial blow-up.
- **Execution Helper**: `execute_router(snapshots, params, order_size)` consolidates cost and rebate logic into a single function.
- **Grid Search**: Exhaustive search over three risk parameters to identify the cost-minimizing set.
- **Baselines**:
  - **Best-Ask**: Execute entire order at the lowest available ask price.
  - **TWAP**: Time-weighted average price over 60-second buckets.
  - **VWAP**: Volume-weighted average price across displayed depth.

---

## Parameter Ranges
- **λ_over (overfill penalty)**: `0.01`, `0.05`, `0.1`  
- **λ_under (underfill penalty)**: `0.05`, `0.10`, `0.20`  
- **θ_queue (queue-risk penalty)**: `1e-4`, `5e-4`, `1e-3`

---

## Usage
```bash
python backtest.py /path/to/l1_day.csv
```
---

## Improving Fill Realism

To capture market microstructure effects, consider:
  - **Slippage Model**: Introduce a concave price-impact function that raises execution cost when consuming multiple price levels.
  - **Queue Position**: Simulate order book queues to estimate fill probability for limit orders, based on historical book dynamics.
