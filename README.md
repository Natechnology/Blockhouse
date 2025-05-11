# README.md

## Overview
This repository provides **`backtest.py`**, a standalone Python script to back-test and tune a Smart Order Router (SOR) using the static cost model from Cont & Kukanov (“Optimal order placement in limit order markets”). It also benchmarks against three baselines and saves a cumulative-cost plot (`results.png`).

## Approach
- **Static Cost Allocator**: Implements the Cont & Kukanov pseudocode to split a parent order across venues, minimizing expected cost.  
- **Branch Pruning**: Prunes infeasible splits early based on remaining capacity to control combinatorial explosion.  
- **Unified Execution Helper**: `execute_router` applies allocation, accrues execution cost and rebates, and records cumulative cash spent.  
- **Grid Search**: Exhaustively searches over three risk parameters to find the set minimizing total cost.  
- **Baselines**:  
  - **Best-Ask**: Immediate execution at the single lowest ask price.  
  - **TWAP**: Time-weighted average price over successive 60-second buckets.  
  - **VWAP**: Volume-weighted average price across all displayed depth.

## Parameter Ranges
- **λ_over (overfill penalty)**: `0.01`, `0.05`, `0.1`  
- **λ_under (underfill penalty)**: `0.05`, `0.10`, `0.20`  
- **θ_queue (queue-risk penalty)**: `1e-4`, `5e-4`, `1e-3`

## Usage
```bash
python backtest.py /path/to/l1_day.csv
