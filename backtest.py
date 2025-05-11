# %%
"""
backtest.py
Back-test and parameter tuning for a Smart Order Router following Cont & Kukanov static cost model.
Implements:
  - allocate(order_size, venues, lambda_over, lambda_under, theta_queue) with branch pruning
  - compute_cost(split, venues, order_size, lambda_over, lambda_under, theta_queue)
  - execute_router(snapshots, params, order_size) helper to consolidate cost logic
  - back-test replay over L1 snapshots from l1_day.csv
  - simple grid search over (lambda_over, lambda_under, theta_queue)
  - baseline strategies: best-ask, TWAP, VWAP
  - cumulative-cost plotting for the tuned router
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Venue representation
class Venue:
    def __init__(self, ask, ask_size, fee, rebate):
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate


def allocate(order_size, venues, lam_over, lam_under, theta):
    """
    Static Cont-Kukanov split across venues with pruning of infeasible branches.
    Falls back to greedy if no exact split sums to order_size.
    """
    n = len(venues)
    capacities = [v.ask_size for v in venues]
    step = 100 if order_size >= 100 else 1
    splits = [[]]

    # Build splits with capacity-based pruning
    for idx in range(n):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            needed = order_size - used
            if sum(capacities[idx:]) < needed:
                continue
            max_v = min(needed, capacities[idx])
            for q in range(0, max_v + 1, step):
                after = used + q
                if sum(capacities[idx+1:]) < order_size - after:
                    continue
                new_splits.append(alloc + [q])
        splits = new_splits

    # Evaluate costs
    best_cost = float('inf')
    best_split = None
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lam_over, lam_under, theta)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc

    # Fallback greedy
    if best_split is None:
        rem = order_size
        best_split = [0]*n
        order = sorted(range(n), key=lambda i: venues[i].ask + venues[i].fee)
        for i in order:
            take = min(capacities[i], rem)
            best_split[i] = take
            rem -= take
            if rem<=0:
                break
        if rem>0:
            best_split[order[0]] += rem
        best_cost = compute_cost(best_split, venues, order_size, lam_over, lam_under, theta)

    return best_split, best_cost


def compute_cost(split, venues, order_size, lam_over, lam_under, theta):
    """Compute total expected cost for a proposed allocation split."""
    executed = 0
    cash = 0.0
    for i, v in enumerate(venues):
        exe = min(split[i], v.ask_size)
        executed += exe
        cash += exe * (v.ask + v.fee)
        cash -= max(split[i] - exe, 0) * v.rebate
    under = max(order_size - executed, 0)
    over = max(executed - order_size, 0)
    return cash + theta*(under+over) + lam_under*under + lam_over*over


def load_snapshots(csv_path):
    """Load L1 feed and aggregate one snapshot per venue per timestamp."""
    df = pd.read_csv(csv_path, parse_dates=['ts_event'])
    df.sort_values('ts_event', inplace=True)
    df = df.drop_duplicates(subset=['ts_event','publisher_id'], keep='first')
    snapshots = []
    for ts, group in df.groupby('ts_event'):
        vs = []
        for _, row in group.iterrows():
            vs.append(Venue(row['ask_px_00'], int(row['ask_sz_00']), 0.003, 0.002))
        snapshots.append(vs)
    return snapshots


def execute_router(snapshots, params, order_size=5000):
    """
    Run the router over snapshots, returning total cash and cumulative cost list.
    """
    rem = order_size
    cash = 0.0
    cum = []
    for vs in snapshots:
        if rem<=0:
            break
        split, _ = allocate(rem, vs, *params)
        executed = 0
        for i, v in enumerate(vs):
            qty = min(split[i], v.ask_size)
            cash += qty*(v.ask+v.fee)
            cash -= max(split[i]-qty,0)*v.rebate
            executed += qty
        rem -= executed
        cum.append(cash)
    if rem>0 and snapshots:
        last = snapshots[-1]
        best = min(last, key=lambda v: v.ask+v.fee)
        cash += rem*(best.ask+best.fee)
        cum.append(cash)
    return cash, cum


def backtest_router(snapshots, params, order_size=5000):
    """Helper that returns total cost and avg price."""
    total, _ = execute_router(snapshots, params, order_size)
    return total, total/order_size


def baseline_best_ask(snapshots, order_size=5000):
    vs = snapshots[0]
    best = min(vs, key=lambda v:v.ask+v.fee)
    cost = order_size*(best.ask+best.fee)
    return cost, cost/order_size


def baseline_twap(snapshots, order_size=5000):
    px = [np.mean([v.ask+v.fee for v in vs]) for vs in snapshots]
    avg = np.mean(px)
    return order_size*avg, avg


def baseline_vwap(snapshots, order_size=5000):
    num = sum((v.ask+v.fee)*v.ask_size for vs in snapshots for v in vs)
    den = sum(v.ask_size for vs in snapshots for v in vs)
    vwap = num/den if den else 0
    return order_size*vwap, vwap


def plot_cumulative(snapshots, params, order_size=5000):
    """Plot cumulative cost curve for the router."""
    _, cum = execute_router(snapshots, params, order_size)
    plt.figure()
    plt.plot(cum)
    plt.xlabel('Snapshot index')
    plt.ylabel('Cumulative cash spent')
    plt.title('Router cumulative cost over time')
    plt.savefig('results.png')


def main():
    # Determine CSV path
    default = '/Users/krx/Downloads/Blockhouse/l1_day.csv'
    data_path = default
    for a in sys.argv[1:]:
        if a.lower().endswith('.csv') and os.path.isfile(a):
            data_path = a
            break

    snapshots = load_snapshots(data_path)

    grid = {'lam_over':[0.01,0.05,0.1], 'lam_under':[0.05,0.1,0.2], 'theta':[1e-4,5e-4,1e-3]}
    best, best_params, best_res = float('inf'), None, None
    for lo in grid['lam_over']:
        for lu in grid['lam_under']:
            for th in grid['theta']:
                cost, avg = backtest_router(snapshots,(lo,lu,th))
                if cost<best:
                    best, best_params, best_res = cost, (lo,lu,th), (cost,avg)

    ba_c, ba_p = baseline_best_ask(snapshots)
    tw_c, tw_p = baseline_twap(snapshots)
    vw_c, vw_p = baseline_vwap(snapshots)
    savings = {
        'best_ask': (ba_c-best_res[0])/ba_c*1e4,
        'TWAP':     (tw_c-best_res[0])/tw_c*1e4,
        'VWAP':     (vw_c-best_res[0])/vw_c*1e4
    }

    result = {
        'best_params': {'lam_over':best_params[0],'lam_under':best_params[1],'theta':best_params[2]},
        'router':      {'total_cost':best_res[0],'avg_price':best_res[1]},
        'baseline':    {'best_ask':{'total_cost':ba_c,'avg_price':ba_p},'TWAP':{'total_cost':tw_c,'avg_price':tw_p},'VWAP':{'total_cost':vw_c,'avg_price':vw_p}},
        'savings_bps': savings
    }

    print(json.dumps(result,indent=2))
    plot_cumulative(snapshots,best_params)

if __name__=='__main__':
    main()


# %%
