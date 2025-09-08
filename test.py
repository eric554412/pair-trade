# -*- coding: utf-8 -*-
"""
From-scratch replication of a crypto pairs trading backtest with strict
no-lookahead design and next-open execution.

Key properties:
- Formation window (close) to select pairs & estimate beta/μ/σ (no leakage).
- Signals on close[t], execute at open[t+1] (next-open).
- No new entries on the last bar of each trade window.
- Two-leg PnL with beta-sign handling; fees charged on absolute notional.
- Robust timestamp parsing (string or ms-epoch) + dedup.
- Safer fluctuation selector ranking (larger std_dev & NZC are better).
"""

import os
import re
import math
import numpy as np
import pandas as pd
from glob import glob
from itertools import combinations
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# --- 0. Configuration ---
CONFIG = {
    'DATA_DIR': '/Users/huyiming/Downloads/python練習/pair_trade/data_1h',  # path with per-symbol CSV/Parquet
    'BAR_INTERVAL': '1h',            # for bars per day calculation only
    'FORMATION_DAYS': 5,             # formation window (days)
    'TRADE_DAYS': 1,                 # trade window (days)
    'ENTRY_Z': 2.0,                  # entry threshold on z-score
    'EXIT_Z': 0.0,                   # mean reversion exit
    'STOP_LOSS': 0.05,               # stop loss (fraction of allocated capital)
    'MAX_PAIRS': 5,                  # number of pairs to trade per window
    'MARKET_SYMBOL': 'BTCUSDT',      # used by SDR selector
    'SELECTION_METHOD': 'cointegration',  # 'distance' | 'correlation' | 'cointegration' | 'fluctuation' | 'sdr'
    'TRANSACTION_FEE': 0.0004,       # per-leg per transaction (0.04%)
    'INITIAL_CAPITAL': 1000.0,       # initial equity
    'TRADE_ALLOCATION_PERCENTAGE': 1/5,  # capital per trade (≈ 1/MAX_PAIRS)
    'MIN_COMMON_BARS': 20            # minimum overlapping points for stats
}

# --- 1. Data Loading and Preparation ---

def load_crypto_data(data_directory: str) -> pd.DataFrame:
    print(f"Loading data from: {data_directory}")
    paths = sorted(glob(os.path.join(data_directory, "*.csv"))) + \
            sorted(glob(os.path.join(data_directory, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No data files found in directory: {data_directory}")

    all_dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}")
            continue

        column_map = {
            'ts':     ['ts', 'timestamp', 'time', 'date', 'datetime', 'open_time', 'Datetime'],
            'open':   ['open', 'o', 'Open'],
            'high':   ['high', 'h', 'High'],
            'low':    ['low', 'l', 'Low'],
            'close':  ['close', 'c', 'price', 'Close'],
            'volume': ['volume', 'vol', 'v', 'Volume']
        }
        std = {}
        for std_name, cands in column_map.items():
            for c in cands:
                if c in df.columns:
                    std[std_name] = df[c]
                    break

        if 'ts' not in std or 'close' not in std:
            continue

        std_df = pd.DataFrame(std)
        # numeric coercion
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in std_df:
                std_df[col] = pd.to_numeric(std_df[col], errors='coerce')
        # fill missing OHLC with close
        for col in ['open', 'high', 'low']:
            if col not in std_df:
                std_df[col] = std_df['close']
        if 'volume' not in std_df:
            std_df['volume'] = 1.0

        # symbol from filename
        filename_symbol = re.split(r'[_\-.]+', os.path.basename(path))[0].upper()
        std_df['symbol'] = filename_symbol
        all_dfs.append(std_df)

    if not all_dfs:
        raise ValueError("No valid data could be loaded.")

    long_df = pd.concat(all_dfs, ignore_index=True)

    # Robust timestamp parse: string first, then ms-epoch
    t1 = pd.to_datetime(long_df['ts'], errors='coerce')
    t2 = pd.to_datetime(long_df['ts'], unit='ms', errors='coerce')
    long_df['ts'] = np.where(t1.notna(), t1, t2)
    long_df = long_df.dropna(subset=['ts'])

    # Sort & deduplicate within (symbol, ts)
    long_df = (long_df
               .sort_values(['symbol', 'ts'])
               .drop_duplicates(subset=['symbol', 'ts'], keep='last'))

    print(f"Successfully loaded data for {long_df['symbol'].nunique()} symbols, "
          f"{len(long_df):,} rows.")
    return long_df


def prepare_data_for_backtest(long_df: pd.DataFrame):
    print("Pivoting data to wide format...")
    px_close = long_df.pivot(index='ts', columns='symbol', values='close').sort_index().ffill()
    px_open  = long_df.pivot(index='ts', columns='symbol', values='open').sort_index().ffill()
    common_symbols = sorted(list(px_close.columns.intersection(px_open.columns)))
    return {'close': px_close[common_symbols], 'open': px_open[common_symbols]}

# --- 2. Pair Selection Strategies ---

def get_pair_spread(p1: pd.Series, p2: pd.Series):
    p2c = sm.add_constant(p2, has_constant='add')
    model = sm.OLS(p1, p2c).fit()
    return model.resid, model.params.iloc[1]

def count_zero_crossings(series: pd.Series) -> int:
    return int(np.sum(np.diff(np.sign(series)) != 0))

def get_hurst_exponent(ts: np.ndarray, max_lag=100):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

PAIR_SELECTION_FUNCTIONS = {}
def register_selector(name):
    def deco(func):
        PAIR_SELECTION_FUNCTIONS[name] = func
        return func
    return deco

@register_selector('distance')
def select_pairs_distance(prices: pd.DataFrame, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable = [s for s in prices.columns if s != market_symbol]
    norm_px = prices / prices.iloc[0]
    all_pairs = []
    for s1, s2 in combinations(tradable, 2):
        p1, p2 = norm_px[s1].dropna(), norm_px[s2].dropna()
        idx = p1.index.intersection(p2.index)
        if len(idx) < config['MIN_COMMON_BARS']: 
            continue
        ssd = float(np.sum((p1[idx] - p2[idx])**2))
        all_pairs.append(((s1, s2), ssd))
    return sorted(all_pairs, key=lambda x: x[1])

@register_selector('correlation')
def select_pairs_correlation(prices: pd.DataFrame, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable = [s for s in prices.columns if s != market_symbol]
    corr = prices[tradable].corr()
    all_pairs = []
    for s1, s2 in combinations(tradable, 2):
        all_pairs.append(((s1, s2), float(corr.loc[s1, s2])))
    return sorted(all_pairs, key=lambda x: x[1], reverse=True)

@register_selector('cointegration')
def select_pairs_cointegration(prices: pd.DataFrame, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable = [s for s in prices.columns if s != market_symbol]

    out = []
    min_len = int(config.get('MIN_COMMON_BARS', 20))

    for s1, s2 in combinations(tradable, 2):
        p1 = prices[s1].replace([np.inf, -np.inf], np.nan).dropna()
        p2 = prices[s2].replace([np.inf, -np.inf], np.nan).dropna()
        idx = p1.index.intersection(p2.index)
        if len(idx) < min_len:
            continue

        x1 = p1[idx]
        x2 = p2[idx]

        # 兩腿都要有波動，否則 OLS/ADF 都不可靠
        if float(np.var(x1)) == 0.0 or float(np.var(x2)) == 0.0:
            continue

        # OLS 殘差
        try:
            resid, _ = get_pair_spread(x1, x2)
        except Exception:
            # statsmodels 可能仍對病態資料報錯，直接跳過這對
            continue

        resid = pd.Series(resid, index=x1.index)
        resid = resid.replace([np.inf, -np.inf], np.nan).dropna()

        # 殘差需要足夠長度與變異
        if len(resid) < min_len:
            continue
        if float(np.var(resid)) == 0.0 or resid.nunique() < 2:
            continue

        # ADF 測試（動態 maxlag，避免長度太短）
        try:
            maxlag = max(0, min(10, len(resid) // 2 - 1))
            pval = float(adfuller(resid.values, maxlag=maxlag, autolag='AIC')[1])
        except Exception:
            # 包含「x is constant」等錯誤 → 跳過
            continue

        out.append(((s1, s2), pval))

    # p值越小越好
    return sorted(out, key=lambda x: x[1])


@register_selector('fluctuation')
def select_pairs_fluctuation(prices: pd.DataFrame, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable = [s for s in prices.columns if s != market_symbol]
    stats = []
    for s1, s2 in combinations(tradable, 2):
        p1, p2 = prices[s1].dropna(), prices[s2].dropna()
        idx = p1.index.intersection(p2.index)
        if len(idx) < config['MIN_COMMON_BARS']:
            continue
        resid, _ = get_pair_spread(p1[idx], p2[idx])
        std_dev = float(np.std(resid))
        if std_dev == 0 or np.isnan(std_dev):
            continue
        nzc = count_zero_crossings(resid)
        stats.append({'pair': (s1, s2), 'std_dev': std_dev, 'nzc': nzc})

    # Larger std_dev & NZC are better → assign ascending ranks after sorting desc
    stats.sort(key=lambda x: x['std_dev'], reverse=True)
    for i, s in enumerate(stats): s['std_rank'] = i
    stats.sort(key=lambda x: x['nzc'], reverse=True)
    for i, s in enumerate(stats): s['nzc_rank'] = i
    ranked = [ (s['pair'], s['std_rank'] + s['nzc_rank']) for s in stats ]
    return sorted(ranked, key=lambda x: x[1])

@register_selector('sdr')
def select_pairs_sdr(prices: pd.DataFrame, config):
    mkt = config.get('MARKET_SYMBOL')
    if mkt not in prices.columns:
        print(f"[WARN] SDR: market symbol {mkt} missing in formation window.")
        return []
    tradable = [s for s in prices.columns if s != mkt]
    m = prices[mkt].dropna()
    out = []
    for s1, s2 in combinations(tradable, 2):
        p1, p2 = prices[s1].dropna(), prices[s2].dropna()
        idx = p1.index.intersection(p2.index).intersection(m.index)
        if len(idx) < config['MIN_COMMON_BARS']:
            continue
        r1 = p1[idx].pct_change().dropna()
        r2 = p2[idx].pct_change().dropna()
        rm = m[idx].pct_change().dropna()
        idx2 = r1.index.intersection(r2.index).intersection(rm.index)
        if len(idx2) < config['MIN_COMMON_BARS']:
            continue
        r1c, r2c, rmc = r1[idx2], r2[idx2], rm[idx2]
        var_m = float(rmc.var())
        if var_m == 0:
            continue
        cov1 = float(r1c.cov(rmc))
        cov2 = float(r2c.cov(rmc))
        gamma = (cov1 / var_m) - (cov2 / var_m)
        g = (float(r1c.mean()) - float(r2c.mean())) - gamma * float(rmc.mean())
        out.append(((s1, s2), g))
    return sorted(out, key=lambda x: x[1], reverse=True)

# --- 3. Backtesting Engine ---

def bars_per_day(bar_interval: str) -> int:
    if 'm' in bar_interval:
        return 24 * 60 // int(bar_interval.replace('m', ''))
    if 'h' in bar_interval:
        return 24 // int(bar_interval.replace('h', ''))
    return 1

def run_backtest(wide_data, config):
    print("\n--- Starting Backtest ---")
    print(f"Configuration: {config}")

    cl = wide_data['close']
    op = wide_data['open']

    selector_func = PAIR_SELECTION_FUNCTIONS.get(config['SELECTION_METHOD'])
    if not selector_func:
        raise ValueError(f"Invalid selection method: {config['SELECTION_METHOD']}")

    bpd = bars_per_day(config['BAR_INTERVAL'])
    form_bars  = int(config['FORMATION_DAYS'] * bpd)
    trade_bars = int(config['TRADE_DAYS'] * bpd)
    fee_rate   = float(config.get('TRANSACTION_FEE', 0.0))
    equity     = float(config.get('INITIAL_CAPITAL', 1000.0))
    alloc_pct  = float(config.get('TRADE_ALLOCATION_PERCENTAGE', 0.2))

    # equity curve
    equity_curve = [equity]
    equity_time  = [cl.index[form_bars - 1]]
    all_trades = []

    # loop condition must ensure we can access open[t+1] on the last trade bar
    t = form_bars
    while t + trade_bars < len(cl):
        # formation window (strict, dropna per column)
        form_cl = cl.iloc[t - form_bars : t].dropna(axis=1, how='any')

        # selector (with fallback for SDR if market missing)
        cur_selector = selector_func
        if config['SELECTION_METHOD'] == 'sdr' and config['MARKET_SYMBOL'] not in form_cl.columns:
            print(f"[{cl.index[t]}] MARKET_SYMBOL missing in formation window → fallback to 'correlation'.")
            cur_selector = PAIR_SELECTION_FUNCTIONS['correlation']

        pairs_scored = cur_selector(form_cl, config)
        top_pairs = [p[0] for p in pairs_scored[:config['MAX_PAIRS']]]

        # Pre-compute beta / mu / sigma from formation only
        pair_stats = {}
        for s1, s2 in top_pairs:
            if s1 not in form_cl.columns or s2 not in form_cl.columns:
                continue
            resid, beta = get_pair_spread(form_cl[s1], form_cl[s2])
            mu  = float(np.mean(resid))
            sig = float(np.std(resid))
            if sig == 0 or np.isnan(sig):
                continue
            pair_stats[(s1, s2)] = {'beta': beta, 'mu': mu, 'sigma': sig}

        open_positions = {}  # (s1,s2) -> dict

        # iterate bars within trade window
        for i in range(trade_bars):
            cur_idx = t + i
            # prices for signal (close[t]) and execution (open[t+1])
            close_now = cl.iloc[cur_idx]
            open_next = op.iloc[cur_idx + 1]

            # --- Entries (skip on last bar to avoid instant forced exit) ---
            if i != trade_bars - 1:
                for s1, s2 in list(pair_stats.keys()):
                    if (s1, s2) in open_positions or (s2, s1) in open_positions:
                        continue

                    stats = pair_stats[(s1, s2)]
                    beta, mu, sigma = stats['beta'], stats['mu'], stats['sigma']

                    live_spread = close_now[s1] - beta * close_now[s2]
                    z = (live_spread - mu) / sigma

                    if z > config['ENTRY_Z']:
                        position_type = -1  # short spread: -P1, +P2
                    elif z < -config['ENTRY_Z']:
                        position_type = 1   # long spread: +P1, -P2
                    else:
                        continue

                    entry_p1 = open_next.get(s1, np.nan)
                    entry_p2 = open_next.get(s2, np.nan)
                    if not (np.isfinite(entry_p1) and np.isfinite(entry_p2)):
                        continue

                    trade_alloc = equity * alloc_pct
                    if trade_alloc <= 0:
                        continue

                    # quantities: use abs(beta) for size; direction handled later
                    p1_qty = trade_alloc / entry_p1
                    p2_qty = p1_qty * abs(beta)

                    # entry fees (absolute notionals)
                    fee_in = (abs(p1_qty) * entry_p1 * fee_rate) + (abs(p2_qty) * entry_p2 * fee_rate)
                    equity -= fee_in

                    open_positions[(s1, s2)] = {
                        'position_type': position_type,
                        'p1_qty': p1_qty,
                        'p2_qty': p2_qty,
                        'entry_p1': entry_p1,
                        'entry_p2': entry_p2,
                        'alloc': trade_alloc,
                        'beta': beta, 'mu': mu, 'sigma': sigma
                    }

                    all_trades.append({
                        'pair': (s1, s2), 'type': 'entry', 'pos': position_type,
                        'time': cl.index[cur_idx], 'p1': entry_p1, 'p2': entry_p2,
                        'p1_qty': p1_qty, 'p2_qty': p2_qty, 'fee_in': fee_in
                    })

            # --- Exits / Stops / Forced at last bar ---
            to_close = []
            for (s1, s2), pos in open_positions.items():
                pos_type = pos['position_type']
                p1_qty   = pos['p1_qty']
                p2_qty   = pos['p2_qty']
                entry_p1 = pos['entry_p1']
                entry_p2 = pos['entry_p2']
                alloc    = pos['alloc']
                beta     = pos['beta']
                mu       = pos['mu']
                sigma    = pos['sigma']
                beta_sign = 1 if beta >= 0 else -1

                live_spread = close_now[s1] - beta * close_now[s2]
                z = (live_spread - mu) / sigma

                # floating PnL using close (MTM)
                if pos_type == 1:   # long spread: +P1, -P2
                    pnl1_cur = (close_now[s1] - entry_p1) * p1_qty
                    pnl2_cur = (entry_p2 - close_now[s2]) * (p2_qty * beta_sign)
                else:               # short spread: -P1, +P2
                    pnl1_cur = (entry_p1 - close_now[s1]) * p1_qty
                    pnl2_cur = (close_now[s2] - entry_p2) * (p2_qty * beta_sign)
                gross_pnl_cur = pnl1_cur + pnl2_cur

                exit_flag = False
                # mean-reversion exit
                if (pos_type == 1 and z >= config['EXIT_Z']) or \
                   (pos_type == -1 and z <= config['EXIT_Z']):
                    exit_flag = True
                # stop-loss vs allocated capital
                if alloc > 0 and (gross_pnl_cur / alloc) < -float(config['STOP_LOSS']):
                    exit_flag = True
                # forced exit on last bar of trade window
                if i == trade_bars - 1:
                    exit_flag = True

                if exit_flag:
                    exit_p1 = open_next.get(s1, np.nan)
                    exit_p2 = open_next.get(s2, np.nan)
                    if not (np.isfinite(exit_p1) and np.isfinite(exit_p2)):
                        # if open is missing (should be rare), skip this tick's exit
                        continue

                    if pos_type == 1:
                        pnl1 = (exit_p1 - entry_p1) * p1_qty
                        pnl2 = (entry_p2 - exit_p2) * (p2_qty * beta_sign)
                    else:
                        pnl1 = (entry_p1 - exit_p1) * p1_qty
                        pnl2 = (exit_p2 - entry_p2) * (p2_qty * beta_sign)
                    gross_pnl = pnl1 + pnl2

                    fee_out = (abs(p1_qty) * exit_p1 * fee_rate) + (abs(p2_qty) * exit_p2 * fee_rate)
                    net_pnl = gross_pnl - fee_out
                    equity += net_pnl

                    all_trades.append({
                        'pair': (s1, s2), 'type': 'exit', 'pos': pos_type,
                        'time': cl.index[cur_idx],
                        'p1': exit_p1, 'p2': exit_p2,
                        'p1_qty': p1_qty, 'p2_qty': p2_qty,
                        'gross_pnl': gross_pnl, 'net_pnl': net_pnl,
                        'fee_out': fee_out
                    })
                    to_close.append((s1, s2))

            for k in to_close:
                open_positions.pop(k, None)

        # end of trade window → record equity
        equity_curve.append(equity)
        equity_time.append(cl.index[t + trade_bars - 1])

        t += trade_bars

    return pd.Series(equity_curve, index=equity_time), all_trades


# --- 4. Main ---
if __name__ == '__main__':
    try:
        long_df = load_crypto_data(CONFIG['DATA_DIR'])
        wide = prepare_data_for_backtest(long_df)
        equity_series, trades = run_backtest(wide, CONFIG)

        print("\n--- Backtest Finished ---")
        print(f"Final Equity: {equity_series.iloc[-1]:.4f}")
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1.0) * 100
        print(f"Total Return: {total_return:.2f}%")
        print(f"Trades executed: {sum(1 for t in trades if t['type']=='exit')}")

        plt.figure(figsize=(12, 6))
        equity_series.plot(title=f"Equity Curve — selector={CONFIG['SELECTION_METHOD']}, "
                                 f"formation={CONFIG['FORMATION_DAYS']}d, trade={CONFIG['TRADE_DAYS']}d")
        plt.ylabel("Equity")
        plt.xlabel("Date")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('equity_curve.png', dpi=150)
        print("Equity curve saved to equity_curve.png")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"An error occurred: {e}")
