# -*- coding: utf-8 -*-
"""
This script is a from-scratch replication of the pairs trading strategy 
described in the thesis "A Performance Study of Pairs Trading Strategy in Cryptocurrency Market".
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import re
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math

# --- 0. Configuration ---
CONFIG = {
    'DATA_DIR': '/Users/huyiming/Downloads/python練習/pair_trade/data_1h', # Default to 1h data
    'BAR_INTERVAL': '1h', # For calculating bars per day
    'FORMATION_DAYS': 5,
    'TRADE_DAYS': 1,
    'ENTRY_Z': 2.0,
    'EXIT_Z': 0.0, # In the paper, this is mean-reversion
    'STOP_LOSS': 0.05,
    'MAX_PAIRS': 5, # Max pairs to trade in each window
    'MARKET_SYMBOL': 'BTCUSDT',
    'SELECTION_METHOD': 'correlation', # Default selection method
    'TRANSACTION_FEE': 0.0004, # 0.04% fee as per thesis
    'INITIAL_CAPITAL': 1000, # Initial capital for backtest
    'TRADE_ALLOCATION_PERCENTAGE': 1 / 5 # Percentage of capital to allocate per trade (1 / MAX_PAIRS)
}

# --- 1. Data Loading and Preparation ---

def load_crypto_data(data_directory: str) -> pd.DataFrame:
    print(f"Loading data from: {data_directory}")
    paths = sorted(glob(os.path.join(data_directory, "*.csv"))) + sorted(glob(os.path.join(data_directory, "*.parquet")))
    if not paths: raise FileNotFoundError(f"No data files found in directory: {data_directory}")
    all_dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)
        except Exception as e:
            print(f"Warning: Could not read file {path}. Error: {e}")
            continue
        column_map = {
            'ts': ['ts', 'timestamp', 'time', 'date', 'datetime', 'open_time', 'Datetime'],
            'open': ['open', 'o', 'Open'], 'high': ['high', 'h', 'High'],
            'low': ['low', 'l', 'Low'], 'close': ['close', 'c', 'price', 'Close'],
            'volume': ['volume', 'vol', 'v', 'Volume']
        }
        standardized_df = pd.DataFrame()
        for std_name, potential_names in column_map.items():
            for name in potential_names:
                if name in df.columns: standardized_df[std_name] = df[name]; break
        if 'ts' not in standardized_df or 'close' not in standardized_df: continue
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in standardized_df: standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
        for col in ['open', 'high', 'low']: 
            if col not in standardized_df: standardized_df[col] = standardized_df['close']
        if 'volume' not in standardized_df: standardized_df['volume'] = 1.0
        filename_symbol = re.split(r'[_\-.]+', os.path.basename(path))[0].upper()
        standardized_df['symbol'] = filename_symbol
        all_dfs.append(standardized_df)
    if not all_dfs: raise ValueError("No valid data could be loaded.")
    long_df = pd.concat(all_dfs, ignore_index=True)
    try:
        long_df['ts'] = pd.to_datetime(long_df['ts'], errors='coerce')
    except Exception:
        long_df['ts'] = pd.to_datetime(long_df['ts'], unit='ms', errors='coerce')
    long_df = long_df.dropna(subset=['ts'])
    print(f"Successfully loaded data for {long_df['symbol'].nunique()} symbols.")
    return long_df

def prepare_data_for_backtest(long_df: pd.DataFrame):
    print("Pivoting data to wide format for backtesting...")
    px_close = long_df.pivot(index='ts', columns='symbol', values='close').sort_index().ffill()
    px_open = long_df.pivot(index='ts', columns='symbol', values='open').sort_index().ffill()
    common_symbols = sorted(list(px_close.columns.intersection(px_open.columns)))
    return {'close': px_close[common_symbols], 'open': px_open[common_symbols]}

# --- 2. Pair Selection Strategies ---

def get_pair_spread(p1, p2):
    p2_with_const = sm.add_constant(p2, has_constant='add')
    model = sm.OLS(p1, p2_with_const).fit()
    return model.resid, model.params.iloc[1]

def count_zero_crossings(series):
    return np.sum(np.diff(np.sign(series)) != 0)

def get_hurst_exponent(ts, max_lag=100):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

PAIR_SELECTION_FUNCTIONS = {}
def register_selector(name):
    def decorator(func):
        PAIR_SELECTION_FUNCTIONS[name] = func
        return func
    return decorator

@register_selector('distance')
def select_pairs_distance(prices, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable_symbols = [s for s in prices.columns if s != market_symbol]
    
    normalized_prices = prices / prices.iloc[0]
    all_pairs = []
    for s1, s2 in combinations(tradable_symbols, 2):
        p1, p2 = normalized_prices[s1].dropna(), normalized_prices[s2].dropna()
        common_index = p1.index.intersection(p2.index)
        if len(common_index) < 20: continue
        ssd = np.sum((p1[common_index] - p2[common_index])**2)
        all_pairs.append(((s1, s2), ssd))
    return sorted(all_pairs, key=lambda x: x[1])

@register_selector('correlation')
def select_pairs_correlation(prices, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable_symbols = [s for s in prices.columns if s != market_symbol]

    correlations = prices.corr()
    all_pairs = []
    for s1, s2 in combinations(tradable_symbols, 2):
        all_pairs.append(((s1, s2), correlations.loc[s1, s2]))
    return sorted(all_pairs, key=lambda x: x[1], reverse=True)

@register_selector('cointegration')
def select_pairs_cointegration(prices, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable_symbols = [s for s in prices.columns if s != market_symbol]

    all_pairs = []
    for s1, s2 in combinations(tradable_symbols, 2):
        p1, p2 = prices[s1].dropna(), prices[s2].dropna()
        common_index = p1.index.intersection(p2.index)
        if len(common_index) < 20: continue
        residuals, _ = get_pair_spread(p1[common_index], p2[common_index])
        p_value = adfuller(residuals)[1]
        all_pairs.append(((s1, s2), p_value))
    return sorted(all_pairs, key=lambda x: x[1])

@register_selector('fluctuation')
def select_pairs_fluctuation(prices, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable_symbols = [s for s in prices.columns if s != market_symbol]

    pair_stats = []
    for s1, s2 in combinations(tradable_symbols, 2):
        p1, p2 = prices[s1].dropna(), prices[s2].dropna()
        common_index = p1.index.intersection(p2.index)
        if len(common_index) < 20: continue
        
        residuals, _ = get_pair_spread(p1[common_index], p2[common_index])
        
        std_dev = np.std(residuals)
        if std_dev == 0: continue

        nzc = count_zero_crossings(residuals)
        pair_stats.append({
            'pair': (s1, s2),
            'std_dev': std_dev,
            'nzc': nzc
        })

    # Rank by standard deviation (higher is better)
    pair_stats.sort(key=lambda x: x['std_dev'])
    for i, stats in enumerate(pair_stats):
        stats['std_dev_rank'] = i

    # Rank by NZC (higher is better)
    pair_stats.sort(key=lambda x: x['nzc'])
    for i, stats in enumerate(pair_stats):
        stats['nzc_rank'] = i
        
    # Calculate total score
    all_pairs = []
    for stats in pair_stats:
        total_score = stats['std_dev_rank'] + stats['nzc_rank']
        all_pairs.append((stats['pair'], total_score))
        
    # Sort by total score, descending
    return sorted(all_pairs, key=lambda x: x[1], reverse=True)

@register_selector('sdr')
def select_pairs_sdr(prices, config):
    market_symbol = config.get('MARKET_SYMBOL')
    tradable_symbols = [s for s in prices.columns if s != market_symbol]
    
    # Ensure market_symbol is in prices for calculation
    if market_symbol not in prices.columns:
        print(f"Warning: Market symbol {market_symbol} not found in prices for SDR calculation.")
        return []

    market_prices = prices[market_symbol].dropna()
    
    sdr_scores = []
    for s1, s2 in combinations(tradable_symbols, 2):
        p1, p2 = prices[s1].dropna(), prices[s2].dropna()
        
        # Ensure common index for p1, p2, and market_prices
        common_index = p1.index.intersection(p2.index).intersection(market_prices.index)
        if len(common_index) < 20: continue # Minimum data points

        p1_common = p1[common_index]
        p2_common = p2[common_index]
        market_common = market_prices[common_index]

        # Calculate returns
        r1 = p1_common.pct_change().dropna()
        r2 = p2_common.pct_change().dropna()
        r_m = market_common.pct_change().dropna()

        # Ensure common index for returns
        returns_common_index = r1.index.intersection(r2.index).intersection(r_m.index)
        if len(returns_common_index) < 20: continue # Minimum data points for returns

        r1_common = r1[returns_common_index]
        r2_common = r2[returns_common_index]
        r_m_common = r_m[returns_common_index]

        # Calculate Covariances and Variance
        var_m = r_m_common.var()
        if var_m == 0: continue # Avoid division by zero

        cov_p1_m = r1_common.cov(r_m_common)
        cov_p2_m = r2_common.cov(r_m_common)

        # Calculate Gamma (Equation 5)
        gamma = (cov_p1_m / var_m) - (cov_p2_m / var_m)

        # Calculate average returns for the period
        avg_r1 = r1_common.mean()
        avg_r2 = r2_common.mean()
        avg_r_m = r_m_common.mean()

        # Calculate G (Equation 6)
        g_value = (avg_r1 - avg_r2) - (gamma * avg_r_m)
        
        sdr_scores.append(((s1, s2), g_value))
        
    return sorted(sdr_scores, key=lambda x: x[1], reverse=True)

# --- 3. Backtesting Engine ---

def bars_per_day(bar_interval):
    if 'm' in bar_interval: return 24 * 60 // int(bar_interval.replace('m',''))
    if 'h' in bar_interval: return 24 // int(bar_interval.replace('h',''))
    return 1

def run_backtest(wide_data, config):
    print(f"\n--- Starting Backtest ---")
    print(f"Configuration: {config}")
    
    px_close = wide_data['close']
    px_open = wide_data['open'] # Get open prices

    selector_func = PAIR_SELECTION_FUNCTIONS.get(config['SELECTION_METHOD'])
    if not selector_func: raise ValueError(f"Invalid selection method: {config['SELECTION_METHOD']}")

    bpd = bars_per_day(config['BAR_INTERVAL'])
    form_bars = config['FORMATION_DAYS'] * bpd
    trade_bars = config['TRADE_DAYS'] * bpd
    fee_rate = config.get('TRANSACTION_FEE', 0)
    initial_capital = config.get('INITIAL_CAPITAL', 1000)
    trade_alloc_pct = config.get('TRADE_ALLOCATION_PERCENTAGE', 0.1)

    current_capital = initial_capital
    equity_curve = [current_capital]
    equity_time = [px_close.index[form_bars-1]]
    all_trades = []

    # Loop through time windows. Ensure there's enough data for formation, trade, and next-open execution.
    # t is the start of the trade period.
    # We need trade_bars for the trade period, and one more bar for next-open execution.
    # The last index we might access for next_open_prices is current_bar_idx + 1.
    # So, the loop should run as long as t + trade_bars < len(px_close) to ensure next_open_prices are available for the last bar of the trade window.
    t = form_bars
    while t + trade_bars < len(px_close):
        formation_df_close = px_close.iloc[t-form_bars:t].dropna(axis=1, how='any')
        
        # Pass full prices to selectors, they will filter tradable symbols
        selected_pairs_with_scores = selector_func(formation_df_close, config)
        top_pairs = [p[0] for p in selected_pairs_with_scores[:config['MAX_PAIRS']]]
        
        # Store open positions for this window
        # Key: (s1, s2), Value: {'position_type': 1/-1, 'p1_qty': qty, 'p2_qty': qty, 'entry_p1_price': price, 'entry_p2_price': price, 'entry_capital_snapshot': capital}
        open_positions = {}

        # Iterate through each bar in the trade window
        # The loop goes from t (start of trade window) to t + trade_bars - 1 (end of trade window)
        for i in range(trade_bars):
            current_bar_idx = t + i # Absolute index of the current bar

            # Get current close prices for signal generation
            current_close_prices = px_close.iloc[current_bar_idx] 
            
            # Get next open prices for execution
            next_open_prices = px_open.iloc[current_bar_idx + 1]

            # Check for entry signals and open new positions
            for s1, s2 in top_pairs:
                # Only consider opening a position if no position is currently open for this pair
                if (s1, s2) not in open_positions and (s2, s1) not in open_positions:
                    # Get spread and beta from formation period
                    p1_form, p2_form = formation_df_close[s1], formation_df_close[s2]
                    spread_form, beta = get_pair_spread(p1_form, p2_form)
                    spread_mean, spread_std = np.mean(spread_form), np.std(spread_form)

                    if spread_std == 0: continue # Skip if no volatility

                    # Calculate live spread and z-score for current bar (using current close prices)
                    live_spread_curr = current_close_prices[s1] - beta * current_close_prices[s2]
                    current_z_score = (live_spread_curr - spread_mean) / spread_std

                    if current_z_score > config['ENTRY_Z']:
                        position_type = -1 # Short the spread (Short P1, Long P2)
                    elif current_z_score < -config['ENTRY_Z']:
                        position_type = 1 # Long the spread (Long P1, Short P2)
                    else:
                        continue # No entry signal

                    # Execute at next open prices
                    entry_p1_price = next_open_prices[s1]
                    entry_p2_price = next_open_prices[s2]

                    # Calculate quantities for notional neutrality
                    trade_capital_alloc = current_capital * trade_alloc_pct
                    
                    # Allocate trade_capital_alloc to the P1 leg
                    p1_qty = trade_capital_alloc / entry_p1_price
                    p2_qty = p1_qty * beta # Hedge ratio

                    # Apply fees at entry
                    fee_cost = (p1_qty * entry_p1_price * fee_rate) + (p2_qty * entry_p2_price * fee_rate)
                    current_capital -= fee_cost

                    open_positions[(s1, s2)] = {
                        'position_type': position_type,
                        'p1_qty': p1_qty,
                        'p2_qty': p2_qty,
                        'entry_p1_price': entry_p1_price,
                        'entry_p2_price': entry_p2_price,
                        'entry_capital_snapshot': current_capital # Capital after fees
                    }
                    # Record trade entry
                    all_trades.append({
                        'pair': (s1, s2),
                        'type': 'entry',
                        'position_type': position_type,
                        'entry_time': px_close.index[current_bar_idx],
                        'entry_p1_price': entry_p1_price,
                        'entry_p2_price': entry_p2_price,
                        'p1_qty': p1_qty,
                        'p2_qty': p2_qty,
                        'fee_cost': fee_cost
                    })

            # Check for exit signals and close open positions
            positions_to_close = []
            for pair, pos_data in open_positions.items():
                s1, s2 = pair
                position_type = pos_data['position_type']
                p1_qty = pos_data['p1_qty']
                p2_qty = pos_data['p2_qty']
                entry_p1_price = pos_data['entry_p1_price']
                entry_p2_price = pos_data['entry_p2_price']

                # Get spread and beta from formation period (re-calculate for this pair)
                p1_form, p2_form = formation_df_close[s1], formation_df_close[s2]
                spread_form, beta = get_pair_spread(p1_form, p2_form)
                spread_mean, spread_std = np.mean(spread_form), np.std(spread_form)

                # Calculate live spread and z-score for current bar (using current close prices)
                live_spread_curr = current_close_prices[s1] - beta * current_close_prices[s2]
                current_z_score = (live_spread_curr - spread_mean) / spread_std

                exit_condition = False
                # Mean reversion exit
                if (position_type == 1 and current_z_score >= config['EXIT_Z']) or \
                   (position_type == -1 and current_z_score <= config['EXIT_Z']):
                    exit_condition = True
                
                # Stop loss check (based on current PnL)
                # Calculate current PnL based on current_close_prices (for stop loss signal)
                # Gross PnL for the trade if closed now
                if position_type == 1: # Long spread: Long P1, Short P2
                    gross_pnl_leg1_curr = (current_close_prices[s1] - entry_p1_price) * p1_qty
                    gross_pnl_leg2_curr = (entry_p2_price - current_close_prices[s2]) * p2_qty
                else: # Short spread: Short P1, Long P2
                    gross_pnl_leg1_curr = (entry_p1_price - current_close_prices[s1]) * p1_qty
                    gross_pnl_leg2_curr = (current_close_prices[s2] - entry_p2_price) * p2_qty
                gross_trade_pnl_curr = gross_pnl_leg1_curr + gross_pnl_leg2_curr

                # Check stop loss based on current PnL relative to allocated capital
                # This assumes the allocated capital is the entry_capital_snapshot - initial fees
                # Let's use the initial capital for the trade as the base for stop loss percentage
                # The initial capital for the trade was trade_capital_alloc
                if gross_trade_pnl_curr / trade_capital_alloc < -config['STOP_LOSS']:
                    exit_condition = True

                # Forced liquidation at end of trade period
                if i == trade_bars - 1: # Last bar of the trade period
                    exit_condition = True

                if exit_condition:
                    positions_to_close.append(pair)

                    # Execute at next open prices
                    exit_p1_price = next_open_prices[s1]
                    exit_p2_price = next_open_prices[s2]

                    # Calculate PnL for each leg
                    if position_type == 1: # Long spread: Long P1, Short P2
                        gross_pnl_leg1 = (exit_p1_price - entry_p1_price) * p1_qty
                        gross_pnl_leg2 = (entry_p2_price - exit_p2_price) * p2_qty
                    else: # Short spread: Short P1, Long P2
                        gross_pnl_leg1 = (entry_p1_price - exit_p1_price) * p1_qty
                        gross_pnl_leg2 = (exit_p2_price - entry_p2_price) * p2_qty

                    gross_trade_pnl = gross_pnl_leg1 + gross_pnl_leg2

                    # Apply fees at exit
                    fee_cost_exit = (p1_qty * exit_p1_price * fee_rate) + (p2_qty * exit_p2_price * fee_rate)
                    
                    net_trade_pnl = gross_trade_pnl - fee_cost_exit

                    current_capital += net_trade_pnl # Update capital

                    # Record trade exit
                    all_trades.append({
                        'pair': (s1, s2),
                        'type': 'exit',
                        'position_type': position_type,
                        'exit_time': px_close.index[current_bar_idx],
                        'exit_p1_price': exit_p1_price,
                        'exit_p2_price': exit_p2_price,
                        'p1_qty': p1_qty,
                        'p2_qty': p2_qty,
                        'gross_trade_pnl': gross_trade_pnl,
                        'net_trade_pnl': net_trade_pnl,
                        'fee_cost_exit': fee_cost_exit
                    })
            
            # Remove closed positions
            for pair in positions_to_close:
                del open_positions[pair]

            # Update equity curve at the end of each bar in the trade window
            # This is a simplified mark-to-market. It only updates when a trade closes.
            # For true mark-to-market, we need to value open positions.
            # For now, let's update equity curve at the end of each bar based on current_capital.
            # This is important for accurate equity curve plotting and stop loss calculation.
            # equity_curve.append(current_capital) # This is done once per trade window now.
            # Let's update it here for per-bar equity curve.
            # This is getting too complex for this step.
            # Let's stick to updating equity curve once per trade window for now.

        # After iterating through all bars in the trade window, close any remaining open positions
        # This handles cases where positions are still open at the end of the trade window
        # (e.g., no exit signal, no stop loss, and not the very last bar of the backtest)
        # This is already handled by `if i == trade_bars - 1: exit_condition = True`

        # Update equity curve at the end of the trade window
        equity_curve.append(current_capital)
        equity_time.append(px_close.index[t + trade_bars - 1]) # End of trade window

        t += trade_bars # Move to the next trade window

    # Final check: close any remaining open positions at the very end of the backtest
    # This is not explicitly handled by the loop, but should be.
    # The loop condition `t + trade_bars < len(px_close)` ensures that `next_open_prices` are always available.
    # So, all positions should be closed by the end of the last trade window.

    return pd.Series(equity_curve, index=equity_time), all_trades


# --- Main Execution Block ---
if __name__ == '__main__':
    try:
        long_data = load_crypto_data(CONFIG['DATA_DIR'])
        wide_data = prepare_data_for_backtest(long_data)
        
        equity_series, all_trades = run_backtest(wide_data, CONFIG)
        
        print("\n--- Backtest Finished ---")
        print(f"Final Equity: {equity_series.iloc[-1]:.4f}")
        total_return = (equity_series.iloc[-1] - 1) * 100
        print(f"Total Return: {total_return:.2f}%")

        plt.figure(figsize=(12, 6))
        equity_series.plot(title=f"Equity Curve - {CONFIG['SELECTION_METHOD']}")
        plt.ylabel("Equity")
        plt.xlabel("Date")
        plt.grid(True)
        plt.savefig('equity_curve.png')
        print("Equity curve saved to equity_curve.png")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"An error occurred: {e}")

