"""Deep investigation of why R:R ratio is not 1:2 as designed"""
import pandas as pd
import numpy as np

# Load trade log
trades = pd.read_csv('backtest_trade_log.csv')

print("=" * 70)
print("INVESTIGATING R:R RATIO DISCREPANCY")
print("=" * 70)

# ============================================================
# 1. BASIC STATISTICS
# ============================================================
print("\n1. BASIC STATISTICS")
print("-" * 50)

wins = trades[trades['pnl_pct'] > 0]
losses = trades[trades['pnl_pct'] <= 0]

print(f"Total Trades: {len(trades)}")
print(f"Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
print(f"Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
print(f"\nAverage Win: {wins['pnl_pct'].mean():.4f}%")
print(f"Average Loss: {abs(losses['pnl_pct'].mean()):.4f}%")
print(f"Actual R:R: 1:{wins['pnl_pct'].mean() / abs(losses['pnl_pct'].mean()):.2f}")

# ============================================================
# 2. EXPECTED vs ACTUAL BY ATR
# ============================================================
print("\n2. EXPECTED vs ACTUAL PnL BASED ON ATR")
print("-" * 50)

# The designed system:
# - SL = 1.0 x ATR (risk 1 ATR)
# - TP = 2.0 x ATR (gain 2 ATR)
# So for each trade:
# - Expected loss if SL hit = -1 * ATR_pct (approximately)
# - Expected win if TP hit = +2 * ATR_pct (approximately)

trades['expected_loss'] = trades['atr_pct'] * 1.0 * 100  # in %
trades['expected_win'] = trades['atr_pct'] * 2.0 * 100   # in %

for exit_type in ['TAKE_PROFIT', 'STOP_LOSS', 'HORIZON']:
    subset = trades[trades['exit_reason'] == exit_type]
    if len(subset) == 0:
        continue
    
    if exit_type == 'TAKE_PROFIT':
        expected = subset['expected_win'].mean()
    elif exit_type == 'STOP_LOSS':
        expected = -subset['expected_loss'].mean()
    else:
        expected = 0
    
    actual = subset['pnl_pct'].mean()
    
    print(f"\n{exit_type}:")
    print(f"  Count: {len(subset)}")
    print(f"  Expected PnL: {expected:+.4f}%")
    print(f"  Actual PnL:   {actual:+.4f}%")
    print(f"  Gap:          {actual - expected:+.4f}%")

# ============================================================
# 3. SLIPPAGE & FEE IMPACT
# ============================================================
print("\n3. SLIPPAGE & FEE ANALYSIS")
print("-" * 50)

# Constants from backtest.py
SLIPPAGE = 0.001  # 0.1%
TRADING_FEE = 0.00075  # 0.075%

# Entry slippage (we buy at worse price)
entry_slippage_cost = SLIPPAGE * 100
# Exit slippage (varies but generally worse)
exit_slippage_cost = SLIPPAGE * 100
# Round-trip fees
round_trip_fee = TRADING_FEE * 2 * 100

total_execution_cost = entry_slippage_cost + exit_slippage_cost + round_trip_fee

print(f"Entry Slippage: {entry_slippage_cost:.3f}%")
print(f"Exit Slippage:  {exit_slippage_cost:.3f}%")
print(f"Round-trip Fee: {round_trip_fee:.3f}%")
print(f"TOTAL EXECUTION COST: {total_execution_cost:.3f}%")

avg_atr = trades['atr_pct'].mean() * 100
print(f"\nAverage ATR: {avg_atr:.3f}%")
print(f"Expected TP (2 ATR): {avg_atr * 2:.3f}%")
print(f"Expected SL (1 ATR): {avg_atr * 1:.3f}%")
print(f"\nExecution cost as % of expected TP: {total_execution_cost / (avg_atr * 2) * 100:.1f}%")
print(f"Execution cost as % of expected SL: {total_execution_cost / (avg_atr * 1) * 100:.1f}%")

# ============================================================
# 4. INDIVIDUAL TRADE ANALYSIS
# ============================================================
print("\n4. SAMPLE TRADES ANALYSIS")
print("-" * 50)

print("\nSample TAKE_PROFIT trades:")
tp_trades = trades[trades['exit_reason'] == 'TAKE_PROFIT'].head(5)
for _, t in tp_trades.iterrows():
    expected = t['atr_pct'] * 2 * 100
    actual = t['pnl_pct']
    ratio = actual / expected if expected != 0 else 0
    print(f"  ATR: {t['atr_pct']*100:.3f}%, Expected: +{expected:.3f}%, Actual: {actual:+.3f}%, Ratio: {ratio:.2f}")

print("\nSample STOP_LOSS trades:")
sl_trades = trades[trades['exit_reason'] == 'STOP_LOSS'].head(5)
for _, t in sl_trades.iterrows():
    expected = -t['atr_pct'] * 1 * 100
    actual = t['pnl_pct']
    ratio = actual / expected if expected != 0 else 0
    print(f"  ATR: {t['atr_pct']*100:.3f}%, Expected: {expected:.3f}%, Actual: {actual:+.3f}%, Ratio: {ratio:.2f}")

# ============================================================
# 5. THE REAL CULPRIT: ATR CALCULATION
# ============================================================
print("\n5. ATR DISCREPANCY CHECK")
print("-" * 50)

# Load test data to check ATR values
test_df = pd.read_csv('testing/test_final.csv')

print(f"ATR in trade log - Min: {trades['atr_pct'].min()*100:.4f}%, Max: {trades['atr_pct'].max()*100:.4f}%")
print(f"ATR in test data - Min: {test_df['atr_pct'].min()*100:.4f}%, Max: {test_df['atr_pct'].max()*100:.4f}%")

# ============================================================
# 6. R:R CALCULATION ISSUE
# ============================================================
print("\n6. R:R RATIO AT TRADE LEVEL")
print("-" * 50)

# For true 1:2 R:R, every win should be ~2x every loss
# Let's calculate what the R:R would be if we didn't have slippage/fees

# Theoretical PnL without execution costs
tp_theoretical = wins['atr_pct'].mean() * 2 * 100
sl_theoretical = losses['atr_pct'].mean() * 1 * 100

print(f"Theoretical TP (no costs): +{tp_theoretical:.4f}%")
print(f"Theoretical SL (no costs): -{sl_theoretical:.4f}%")
print(f"Theoretical R:R: 1:{tp_theoretical/sl_theoretical:.2f}")

print(f"\nActual TP: +{wins['pnl_pct'].mean():.4f}%")
print(f"Actual SL: -{abs(losses['pnl_pct'].mean()):.4f}%")
print(f"Actual R:R: 1:{wins['pnl_pct'].mean() / abs(losses['pnl_pct'].mean()):.2f}")

# Loss from execution costs
tp_loss = tp_theoretical - wins['pnl_pct'].mean()
sl_extra = abs(losses['pnl_pct'].mean()) - sl_theoretical

print(f"\n>>> TP DEGRADATION: -{tp_loss:.4f}% (execution costs)")
print(f">>> SL WORSENING:  +{sl_extra:.4f}% (execution costs)")

# ============================================================
# 7. EXIT PRICE VS TARGET PRICE
# ============================================================
print("\n7. EXIT PRICE ACCURACY")
print("-" * 50)

# Check if TP/SL are being hit at exact prices or worse
print("This requires looking at the backtest logic...")
print("\nKey findings from code review:")
print("- Entry: Close price * (1 + SLIPPAGE) = 0.1% worse")
print("- TP Exit: Uses min(tp_price, candle_high) * (1 - SLIPPAGE)")  
print("- SL Exit: Uses max(sl_price, candle_low) * (1 - SLIPPAGE)")
print("\nBOTH entries and exits have slippage applied, causing double-hit on costs!")
