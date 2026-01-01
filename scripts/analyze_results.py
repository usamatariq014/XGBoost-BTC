import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score

# ============================================================
# 1. TRADE LOG ANALYSIS
# ============================================================
print("=" * 60)
print("1. TRADE LOG ANALYSIS")
print("=" * 60)

trade_log = pd.read_csv('backtest_trade_log.csv')
print(f"Total trades: {len(trade_log)}")
print(f"\nExit Reasons:")
print(trade_log['exit_reason'].value_counts())

print(f"\nPnL Statistics by Exit Reason:")
print(trade_log.groupby('exit_reason')['pnl_pct'].agg(['mean', 'std', 'count']))

print(f"\nActual R:R Analysis:")
avg_win = trade_log[trade_log['pnl_pct'] > 0]['pnl_pct'].mean()
avg_loss = abs(trade_log[trade_log['pnl_pct'] <= 0]['pnl_pct'].mean())
print(f"Average Win: {avg_win:.2f}%")
print(f"Average Loss: {avg_loss:.2f}%")
print(f"Actual R:R Ratio: 1:{avg_win/avg_loss:.2f}")

# ============================================================
# 2. TARGET DISTRIBUTION COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("2. TARGET DISTRIBUTION (TRAIN vs TEST)")
print("=" * 60)

train = pd.read_csv('training/train_final.csv')
test = pd.read_csv('testing/test_final.csv')

print(f"Training Period - Wins: {train['target'].sum()} / {len(train)} ({train['target'].mean()*100:.2f}%)")
print(f"Testing Period  - Wins: {test['target'].sum()} / {len(test)} ({test['target'].mean()*100:.2f}%)")

# ============================================================
# 3. MODEL PROBABILITY THRESHOLD ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("3. MODEL THRESHOLD ANALYSIS ON TEST DATA")
print("=" * 60)

drop_cols = [
    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Quote asset volume', 'Number of trades',
    'Taker buy base asset volume', 'Taker buy quote asset volume',
    'Ignore', 'target', 'Close time'
]

X_test = test.drop(columns=[c for c in drop_cols if c in test.columns])
y_test = test['target']

model = joblib.load('models/xgb_model.pkl')

# Use raw features (no polynomial expansion)
probs = model.predict_proba(X_test)[:, 1]

print(f"{'Threshold':<12} {'Trades':<10} {'Wins':<10} {'Win Rate':<12} {'Precision':<12}")
print("-" * 56)

for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    preds = (probs >= thresh).astype(int)
    num_trades = preds.sum()
    if num_trades < 1:
        continue
    wins = ((preds == 1) & (y_test == 1)).sum()
    win_rate = wins / num_trades * 100
    precision = precision_score(y_test, preds, zero_division=0) * 100
    print(f"{thresh:<12} {num_trades:<10} {wins:<10} {win_rate:<12.1f} {precision:<12.1f}")

# ============================================================
# 4. PROBABILITY DISTRIBUTION
# ============================================================
print("\n" + "=" * 60)
print("4. PROBABILITY DISTRIBUTION")
print("=" * 60)

print(f"Prob Distribution on Test Set:")
print(f"  Min: {probs.min():.3f}")
print(f"  Max: {probs.max():.3f}")
print(f"  Mean: {probs.mean():.3f}")
print(f"  Median: {np.median(probs):.3f}")
print(f"  Signals >= 0.75: {(probs >= 0.75).sum()}")
print(f"  Signals >= 0.80: {(probs >= 0.80).sum()}")

# ============================================================
# 5. TIME-BASED ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("5. MONTHLY WIN RATE FROM TRADE LOG")
print("=" * 60)

trade_log['entry_time'] = pd.to_datetime(trade_log['entry_time'])
trade_log['month'] = trade_log['entry_time'].dt.to_period('M')

monthly = trade_log.groupby('month').agg({
    'pnl_pct': ['count', lambda x: (x > 0).mean() * 100, 'sum']
}).round(2)
monthly.columns = ['Trades', 'Win Rate %', 'Total PnL %']
print(monthly)

# ============================================================
# 6. FEATURE IMPORTANCE CHECK
# ============================================================
print("\n" + "=" * 60)
print("6. TOP 15 FEATURE IMPORTANCES")
print("=" * 60)

feature_names = X_test.columns
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df.head(15).to_string(index=False))

# ============================================================
# 7. CRITICAL FINDING: COMPARE TRAIN VS TEST PRECISION
# ============================================================
print("\n" + "=" * 60)
print("7. OVERFITTING CHECK: TRAIN vs TEST PRECISION")
print("=" * 60)

X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
y_train = train['target']

# Use raw features (no polynomial expansion)
train_probs = model.predict_proba(X_train)[:, 1]

for thresh in [0.5, 0.75]:
    train_preds = (train_probs >= thresh).astype(int)
    test_preds = (probs >= thresh).astype(int)
    
    train_precision = precision_score(y_train, train_preds, zero_division=0) * 100
    test_precision = precision_score(y_test, test_preds, zero_division=0) * 100
    
    print(f"Threshold {thresh}:")
    print(f"  Train Precision: {train_precision:.1f}%")
    print(f"  Test Precision:  {test_precision:.1f}%")
    print(f"  Gap (Overfitting indicator): {train_precision - test_precision:.1f}%")
    print()
