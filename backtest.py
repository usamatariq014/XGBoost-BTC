import pandas as pd
import joblib
import matplotlib.pyplot as plt
import logging
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
TEST_DATA_PATH = Path("testing/test_final.csv")
MODEL_PATH = Path("models/xgb_model.pkl")

# Strategy Settings
THRESHOLD = 0.55       # Balance between quality and quantity
RISK_PER_TRADE = 0.02  # Risk 2% of account per trade
STARTING_BALANCE = 10000  # More realistic starting amount
TRADING_FEE = 0.00075  # 0.075% per trade (Binance spot fee)
SLIPPAGE = 0.001       # 0.1% slippage for entries/exits

# Risk Management
MAX_LEVERAGE = 1.0     # 1x for spot trading (no leverage)
MAX_DRAWDOWN = 0.25    # 25% max drawdown stop
HORIZON = 24           # 24-hour trade window
ATR_MULTIPLIER_SL = 1.0
ATR_MULTIPLIER_TP = 1.5   # Reduced from 2.0 for higher win rate (1:1.5 R:R still profitable)

# Trailing Stop Loss Settings
TRAILING_ACTIVATION = 1.0  # Activate trailing at 1R profit (1 ATR)
TRAILING_DISTANCE = 0.5    # Trail by 0.5 ATR behind price

# Filters
TREND_FILTER = True        # Only trade when price > EMA200
VOLATILITY_FILTER = False  # Disabled - too restrictive
MAX_ATR_PCT = 0.03         # Max ATR 3% for volatility filter (if enabled)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self):
        # EXACT SAME COLUMNS DROPPED DURING TRAINING
        self.drop_cols = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Quote asset volume', 'Number of trades', 
            'Taker buy base asset volume', 'Taker buy quote asset volume', 
            'Ignore', 'target', 'Close time'
        ]
        
        self.start_date = None  # Will be set during data loading

    def prepare_features(self, df):
        """Prepare features EXACTLY as done during training (raw features, no polynomial expansion)"""
        # Drop non-feature columns, return raw features
        features = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        return features

    def calculate_trade_outcome(self, entry_row, future_data):
        """
        Simulate a single trade with TRAILING STOP LOSS.
        Returns: (pnl, exit_reason, hold_time)
        
        Features:
        1. OHLC-based priority for same-candle TP/SL hits
        2. Trailing stop activates at 1R profit
        3. Trail distance is 0.5 ATR behind highest price
        """
        entry_price = entry_row['Close']
        atr_pct = entry_row['atr_pct']
        atr_value = entry_price * atr_pct
        
        # Apply slippage to entry
        entry_price_with_slippage = entry_price * (1 + SLIPPAGE)
        
        # Calculate initial SL and TP from execution price
        initial_sl = entry_price_with_slippage * (1 - (atr_pct * ATR_MULTIPLIER_SL))
        tp_price = entry_price_with_slippage * (1 + (atr_pct * ATR_MULTIPLIER_TP))
        
        # Trailing stop state
        current_sl = initial_sl
        trailing_active = False
        highest_price = entry_price_with_slippage
        
        # Activation price for trailing (1R profit)
        trailing_activation_price = entry_price_with_slippage * (1 + (atr_pct * TRAILING_ACTIVATION))
        trailing_distance = atr_pct * TRAILING_DISTANCE
        
        # Track trade outcome
        exit_price = None
        exit_reason = "HORIZON"
        hold_time = HORIZON
        
        # Check each future candle
        for idx, (i, row) in enumerate(future_data.iterrows()):
            candle_open = row['Open']
            candle_high = row['High']
            candle_low = row['Low']
            candle_close = row['Close']
            
            # Update highest price seen (for trailing)
            if candle_high > highest_price:
                highest_price = candle_high
                
                # If trailing is active, update SL
                if trailing_active:
                    new_trailing_sl = highest_price * (1 - trailing_distance)
                    current_sl = max(current_sl, new_trailing_sl)
            
            # Check if trailing should activate (price reached 1R profit)
            if not trailing_active and candle_high >= trailing_activation_price:
                trailing_active = True
                # Move SL to at least breakeven
                breakeven = entry_price_with_slippage
                current_sl = max(current_sl, breakeven)
                # Start trailing from current highest
                new_trailing_sl = highest_price * (1 - trailing_distance)
                current_sl = max(current_sl, new_trailing_sl)
            
            # Check for exits
            tp_hit = candle_high >= tp_price
            sl_hit = candle_low <= current_sl
            
            if tp_hit and sl_hit:
                # Both hit - use OHLC order
                if candle_close >= candle_open:
                    # Bullish: Low hit first (SL)
                    exit_price = current_sl
                    exit_reason = "TRAILING_SL" if trailing_active else "STOP_LOSS"
                else:
                    # Bearish: High hit first (TP)
                    exit_price = tp_price
                    exit_reason = "TAKE_PROFIT"
                hold_time = idx + 1
                break
            elif tp_hit:
                exit_price = tp_price
                exit_reason = "TAKE_PROFIT"
                hold_time = idx + 1
                break
            elif sl_hit:
                exit_price = current_sl
                exit_reason = "TRAILING_SL" if trailing_active else "STOP_LOSS"
                hold_time = idx + 1
                break
        
        # If no exit triggered, close at horizon
        if exit_price is None:
            exit_price = future_data.iloc[-1]['Close']
        
        # Calculate PnL
        pnl_pct = (exit_price - entry_price_with_slippage) / entry_price_with_slippage
        
        return pnl_pct, exit_reason, hold_time

    def run_backtest(self):
        logger.info("--- Starting REALISTIC Backtest (No Cheating) ---")
        logger.info(f"Using threshold: {THRESHOLD}, Risk per trade: {RISK_PER_TRADE*100:.1f}%")
        
        if not TEST_DATA_PATH.exists():
            logger.error("Test data not found!")
            return
            
        if not MODEL_PATH.exists():
            logger.error("Model not found!")
            return
            
        # Load data and model
        df = pd.read_csv(TEST_DATA_PATH)
        model = joblib.load(MODEL_PATH)
        
        # Store start date for plotting
        self.start_date = pd.to_datetime(df['Open time'].iloc[0])
        
        # Convert datetime columns
        df['Open time'] = pd.to_datetime(df['Open time'])
        df = df.sort_values('Open time').reset_index(drop=True)
        
        # PREPARE FEATURES EXACTLY AS IN TRAINING
        logger.info("Preparing features with polynomial interactions...")
        features = self.prepare_features(df)
        
        logger.info(f"Model expects {len(model.feature_names_in_)} features")
        logger.info(f"Prepared features: {features.shape[1]} columns")
        
        # Verify feature alignment
        missing_features = [f for f in model.feature_names_in_ if f not in features.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features[:5]}...")
            return
        
        logger.info(f"Predicting on {len(df)} candles...")
        probs = model.predict_proba(features)[:, 1]
        
        # Initialize tracking variables
        balance = STARTING_BALANCE
        equity_curve = [balance]
        trade_log = []
        
        current_position = None  # {entry_index, entry_time, entry_price, position_size, atr_pct}
        backtest_ended_early = False
        
        # Main backtest loop
        for i in tqdm(range(len(df) - HORIZON), desc="Backtesting"):
            current_row = df.iloc[i]
            current_time = current_row['Open time']
            
            # Check for max drawdown stop
            if balance < STARTING_BALANCE * (1 - MAX_DRAWDOWN):
                logger.warning(f"Max drawdown reached at {current_time}. Stopping backtest.")
                # Extend equity curve to match full length
                remaining_periods = len(df) - i - 1
                equity_curve.extend([balance] * remaining_periods)
                backtest_ended_early = True
                break
            
            # Handle active position
            if current_position is not None:
                # Get future price data for trade simulation (include Open for OHLC priority)
                future_data = df.iloc[i+1:i+1+HORIZON][['Open', 'High', 'Low', 'Close']]
                
                # Calculate trade outcome
                pnl_pct, exit_reason, hold_time = self.calculate_trade_outcome(
                    df.iloc[current_position['entry_index']], 
                    future_data
                )
                
                # Calculate position value and fees
                position_value = current_position['position_size'] * (1 + pnl_pct)
                total_fees = current_position['position_size'] * TRADING_FEE * 2
                
                # Update balance
                balance = balance - current_position['position_size'] + position_value - total_fees
                
                # Log the trade
                trade_log.append({
                    'entry_time': current_position['entry_time'],
                    'exit_time': current_time + pd.Timedelta(hours=hold_time),
                    'entry_price': current_position['entry_price'],
                    'exit_price': current_position['entry_price'] * (1 + pnl_pct),
                    'position_size': current_position['position_size'],
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': position_value - current_position['position_size'] - total_fees,
                    'fees': total_fees,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time,
                    'atr_pct': current_position['atr_pct']
                })
                
                # Reset position
                current_position = None
            
            # Check for new signal (only if no active position)
            if current_position is None and probs[i] >= THRESHOLD:
                atr_pct = current_row['atr_pct']
                
                # TREND FILTER: Only trade when price > EMA200 (with the trend)
                if TREND_FILTER:
                    dist_ema200 = current_row['dist_ema200']
                    if dist_ema200 < 0:  # Price below EMA200
                        continue  # Skip counter-trend signal
                
                # VOLATILITY FILTER: Skip high-volatility periods
                if VOLATILITY_FILTER:
                    if atr_pct > MAX_ATR_PCT:
                        continue  # Skip volatile period
                
                # Calculate position size based on risk
                risk_amount = balance * RISK_PER_TRADE
                
                if atr_pct <= 0.0001:  # Avoid division by zero
                    continue
                
                # Position size = risk amount / (SL distance in %)
                sl_distance_pct = atr_pct * ATR_MULTIPLIER_SL
                ideal_position_size = risk_amount / sl_distance_pct
                
                # Apply leverage cap
                max_position_size = balance * MAX_LEVERAGE
                position_size = min(ideal_position_size, max_position_size)
                
                # Ensure we have enough capital
                if position_size < 10:  # Minimum $10 trade
                    continue
                
                # Enter position at current candle's close (simulated)
                current_position = {
                    'entry_index': i,
                    'entry_time': current_time,
                    'entry_price': current_row['Close'],
                    'position_size': position_size,
                    'atr_pct': atr_pct
                }
            
            # Record equity (even if no trade)
            equity_curve.append(balance)
        
        # Handle early termination
        if not backtest_ended_early and len(equity_curve) < len(df):
            equity_curve.extend([balance] * (len(df) - len(equity_curve)))
        
        # Trim equity curve to match data length
        equity_curve = equity_curve[:len(df)]
        
        # Analysis
        equity_curve = np.array(equity_curve)
        final_balance = equity_curve[-1]
        net_profit = final_balance - STARTING_BALANCE
        total_return = (net_profit / STARTING_BALANCE) * 100
        
        # Drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Trade statistics
        total_trades = len(trade_log)
        if total_trades > 0:
            trade_df = pd.DataFrame(trade_log)
            win_trades = trade_df[trade_df['pnl_pct'] > 0]
            win_rate = len(win_trades) / total_trades * 100
            avg_win = win_trades['pnl_pct'].mean() if len(win_trades) > 0 else 0
            avg_loss = trade_df[trade_df['pnl_pct'] <= 0]['pnl_pct'].mean() if len(trade_df[trade_df['pnl_pct'] <= 0]) > 0 else 0
            profit_factor = abs(win_trades['pnl_usd'].sum() / trade_df[trade_df['pnl_pct'] <= 0]['pnl_usd'].sum()) if len(trade_df[trade_df['pnl_pct'] <= 0]) > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Sharpe ratio approximation (hourly returns)
        hourly_returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = (hourly_returns.mean() / hourly_returns.std()) * np.sqrt(252 * 24) if hourly_returns.std() != 0 else 0
        
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS (REALISTIC, NO CHEATING)")
        logger.info("="*60)
        logger.info(f"INITIAL BALANCE: ${STARTING_BALANCE:,.2f}")
        logger.info(f"FINAL BALANCE:   ${final_balance:,.2f}")
        logger.info(f"TOTAL RETURN:    {total_return:.2f}%")
        logger.info(f"MAX DRAWDOWN:    {max_drawdown:.2f}%")
        logger.info(f"SHARPE RATIO:    {sharpe_ratio:.2f}")
        logger.info("-"*60)
        logger.info(f"TOTAL TRADES:    {total_trades}")
        logger.info(f"WIN RATE:        {win_rate:.2f}%")
        logger.info(f"AVG WIN:         {avg_win:.2f}%")
        logger.info(f"AVG LOSS:        {avg_loss:.2f}%")
        logger.info(f"PROFIT FACTOR:   {profit_factor:.2f}")
        logger.info("="*60)
        
        # Save trade log for analysis
        if trade_log:
            trade_df.to_csv("backtest_trade_log.csv", index=False)
            logger.info("Trade log saved to backtest_trade_log.csv")
        
        self.plot_results(equity_curve, drawdown, trade_log, len(df))

    def plot_results(self, equity, drawdown, trade_log, total_periods):
        """Generate professional backtest visualization"""
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), 
                                           gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. Equity Curve - USE STORED START DATE
        dates = pd.date_range(start=self.start_date, periods=total_periods, freq='H')
        ax1.plot(dates[:len(equity)], equity, label='Account Balance', color='#2E86AB', linewidth=2)
        ax1.set_title('Realistic Backtest Results (No Data Leakage)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Balance ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Mark trades on equity curve
        if trade_log:
            trade_df = pd.DataFrame(trade_log)
            for _, trade in trade_df.iterrows():
                trade_idx = (trade['entry_time'] - self.start_date).total_seconds() / 3600
                if 0 <= trade_idx < len(equity):
                    color = 'green' if trade['pnl_pct'] > 0 else 'red'
                    ax1.scatter(dates[int(trade_idx)], equity[int(trade_idx)], 
                               color=color, s=30, alpha=0.7, zorder=5)
        
        # 2. Drawdown
        ax2.fill_between(dates[:len(drawdown)], drawdown[:len(dates)] * 100, 0, color='#A23B72', alpha=0.3)
        ax2.set_title('Drawdown (%)', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(min(drawdown * 100) * 1.1, 0)
        
        # 3. Trade Outcomes
        if trade_log:
            trade_df = pd.DataFrame(trade_log)
            trade_df['month'] = trade_df['entry_time'].dt.to_period('M')
            monthly_win_rate = trade_df.groupby('month').apply(
                lambda x: (x['pnl_pct'] > 0).mean() * 100
            )
            
            ax3.bar(monthly_win_rate.index.astype(str), monthly_win_rate.values, 
                   color='#F18F01', alpha=0.8)
            ax3.set_title('Monthly Win Rate (%)', fontsize=12)
            ax3.set_ylabel('Win Rate (%)', fontsize=10)
            ax3.set_xticklabels(monthly_win_rate.index.astype(str), rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
        else:
            ax3.text(0.5, 0.5, 'No trades executed', ha='center', va='center', fontsize=12)
            ax3.set_title('Monthly Win Rate (%)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        logger.info("Backtest chart saved as backtest_results.png")
        plt.show()

if __name__ == "__main__":
    bot = Backtester()
    bot.run_backtest()