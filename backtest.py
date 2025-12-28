import pandas as pd
import joblib
import matplotlib.pyplot as plt
import logging
import sys
import numpy as np
from pathlib import Path

# --- Configuration ---
TEST_DATA_PATH = Path("testing/test_final.csv")
MODEL_PATH = Path("models/xgb_model.pkl")

# Strategy Settings
THRESHOLD = 0.75       # Trying the same threshold first
RISK_PER_TRADE = 0.02  # Risk 2%
STARTING_BALANCE = 1000
TRADING_FEE = 0.00075  # 0.075%

# *** THE FIX ***
MAX_LEVERAGE = 5.0     # Cap leverage at 1x (Spot Mode). Set to 3.0 or 5.0 for Futures.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self):
        self.drop_cols = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Quote asset volume', 'Number of trades', 
            'Taker buy base asset volume', 'Taker buy quote asset volume', 
            'Ignore', 'target', 'Close time'
        ]

    def run_backtest(self):
        logger.info("--- Starting Corrected Backtest (Spot Mode) ---")
        
        if not TEST_DATA_PATH.exists():
            logger.error("Test data not found!")
            return
            
        df = pd.read_csv(TEST_DATA_PATH)
        model = joblib.load(MODEL_PATH)
        
        raw_df = df.copy()
        features = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        
        logger.info(f"Predicting on {len(df)} candles...")
        probs = model.predict_proba(features)[:, 1]
        
        balance = STARTING_BALANCE
        equity_curve = [balance]
        
        wins = 0
        losses = 0
        skipped_low_vol = 0
        
        for i in range(len(df)):
            if probs[i] < THRESHOLD:
                equity_curve.append(balance)
                continue
                
            # 1. Calculate Ideal Position Size based on Risk
            risk_amount = balance * RISK_PER_TRADE
            atr_pct = raw_df.iloc[i]['atr_pct']
            
            if atr_pct <= 0: 
                equity_curve.append(balance)
                continue
            
            ideal_position_size = risk_amount / atr_pct
            
            # 2. CAP THE LEVERAGE (The Fix)
            # Maximum allowed position is Balance * Max_Leverage
            max_position_size = balance * MAX_LEVERAGE
            
            # Take the smaller of the two
            actual_position_size = min(ideal_position_size, max_position_size)
            
            # If we are capped, our 'actual' risk is lower than 2%
            # actual_risk = actual_position_size * atr_pct
            
            # 3. Calculate Fees
            fees = actual_position_size * TRADING_FEE * 2
            
            # 4. Result
            result = raw_df.iloc[i]['target']
            
            pnl = 0
            if result == 1:
                # Profit = (Position * 2 * ATR%) - Fees
                gross_profit = actual_position_size * (atr_pct * 2) 
                pnl = gross_profit - fees
                wins += 1
            else:
                # Loss = (Position * 1 * ATR%) + Fees
                gross_loss = actual_position_size * atr_pct
                pnl = -gross_loss - fees
                losses += 1
            
            balance += pnl
            equity_curve.append(balance)
            
            if balance <= 0:
                logger.warning("Account blown! Game over.")
                break
            
        # Analysis
        equity_curve = np.array(equity_curve)
        total_trades = wins + losses
        final_balance = equity_curve[-1]
        net_profit = final_balance - STARTING_BALANCE
        
        # Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        logger.info("="*40)
        logger.info(f"FINAL BALANCE:  ${final_balance:.2f}")
        logger.info(f"NET PROFIT:     ${net_profit:.2f} ({(net_profit/STARTING_BALANCE)*100:.2f}%)")
        logger.info(f"MAX DRAWDOWN:   {max_drawdown:.2f}%")
        logger.info(f"TOTAL TRADES:   {total_trades}")
        logger.info("="*40)
        
        self.plot_results(equity_curve, drawdown)

    def plot_results(self, equity, drawdown):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(equity, label='Account Balance', color='blue')
        ax1.set_title(f'Corrected Equity Curve (Leverage Cap={MAX_LEVERAGE}x)')
        ax1.set_ylabel('Balance ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax2.fill_between(range(len(drawdown)), drawdown * 100, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    bot = Backtester()
    bot.run_backtest()