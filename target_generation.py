import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
TRAIN_FEATURES_PATH = Path("training/train_features.csv")
TEST_FEATURES_PATH = Path("testing/test_features.csv")

TRAIN_FINAL_PATH = Path("training/train_final.csv")
TEST_FINAL_PATH = Path("testing/test_final.csv")

# Horizon: 24 Candles (24 Hours)
HORIZON = 24 
# Risk Ratio: 1:2 (Risk 1 to make 2)
ATR_MULTIPLIER_SL = 1.0
ATR_MULTIPLIER_TP = 2.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def triple_barrier_labeling(df: pd.DataFrame, horizon: int, sl_mult: float, tp_mult: float):
    atr_raw = df['atr_pct'] * df['Close']
    entries = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    stop_losses = entries - (atr_raw.values * sl_mult)
    take_profits = entries + (atr_raw.values * tp_mult)
    
    labels = []
    num_rows = len(df)
    
    for i in tqdm(range(num_rows), desc="Generating Targets"):
        if i + horizon >= num_rows:
            labels.append(0)
            continue
            
        future_highs = highs[i+1 : i+1+horizon]
        future_lows = lows[i+1 : i+1+horizon]
        
        target_price = take_profits[i]
        stop_price = stop_losses[i]
        
        tp_hit_indices = np.where(future_highs >= target_price)[0]
        sl_hit_indices = np.where(future_lows <= stop_price)[0]
        
        if len(tp_hit_indices) == 0 and len(sl_hit_indices) == 0:
            labels.append(0)
        elif len(sl_hit_indices) > 0 and len(tp_hit_indices) == 0:
            labels.append(0)
        elif len(tp_hit_indices) > 0 and len(sl_hit_indices) == 0:
            labels.append(1)
        else:
            if tp_hit_indices[0] < sl_hit_indices[0]:
                labels.append(1)
            else:
                labels.append(0)
                
    return labels

# Replace the process_file function with this version:
def process_file(input_path: Path, output_path: Path):
    logger.info(f"--- Processing Target Generation: {input_path} ---")
    if not input_path.exists(): 
        logger.error(f"Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    # CRITICAL: Convert Open time to datetime if needed
    if not np.issubdtype(df['Open time'].dtype, np.datetime64):
        df['Open time'] = pd.to_datetime(df['Open time'])
    
    targets = triple_barrier_labeling(df, HORIZON, ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP)
    df['target'] = targets
    
    wins = sum(targets)
    logger.info(f"Target Distribution: {wins} Wins out of {len(targets)} rows ({wins/len(targets):.2%})")
    
    # Drop the last HORIZON rows (no future data to calculate targets)
    df_final = df.iloc[:-HORIZON].copy()
    logger.info(f"Dropped last {HORIZON} rows (no future data for target calculation)")
    
    df_final.to_csv(output_path, index=False)
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Final shape: {df_final.shape}")

if __name__ == "__main__":
    process_file(TRAIN_FEATURES_PATH, TRAIN_FINAL_PATH)
    process_file(TEST_FEATURES_PATH, TEST_FINAL_PATH)