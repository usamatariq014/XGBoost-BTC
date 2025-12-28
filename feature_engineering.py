import pandas as pd
import numpy as np
import talib
import logging
import sys
from pathlib import Path

# --- Configuration ---
TRAIN_INPUT_PATH = Path("training/train.csv")
TEST_INPUT_PATH = Path("testing/test.csv")

TRAIN_OUTPUT_PATH = Path("training/train_features.csv")
TEST_OUTPUT_PATH = Path("testing/test_features.csv")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.required_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Open time', 'Taker buy base asset volume'
        ]

    def validate_data(self, df: pd.DataFrame) -> bool:
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        return True

    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Trend features using Distance to EMA (Percentage).
        """
        close = df['Close'].values
        
        # EMA 10 (Short-term momentum)
        ema10 = talib.EMA(close, timeperiod=10)
        df['dist_ema10'] = (close - ema10) / ema10

        # EMA 20 (Pullback zone)
        ema20 = talib.EMA(close, timeperiod=20)
        df['dist_ema20'] = (close - ema20) / ema20

        # EMA 50 (Medium trend)
        ema50 = talib.EMA(close, timeperiod=50)
        df['dist_ema50'] = (close - ema50) / ema50

        # EMA 200 (Macro trend filter)
        ema200 = talib.EMA(close, timeperiod=200)
        df['dist_ema200'] = (close - ema200) / ema200
        
        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        # Normalized ATR (Volatility relative to price)
        atr = talib.ATR(high, low, close, timeperiod=14)
        df['atr_pct'] = atr / close

        # Bollinger Band Width (Squeeze Detector)
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        middle = np.where(middle == 0, np.nan, middle) 
        df['bb_width'] = (upper - lower) / middle
        
        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df['Volume'].values
        taker_vol = df['Taker buy base asset volume'].values
        
        # Relative Volume (Current Vol / Avg Vol 20)
        avg_vol = talib.SMA(vol, timeperiod=20)
        df['rel_vol'] = vol / (avg_vol + 1e-6)

        # Taker Buy Ratio (Aggressive Buy Pressure)
        df['taker_buy_ratio'] = taker_vol / (vol + 1e-6)
        
        return df

    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Candlestick Patterns (100 = Bull, -100 = Bear, 0 = None)."""
        open_p = df['Open'].values
        high_p = df['High'].values
        low_p = df['Low'].values
        close_p = df['Close'].values
        
        # Strong Reversal Patterns
        df['cdl_engulfing'] = talib.CDLENGULFING(open_p, high_p, low_p, close_p)
        df['cdl_hammer'] = talib.CDLHAMMER(open_p, high_p, low_p, close_p)
        df['cdl_shootingstar'] = talib.CDLSHOOTINGSTAR(open_p, high_p, low_p, close_p)
        
        # Indecision Pattern
        df['cdl_doji'] = talib.CDLDOJI(open_p, high_p, low_p, close_p)
        
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not np.issubdtype(df['Open time'].dtype, np.datetime64):
             df['Open time'] = pd.to_datetime(df['Open time'])
             
        # Hour encoding (Cyclical)
        df['hour_sin'] = np.sin(2 * np.pi * df['Open time'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Open time'].dt.hour / 24)
        
        # Day of week
        df['day_of_week'] = df['Open time'].dt.dayofweek
        
        return df

    def process_file(self, input_path: Path, output_path: Path):
        logger.info(f"--- Processing File: {input_path} ---")
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        try:
            df = pd.read_csv(input_path)
            
            if not self.validate_data(df):
                return

            df = self.add_trend_features(df)
            df = self.add_volatility_features(df)
            df = self.add_volume_features(df)
            df = self.add_pattern_features(df)
            df = self.add_time_features(df)

            # Cleanup Warmup Period
            df_clean = df.dropna()
            dropped_count = len(df) - len(df_clean)
            
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows (Indicator Warmup Period).")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(output_path, index=False)
            logger.info(f"SUCCESS. Saved features to: {output_path}")

        except Exception as e:
            logger.exception(f"Critical Error processing {input_path}: {e}")

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.process_file(TRAIN_INPUT_PATH, TRAIN_OUTPUT_PATH)
    engineer.process_file(TEST_INPUT_PATH, TEST_OUTPUT_PATH)