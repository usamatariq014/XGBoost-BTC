# Bitcoin Intraday Trading Bot

This is a systematic hourly Bitcoin (BTC) trading system built using XGBoost. The project focuses on creating a robust machine learning pipeline that handles data preparation, feature engineering with zero temporal leakage, target generation using triple barrier labeling, and realistic backtesting with fees and slippage.

The goal is to provide a framework for training a high confidence classifier that identifies profitable intraday trading opportunities while maintaining strict risk management.

## Dataset Information

The project utilizes historical Bitcoin data at a 1 hour (1H) interval. The dataset was sourced from the following collection:

Kaggle: [Bitcoin Historical Datasets (2018-2024)](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)

### Data Split
- Training Period: 2018-01-01 to 2023-12-31 (6 years)
- Testing Period: 2024-01-01 to mid-2025 (1.5 years)

## Project Structure

- cleaner.py: Handles initial data cleaning, datetime formatting, and time based train/test splitting.
- feature_engineering.py: Generates technical indicators (EMA, ATR, BB, Volume metrics) and candlestick patterns. All features are shifted to prevent look ahead bias.
- target_generation.py: Implements Triple Barrier Labeling with a 24 hour horizon and ATR based stop loss and take profit levels.
- train_model.py: Trains an XGBoost classifier with deep trees and polynomial feature interactions.
- backtest.py: Provides a realistic trading simulation including fees, slippage, and drawdown limits.
- Project_Infrastructure.md: Detailed documentation of the project architecture and lessons learned.

## Technical Implementation

### Feature Engineering
Features are grouped into four categories:
1. Trend: Distance to EMA 10, 20, 50, and 200.
2. Volatility: Normalized ATR percentage and Bollinger Band width.
3. Volume: Relative volume and Taker Buy Ratio.
4. Patterns: Candlestick patterns (Engulfing, Hammer, Shooting Star, Doji) using TA-Lib.

Crucially, all technical indicators are shifted by 1 period to ensure the model only uses data available at the time of the trade.

### Interaction Features
The model uses PolynomialFeatures (degree 2) to create interaction terms between base features, allowing the XGBoost model to capture complex relationships between trend, volatility, and volume.

### Triple Barrier Labeling
Targets are generated based on:
- Take Profit (TP): 2.0 x ATR
- Stop Loss (SL): 1.0 x ATR
- Vertical Barrier: 24 hours (Horizon)

A label of 1 is assigned only if the Take Profit is hit before the Stop Loss or the Horizon.

## Workflow and Execution

To run the complete pipeline, execute the following commands in sequence:

1. Prepare and split the data:
python cleaner.py

2. Generate engineered features:
python feature_engineering.py

3. Create labels using triple barrier method:
python target_generation.py

4. Train the XGBoost model:
python train_model.py

5. Run the realistic backtest:
python backtest.py

## Performance and Risk Management

The system is designed to only trade high probability signals (threshold >= 0.75).

### Key Metrics (Reference)
- Strategy: Fixed 1:2 Risk-Reward Ratio.
- Position Sizing: Based on account balance and current volatility (ATR).
- Safety Stop: Max drawdown limit set at 25%.
- Execution Costs: 0.1% slippage and 0.075% fee per trade.

## Installation

This project requires Python 3.10+ and the following key libraries:
- pandas
- numpy
- xgboost
- TA-Lib
- scikit-learn
- matplotlib
- joblib
- tqdm
