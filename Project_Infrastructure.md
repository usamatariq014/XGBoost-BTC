AlgoXG Project Infrastructure Documentation
Project Goal
Build a systematic hourly Bitcoin trading system using machine learning with proper risk management and realistic performance evaluation.

File Structure
AlgoXG/
├── training/
│   ├── btc_1h_data_2018_to_2025.csv  # Raw historical data
│   ├── train.csv                     # Cleaned training data (pre-2024)
│   ├── train_features.csv            # Engineered features
│   └── train_final.csv               # Features + targets
├── testing/
│   ├── test.csv                      # Cleaned test data (post-2024)
│   ├── test_features.csv             # Engineered features
│   └── test_final.csv                # Features + targets
├── models/
│   └── xgb_model.pkl                 # Trained XGBoost model
├── cleaner.py                       # Data preparation
├── feature_engineering.py           # Feature creation (no temporal leakage)
├── target_generation.py             # Triple barrier labeling
├── train_model.py                   # Model training & evaluation
└── backtest.py                      # Realistic performance testing
Core Workflow
1. Data Preparation (cleaner.py)
Input: Raw BTC 1-hour OHLCV data (2018-2025)
Processing:
Clean datetime formatting
Sort chronologically
Split at 2024-01-01 (train: pre-2024, test: post-2024)
Output: training/train.csv, testing/test.csv
2. Feature Engineering (feature_engineering.py)
Critical: Zero temporal leakage implementation
Features:
Trend: Distance to EMA10/20/50/200 (%)
Volatility: ATR%, Bollinger Band Width
Volume: Relative volume, Taker buy ratio
Patterns: Engulfing, Hammer, Shooting Star, Doji
Time: Hour (sin/cos encoding), Day of week
Key Implementation:
All technical indicators shifted by 1 period
Separate processing for train/test sets
Warmup period cleanup (200 rows dropped)
Output: Feature CSV files with 27 columns
3. Target Generation (target_generation.py)
Method: Triple barrier labeling
Parameters:
Horizon: 24 hours
Stop Loss: 1.0 × ATR
Take Profit: 2.0 × ATR
Processing:
Forward-looking target calculation
Drop last HORIZON rows (no future data)
Output: CSV files with target column (0/1)
4. Model Training (train_model.py)
Algorithm: XGBoost Classifier
Key Settings:
n_estimators=1500, learning_rate=0.05, max_depth=10
scale_pos_weight for class imbalance
Polynomial feature interactions (degree=2, interaction_only=True)
Evaluation:
Probability threshold analysis
Feature importance visualization
Output: models/xgb_model.pkl, feature importance plot
5. Realistic Backtesting (backtest.py)
Critical: No use of future information
Simulated Trading:
Slippage: 0.1% on all entries/exits
Fees: 0.075% per trade (Binance spot)
Hourly barrier checking (no pre-computed targets)
Risk Management:
Max drawdown stop: 25%
Position sizing based on account balance and ATR
Output: Performance metrics, equity curve plot, trade log
Critical Technical Fixes
Issue
Solution
Temporal Leakage
.shift(1) applied to all features; separate train/test processing
Feature Mismatch
Exact replication of training feature pipeline in backtest
Future Data Usage
No 'target' column used during backtesting; only raw price data
Scope Errors
Proper date handling and equity curve management
Performance Metrics (Threshold 0.75)
Win Rate: 35.16% (91 trades over 1.5 years test period)
Profit Factor: 0.53 (Gross Wins / Gross Losses)
Max Drawdown: -29.67% (stopped at -25% safety limit)
Total Return: -25.50% after fees and slippage
Sharpe Ratio: -1.64 (risk-adjusted performance)
Hardware Specifications
Machine: Lenovo L14 Gen 5
RAM: 64GB
Processor: Intel Core Ultra 7
Capabilities: Efficient training of complex models and large-scale backtesting
Execution Sequence
powershell
# 1. Prepare data
python cleaner.py

# 2. Generate features (CRITICAL: no leakage version)
python feature_engineering.py

# 3. Create targets
python target_generation.py

# 4. Train model
python train_model.py

# 5. Realistic backtest
python backtest.py

Critical Success Factors
Temporal Integrity: Never use future data in feature calculation
Realistic Simulation: Backtest must exclude target column and model actual execution
Volatility Awareness: Position sizing must adapt to market conditions
Risk Management: Max drawdown limits are non-negotiable
Threshold Discipline: Only trade high-confidence signals (≥0.75 probability)
Document generated on December 28, 2025. This captures the complete infrastructure and lessons learned from the initial AlgoXG implementation.