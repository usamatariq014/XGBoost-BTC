import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
import logging
import sys
from pathlib import Path
import joblib

# --- Configuration ---
TRAIN_PATH = Path("training/train_final.csv")
TEST_PATH = Path("testing/test_final.csv")
MODEL_SAVE_PATH = Path("models/xgb_model.pkl")

# Walk-Forward Validation Settings
N_SPLITS = 3  # Number of chronological folds

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        # Drop non-feature columns
        self.drop_cols = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Quote asset volume', 'Number of trades', 
            'Taker buy base asset volume', 'Taker buy quote asset volume', 
            'Ignore', 'target', 'Close time'
        ]

    def load_data(self):
        """Load data WITHOUT polynomial feature expansion to prevent overfitting"""
        logger.info("Loading Datasets...")
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        # Extract features and targets directly (NO polynomial features)
        X_train = train_df.drop(columns=[c for c in self.drop_cols if c in train_df.columns])
        y_train = train_df['target']
        
        X_test = test_df.drop(columns=[c for c in self.drop_cols if c in test_df.columns])
        y_test = test_df['target']
        
        logger.info(f"Feature Count: {X_train.shape[1]} (raw features, no polynomial expansion)")
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return X_train, y_train, X_test, y_test

    def walk_forward_validation(self, X_train, y_train):
        """
        Walk-Forward Validation: Train on past, validate on future.
        This prevents overfitting by simulating real trading conditions.
        """
        logger.info(f"--- Running Walk-Forward Validation ({N_SPLITS} folds) ---")
        
        fold_size = len(X_train) // N_SPLITS
        val_precisions = []
        
        for fold in range(1, N_SPLITS):
            # Train on all data up to this fold
            train_end = fold * fold_size
            val_start = train_end
            val_end = val_start + fold_size
            
            X_fold_train = X_train.iloc[:train_end]
            y_fold_train = y_train.iloc[:train_end]
            X_fold_val = X_train.iloc[val_start:val_end]
            y_fold_val = y_train.iloc[val_start:val_end]
            
            # Train a quick model for validation
            model = self._create_model(y_fold_train)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            # Evaluate at threshold 0.75
            probs = model.predict_proba(X_fold_val)[:, 1]
            preds = (probs >= 0.75).astype(int)
            
            if preds.sum() > 0:
                precision = precision_score(y_fold_val, preds, zero_division=0)
                val_precisions.append(precision)
                logger.info(f"Fold {fold}: Train={len(X_fold_train)}, Val={len(X_fold_val)}, "
                           f"Signals={preds.sum()}, Precision={precision:.2%}")
            else:
                logger.info(f"Fold {fold}: No signals at threshold 0.75")
        
        if val_precisions:
            avg_precision = np.mean(val_precisions)
            logger.info(f"Walk-Forward Avg Precision: {avg_precision:.2%}")
        
        return val_precisions

    def _create_model(self, y_train):
        """Create XGBoost model with anti-overfitting configuration"""
        # Balance Weight
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_weight = num_neg / num_pos
        
        return xgb.XGBClassifier(
            objective='binary:logistic',
            # REDUCED COMPLEXITY (was: 1500 estimators, depth 10)
            n_estimators=200,
            learning_rate=0.01,
            max_depth=4,
            # REGULARIZATION (NEW)
            reg_alpha=1.0,          # L1 regularization
            reg_lambda=1.0,         # L2 regularization
            min_child_weight=10,    # Prevent small leaf nodes
            gamma=0.1,              # Min loss reduction for split
            # Sampling
            subsample=0.8,
            colsample_bytree=0.8,
            # Class balance
            scale_pos_weight=scale_weight,
            # Other
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30
        )

    def train(self, X_train, y_train, X_test, y_test):
        """Train the final model on full training data"""
        logger.info("--- Training Final Model (Regularized, Shallow Trees) ---")
        
        model = self._create_model(y_train)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        logger.info("Training Complete.")
        return model

    def evaluate_thresholds(self, model, X_train, y_train, X_test, y_test):
        """Evaluate model at different thresholds and check for overfitting"""
        logger.info("--- Running Threshold Analysis ---")
        
        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        
        print("\n" + "=" * 70)
        print(f"{'THRESHOLD':<10} | {'TRAIN PREC':<12} | {'TEST PREC':<12} | {'GAP':<8} | {'TRADES'}")
        print("=" * 70)
        
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            train_preds = (train_probs >= thresh).astype(int)
            test_preds = (test_probs >= thresh).astype(int)
            
            train_trades = train_preds.sum()
            test_trades = test_preds.sum()
            
            if test_trades < 5:
                continue
            
            train_precision = precision_score(y_train, train_preds, zero_division=0) * 100
            test_precision = precision_score(y_test, test_preds, zero_division=0) * 100
            gap = train_precision - test_precision
            
            print(f"{thresh:<10} | {train_precision:>10.1f}% | {test_precision:>10.1f}% | {gap:>6.1f}% | {test_trades}")
        
        print("=" * 70)
        print("TARGET: Gap should be < 15% (indicates good generalization)")
        print()

    def plot_importance(self, model, feature_names):
        """Generate feature importance plot"""
        logger.info("Generating Feature Importance Plot...")
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(model, max_num_features=14, importance_type='weight', height=0.5)
        plt.title("Feature Importances (Raw Features)")
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        logger.info("Feature importance plot saved as feature_importance.png")
        plt.show()

    def save_model(self, model):
        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_SAVE_PATH)
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Load data (no polynomial features)
    X_train, y_train, X_test, y_test = trainer.load_data()
    
    # Run walk-forward validation to check generalization
    trainer.walk_forward_validation(X_train, y_train)
    
    # Train final model
    model = trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate with overfitting check
    trainer.evaluate_thresholds(model, X_train, y_train, X_test, y_test)
    
    # Plot and save
    trainer.plot_importance(model, X_train.columns.tolist())
    trainer.save_model(model)