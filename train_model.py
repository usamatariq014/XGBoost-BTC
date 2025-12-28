import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.preprocessing import PolynomialFeatures
import logging
import sys
from pathlib import Path
import joblib

# --- Configuration ---
TRAIN_PATH = Path("training/train_final.csv")
TEST_PATH = Path("testing/test_final.csv")
MODEL_SAVE_PATH = Path("models/xgb_model.pkl")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        # Drop non-math columns
        self.drop_cols = [
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
            'Quote asset volume', 'Number of trades', 
            'Taker buy base asset volume', 'Taker buy quote asset volume', 
            'Ignore', 'target', 'Close time'
        ]

    def load_and_engineer_interactions(self):
        logger.info("Loading Datasets...")
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        # 1. Clean X and y
        X_train_raw = train_df.drop(columns=[c for c in self.drop_cols if c in train_df.columns])
        y_train = train_df['target']
        
        X_test_raw = test_df.drop(columns=[c for c in self.drop_cols if c in test_df.columns])
        y_test = test_df['target']
        
        # 2. GENERATE INTERACTIONS (The "Hybrid" Step)
        # degree=2 creates pairs. interaction_only=True avoids squaring (A*A).
        logger.info(f"Generating Interaction Features (Degree 2)...")
        logger.info(f"Original Count: {X_train_raw.shape[1]}")
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        
        # Fit/Transform
        X_train_poly = poly.fit_transform(X_train_raw)
        X_test_poly = poly.transform(X_test_raw)
        
        # Recover Names (so we can see what works in the plot)
        feature_names = poly.get_feature_names_out(X_train_raw.columns)
        
        X_train = pd.DataFrame(X_train_poly, columns=feature_names)
        X_test = pd.DataFrame(X_test_poly, columns=feature_names)
        
        logger.info(f"New Feature Count: {X_train.shape[1]}")
        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, X_test, y_test):
        logger.info("--- Starting XGBoost Training (Deep Trees) ---")
        
        # Balance Weight
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_weight = num_neg / num_pos
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=1500,          # Increased for more features
            learning_rate=0.05,         # Slow learning
            max_depth=10,               # <--- DEEP TREES (Logic Chains)
            subsample=0.8,
            colsample_bytree=0.5,       # <--- Force exploration of new features
            scale_pos_weight=scale_weight,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        logger.info("Training Complete.")
        return model

    def evaluate_thresholds(self, model, X_test, y_test):
        logger.info("--- Running Probability Threshold Test ---")
        probs = model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*50)
        print(f"{'THRESHOLD':<10} | {'TRADES':<10} | {'WIN RATE':<10} | {'PROFITABLE?'}")
        print("="*50)
        
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            preds = (probs > thresh).astype(int)
            num_trades = preds.sum()
            if num_trades < 5: continue
            
            precision = precision_score(y_test, preds, zero_division=0)
            is_profitable = "YES" if precision > 0.34 else "NO"
            print(f"{thresh:<10} | {num_trades:<10} | {precision:.2%}     | {is_profitable}")
        print("="*50 + "\n")

    def plot_importance(self, model):
        logger.info("Generating Feature Importance Plot...")
        plt.figure(figsize=(12, 12))
        # Top 25 features to see if interactions made it
        xgb.plot_importance(model, max_num_features=25, importance_type='weight', height=0.5)
        plt.title("Top 25 Features (Looking for Interactions)")
        plt.tight_layout()
        plt.show()

    def save_model(self, model):
        joblib.dump(model, MODEL_SAVE_PATH)
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    X_train, y_train, X_test, y_test = trainer.load_and_engineer_interactions()
    model = trainer.train(X_train, y_train, X_test, y_test)
    trainer.evaluate_thresholds(model, X_test, y_test)
    trainer.plot_importance(model)
    trainer.save_model(model)