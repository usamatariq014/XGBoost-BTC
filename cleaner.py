import pandas as pd

# --- Configuration ---
INPUT_FILENAME = "training/btc_1h_data_2018_to_2025.csv"
TRAIN_OUTPUT = "training/train.csv"
TEST_OUTPUT = "training/test.csv"

# The Cutoff Date: Everything BEFORE this is Train, everything AFTER is Test.
# Recommendation: Use Jan 1st, 2024. This gives you 6 years of training, 1.5 years of testing.
SPLIT_DATE = "2024-01-01" 

DATE_COLUMN = "Open time"
# Updated format to match typical crypto data (adjust if yours differs)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" 

def split_data():
    try:
        print(f"Reading {INPUT_FILENAME}...")
        df = pd.read_csv(INPUT_FILENAME)

        # 1. Clean Dates
        df[DATE_COLUMN] = df[DATE_COLUMN].astype(str).str.strip()
        # Using mixed=True allows pandas to guess format if it varies
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format='mixed')

        # 2. Sort Data (CRITICAL: You cannot train on random shuffled rows)
        df = df.sort_values(by=DATE_COLUMN).reset_index(drop=True)

        # 3. The Split
        split_cutoff = pd.Timestamp(SPLIT_DATE)
        
        train_df = df[df[DATE_COLUMN] < split_cutoff]
        test_df = df[df[DATE_COLUMN] >= split_cutoff]

        # 4. Sanity Check
        if train_df.empty or test_df.empty:
            print("ERROR: One of the datasets is empty. Check your SPLIT_DATE.")
            return

        print(f"--- Split Complete ---")
        print(f"Training Data: {train_df.shape[0]} rows (Ends: {train_df[DATE_COLUMN].max()})")
        print(f"Testing Data:  {test_df.shape[0]} rows (Starts: {test_df[DATE_COLUMN].min()})")

        # 5. Save
        train_df.to_csv(TRAIN_OUTPUT, index=False)
        test_df.to_csv(TEST_OUTPUT, index=False)
        print("Files saved successfully.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    split_data()