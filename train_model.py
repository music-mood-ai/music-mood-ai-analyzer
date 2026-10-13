"""Train Moodify with a real labeled feature CSV."""
import argparse
from mood_engine import train_from_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Moodify mood classifier from labeled audio features.")
    parser.add_argument("csv_path", help="CSV containing the feature columns and a mood column")
    args = parser.parse_args()
    print(train_from_csv(args.csv_path))
