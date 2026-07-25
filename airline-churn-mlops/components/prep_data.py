import argparse
import os
import mltable
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, help="Path to raw dataset")
    parser.add_argument(
        "--train_data", type=str, help="Path to save train dataset"
    )
    parser.add_argument(
        "--test_data", type=str, help="Path to save test dataset"
    )
    args = parser.parse_args()

    # 1. Read input dataset
    tbl = mltable.load(args.raw_data)
    df = tbl.to_pandas_dataframe()

    # 2. Perform preprocessing & split (creating train_df & test_df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Save train & test output datasets (4 spaces indent!)
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # Make sure to use .to_parquet() to match train_model.py!
    train_df.to_parquet(
        os.path.join(args.train_data, "train.parquet"), index=False
    )
    test_df.to_parquet(
        os.path.join(args.test_data, "test.parquet"), index=False
    )


if __name__ == "__main__":
    main()