import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Model Component")
    parser.add_argument("--train_data", type=str, required=True, help="Input directory containing train.parquet")
    parser.add_argument("--model_output", type=str, required=True, help="Output path for MLflow model artifact")
    args = parser.parse_args()

    # Enable MLflow autologging
    mlflow.autolog()

    print("📥 Reading training dataset...")
    train_df = pd.read_parquet(os.path.join(args.train_data, "train.parquet"))
    X_train = train_df.drop(columns=['Churned'])
    y_train = train_df['Churned']

    # Set up preprocessing and pipeline
    cat_cols = ['Education', 'MaritalStatus', 'Gender', 'EnrollmentType', 'LoyaltyCard']
    existing_cat = [c for c in cat_cols if c in X_train.columns]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), existing_cat)
    ], remainder='passthrough')

    pipeline = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='logloss'))
    ])

    print("🚀 Fitting XGBoost Pipeline with SMOTE...")
    pipeline.fit(X_train, y_train)

    # Save MLflow model
    print(f"💾 Saving MLflow model artifact to {args.model_output}")
    # Pass trusted types to bypass the skops security check
    mlflow.sklearn.save_model(
        sk_model=pipeline,
        path=args.model_output,
        skops_trusted_types=[
            "imblearn.over_sampling._smote.base.SMOTE",
            "imblearn.pipeline.Pipeline",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier",
        ],
    )
    print("✅ Training complete!")

if __name__ == "__main__":
    main()