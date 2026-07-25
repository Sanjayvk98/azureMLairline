import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model & Quality Gate Component")
    parser.add_argument("--model_input", type=str, required=True, help="Path to input MLflow model artifact")
    parser.add_argument("--test_data", type=str, required=True, help="Input directory containing test.parquet")
    parser.add_argument("--auc_threshold", type=float, default=0.90, help="Minimum ROC-AUC required for quality gate")
    args = parser.parse_args()

    print("📥 Reading test dataset...")
    test_df = pd.read_parquet(os.path.join(args.test_data, "test.parquet"))
    X_test = test_df.drop(columns=['Churned'])
    y_test = test_df['Churned']

    print("LOAD Loading MLflow model artifact...")
    model = mlflow.sklearn.load_model(args.model_input)

    # Generate predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n📊 --- Evaluation Results ---")
    print(f"   - Test ROC-AUC:   {roc_auc:.4f}")
    print(f"   - Test Precision: {precision:.4f}")
    print(f"   - Test Recall:    {recall:.4f}")
    print(f"   - Test F1-Score:  {f1:.4f}")
    print(f"   - Test Accuracy:  {accuracy:.4f}\n")

    # Enforce Quality Gate
    if roc_auc < args.auc_threshold:
        raise ValueError(
            f"❌ Quality Gate Failed! Test ROC-AUC ({roc_auc:.4f}) is below required threshold ({args.auc_threshold})."
        )

    print(f"🚀 Quality Gate Passed! ROC-AUC ({roc_auc:.4f}) >= Threshold ({args.auc_threshold}). Model promoted.")

if __name__ == "__main__":
    main()