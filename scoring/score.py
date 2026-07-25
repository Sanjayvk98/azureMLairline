import os
import pandas as pd
import mlflow.pyfunc

def init():
    global model
    # AZUREML_MODEL_DIR points to the mounted local folder containing model artifacts
    model_path = os.getenv("AZUREML_MODEL_DIR")
    model = mlflow.pyfunc.load_model(model_path)

def run(mini_batch):
    results = []
    for file_path in mini_batch:
        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_parquet(file_path)
        
        # Clean input columns
        input_data = df.copy()
        cols_to_drop = ['LoyaltyNumber', 'is_active', 'Churned', 'CLV', 'Salary', 'CLV_log', 'PostalCode', 'City', 'Province', 'Country']
        input_data = input_data.drop(columns=[c for c in cols_to_drop if c in input_data.columns])
        
        # Generate predictions
        preds = model.predict(input_data)
        
        output_df = df.copy()
        output_df["churn_prediction"] = preds
        results.append(output_df)
        
    return pd.concat(results)