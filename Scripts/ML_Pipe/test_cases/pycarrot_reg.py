import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# PyCaret Regression Imports
from pycaret.regression import RegressionExperiment, predict_model

# ==================================
# Scripting Functions
# ==================================

def data_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw pitch data by PitcherId and BatterId.
    - For 'PitchofPA': use the maximum value.
    - For numeric columns: use the mean.
    - For categorical columns: take the first occurrence.
    
    Parameters:
        df (pd.DataFrame): The raw pitch data.
    
    Returns:
        pd.DataFrame: The aggregated DataFrame.
    """
    agg_dict = {}
    for col in df.columns:
        if col in ["PitcherId", "BatterId"]:
            continue
        if col == "PitchofPA":
            agg_dict[col] = "max"
        elif np.issubdtype(df[col].dtype, np.number):
            agg_dict[col] = "mean"
        else:
            agg_dict[col] = "first"
    
    df_grouped = df.groupby(["PitcherId", "BatterId"]).agg(agg_dict).reset_index()
    return df_grouped

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the data by:
      - Aggregation by PitcherId and BatterId.
      - (Optional) Minimal cleaning or type conversions.
      - Dropping rows where 'PitchofPA' is missing (target).
    
    Parameters:
        df (pd.DataFrame): The raw input DataFrame.
    
    Returns:
        pd.DataFrame: The preprocessed and aggregated DataFrame.
    """
    df_agg = data_transform(df)

    # Convert any datetime columns to numeric (int64) if present
    for col in df_agg.columns:
        if pd.api.types.is_datetime64_any_dtype(df_agg[col]):
            df_agg[col] = df_agg[col].astype("int64")
    
    # Drop rows if target is missing
    df_agg.dropna(subset=["PitchofPA"], inplace=True)
    
    return df_agg

# ================================================
# Main Predict Function
# ================================================
def predict(pitcher: float,
            batter: float,
            model,  # PyCaret model pipeline returned from finalize_model(...)
            df: pd.DataFrame) -> int:
    """
    Predict PitchofPA for a given PitcherId and BatterId using the finalized PyCaret model.
    
    Parameters:
        pitcher (float): The PitcherId.
        batter (float): The BatterId.
        model: The finalized PyCaret regression pipeline/model.
        df (pd.DataFrame): The aggregated DataFrame (same structure used in PyCaret).
    
    Returns:
        int or None: The predicted PitchofPA as an integer, or None if no matching row is found.
    """
    row = df[(df["PitcherId"] == pitcher) & (df["BatterId"] == batter)]
    if row.empty:
        print(f"No aggregated data found for Pitcher {pitcher} vs Batter {batter}.")
        return None
    
    # Create a copy for prediction
    predict_df = row.copy()
    
    # predict_model applies the trained pipeline to produce predictions.
    predictions = predict_model(estimator=model, data=predict_df)
    
    # The default output of predict_model has a 'Label' column with predictions
    pred_val = np.ceil(predictions["Label"].values[0]).astype(int)
    print(f"Predicted PitchofPA for Pitcher {pitcher} vs Batter {batter}: {pred_val}")
    return pred_val

# =====================================
# Director Function
# =====================================
def model_train():
    """
    Director function that:
      - Loads data from a parquet file.
      - Aggregates the data by PitcherId and BatterId.
      - Uses PyCaret regression to compare all models, select the best one,
        and finalize that best model pipeline.
    
    Returns:
        - finalized_model: The finalized PyCaret regression pipeline.
        - df_agg: The aggregated and preprocessed DataFrame.
    """
    data_path = "Derived_Data/filter/filtered_20250301_000033.parquet"  
    if not os.path.exists(data_path):
        print(f"Data file not found at: {data_path}")
        return None, None

    table = pq.read_table(source=data_path)
    df = table.to_pandas()
    
    df_agg = prepare_data(df)

    # Create an instance of PyCaret's RegressionExperiment
    exp = RegressionExperiment()

    # Set up the experiment using the instance-based API
    exp.setup(
        data=df_agg,
        target="PitchofPA",
        session_id=42,       # for reproducibility
        train_size=0.8,      # 80/20 split
        silent=True,         # no interactive prompt
        log_experiment=False # disable experiment logging unless needed
    )
    
    # Compare models within the experiment (exp) and select the best based on RMSE
    best_model = exp.compare_models(sort="RMSE")
    
    # Finalize the best model (train on the entire dataset in that session)
    finalized_model = exp.finalize_model(best_model)

    return finalized_model, df_agg

# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    trained_model, df_agg = model_train()
    
    # Example usage if the model was successfully trained
    if trained_model is not None and df_agg is not None:
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        _ = predict(pitcher_id, batter_id, trained_model, df_agg)
