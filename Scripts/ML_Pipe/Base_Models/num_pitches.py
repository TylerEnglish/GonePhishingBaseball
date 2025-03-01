import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

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

def build_model() -> TabNetRegressor:
    """
    Build and return a TabNet regression model.
    
    Returns:
        TabNetRegressor: The constructed model.
    """
    model = TabNetRegressor(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='sparsemax'
    )
    return model

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the data by performing:
      - Aggregation by PitcherId and BatterId.
      - Conversion of datetime columns to numeric.
      - Missing value imputation:
            * Numeric columns: fill with the median (or 0 if median is NaN).
            * Categorical columns: fill with "Missing".
      - Conversion of non-numeric features to strings.
      - Label encoding for categorical features.
    
    Parameters:
        df (pd.DataFrame): The raw input DataFrame.
    
    Returns:
        pd.DataFrame: The preprocessed and aggregated DataFrame.
    """
    df_agg = data_transform(df)

    # Convert datetime columns to numeric (int64)
    for col in df_agg.columns:
        if pd.api.types.is_datetime64_any_dtype(df_agg[col]):
            df_agg[col] = df_agg[col].astype("int64")
    
    # Fill missing values for each column except the target
    for col in df_agg.columns:
        if col == "PitchofPA":
            continue
        if np.issubdtype(df_agg[col].dtype, np.number):
            median_val = df_agg[col].median()
            if np.isnan(median_val):
                df_agg[col] = df_agg[col].fillna(0)
            else:
                df_agg[col] = df_agg[col].fillna(median_val)
        else:
            df_agg[col] = df_agg[col].fillna("Missing")
    
    # Drop rows where the target is missing
    df_agg = df_agg.dropna(subset=["PitchofPA"])
    
    # Convert non-numeric features to strings before label encoding
    feature_cols = [col for col in df_agg.columns if col != "PitchofPA"]
    for col in feature_cols:
        if not np.issubdtype(df_agg[col].dtype, np.number):
            df_agg[col] = df_agg[col].astype(str)
    
    # Label encode categorical columns
    cat_cols = df_agg.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_agg[col] = le.fit_transform(df_agg[col])
        encoders[col] = le
    
    return df_agg

def train_model(df: pd.DataFrame, model: TabNetRegressor):
    """
    Splits the data, scales features, and trains the TabNet model.
    
    Parameters:
        df (pd.DataFrame): Preprocessed data including the target "PitchofPA".
        model (TabNetRegressor): The untrained TabNet model.
    
    Returns:
        Tuple containing:
            - Trained TabNet model.
            - Fitted StandardScaler.
            - Scaled validation features.
            - Validation target values.
            - List of feature column names.
    """
    feature_cols = [col for col in df.columns if col != "PitchofPA"]
    X = df[feature_cols].values
    y = df["PitchofPA"].values.reshape(-1, 1)
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_valid_scaled, y_valid)],
        eval_metric=["rmse"],
        max_epochs=100,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    return model, scaler, X_valid_scaled, y_valid, feature_cols

def validate_model(model: TabNetRegressor, X_valid, y_valid) -> float:
    """
    Evaluate the trained model on the validation set using RMSE.
    
    Parameters:
        model (TabNetRegressor): The trained model.
        X_valid (np.ndarray): Scaled validation features.
        y_valid (np.ndarray): Validation target values.
    
    Returns:
        float: The RMSE score.
    """
    preds = model.predict(X_valid)
    # Convert predictions to integers
    preds = np.ceil(preds).astype(int)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

# ================================================
# Main Predict Function
# ================================================

def predict(pitcher: float, batter: float, model: TabNetRegressor, scaler: StandardScaler, df: pd.DataFrame):
    """
    Predict PitchofPA for a given PitcherId and BatterId.
    
    Parameters:
        pitcher (float): The PitcherId.
        batter (float): The BatterId.
        model (TabNetRegressor): The trained model.
        scaler (StandardScaler): The fitted scaler.
        df (pd.DataFrame): The aggregated and preprocessed DataFrame.
    
    Returns:
        int or None: The predicted PitchofPA as an integer, or None if no matching row is found.
    """
    row = df[(df["PitcherId"] == pitcher) & (df["BatterId"] == batter)]
    if row.empty:
        print(f"No aggregated data found for Pitcher {pitcher} vs Batter {batter}.")
        return None
    feature_cols = [col for col in df.columns if col != "PitchofPA"]
    X_new = row[feature_cols].values
    X_new_scaled = scaler.transform(X_new)
    
    # Model returns an array; we convert to int
    pred = model.predict(X_new_scaled)
    prediction = int(np.ceil(pred[0][0]))
    print(f"Predicted PitchofPA for Pitcher {pitcher} vs Batter {batter}: {prediction}")
    return prediction

# =====================================
# Director Function
# =====================================

def model_train():
    """
    Director function that:
      - Loads data from a parquet file.
      - Prepares the data (aggregation, conversion, missing value imputation, encoding).
      - Trains the TabNet model and validates it using RMSE.
    
    Returns:
        Tuple containing:
            - The trained TabNet model.
            - The fitted StandardScaler.
            - The aggregated and preprocessed DataFrame.
            - The list of feature column names.
    """
    data_path = "Derived_Data/feature/nDate_feature.parquet"  
    if not os.path.exists(data_path):
        print(f"Data file not found at: {data_path}")
        return None, None, None, None

    table = pq.read_table(source=data_path)
    df = table.to_pandas()
    
    df_agg = prepare_data(df)
    
    # Build and train the model
    reg_model = build_model()
    reg_model, scaler, X_valid, y_valid, feature_cols = train_model(df_agg, reg_model)
    validate_model(reg_model, X_valid, y_valid)
    
    return reg_model, scaler, df_agg, feature_cols

# =====================================
# Main Execution
# =====================================

if __name__ == "__main__":
    reg_model, scaler, df_agg, feature_cols = model_train()
    if reg_model is not None and scaler is not None and df_agg is not None:
        # Example usage
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        predict(pitcher_id, batter_id, reg_model, scaler, df_agg)
