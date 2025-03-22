import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ==================================
# Scripting Functions
# ==================================

def data_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw pitch data by PitcherId and BatterId.
    - For 'PitchofPA': use the maximum value.
    - For numeric columns: use the mean.
    - For categorical columns: take the first occurrence.
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

def build_model() -> xgb.XGBRegressor:
    """
    Build and return an XGBoost regression model.
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse"  # Set eval_metric here
    )
    return model

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data:
      - Aggregation by PitcherId and BatterId.
      - Convert datetime columns to numeric.
      - Fill missing values.
      - Convert non-numeric features to strings and label encode them.
    """
    df_agg = data_transform(df)
    
    # Convert datetime columns to numeric (int64)
    for col in df_agg.columns:
        if pd.api.types.is_datetime64_any_dtype(df_agg[col]):
            df_agg[col] = df_agg[col].astype("int64")
    
    # Fill missing values (except target)
    for col in df_agg.columns:
        if col == "PitchofPA":
            continue
        if np.issubdtype(df_agg[col].dtype, np.number):
            median_val = df_agg[col].median()
            df_agg[col] = df_agg[col].fillna(median_val if not np.isnan(median_val) else 0)
        else:
            df_agg[col] = df_agg[col].fillna("Missing")
    
    # Drop rows with missing target
    df_agg = df_agg.dropna(subset=["PitchofPA"])
    
    # Convert non-numeric features to strings and label encode
    feature_cols = [col for col in df_agg.columns if col != "PitchofPA"]
    for col in feature_cols:
        if not np.issubdtype(df_agg[col].dtype, np.number):
            df_agg[col] = df_agg[col].astype(str)
    
    cat_cols = df_agg.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_agg[col] = le.fit_transform(df_agg[col])
        encoders[col] = le
    
    return df_agg

def train_model(df: pd.DataFrame, model: xgb.XGBRegressor):
    """
    Splits the data, scales features, and trains the XGBoost model.
    """
    feature_cols = [col for col in df.columns if col != "PitchofPA"]
    X = df[feature_cols].values
    y = df["PitchofPA"].values.reshape(-1, 1)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    model.fit(X_train_scaled, y_train, eval_set=[(X_valid_scaled, y_valid)], verbose=True)
    
    return model, scaler, X_valid_scaled, y_valid, feature_cols

def validate_model(model: xgb.XGBRegressor, X_valid, y_valid) -> float:
    """
    Evaluate the model on the validation set using RMSE.
    """
    preds = model.predict(X_valid)
    preds = np.ceil(preds).astype(int)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

# -----------------------------------------------
# Helper Functions for Saving/Loading the Model
# -----------------------------------------------

def save_regression_model(model: xgb.XGBRegressor, file_path="Derived_Data/model_params/model_params.json"):
    """
    Save the trained XGBoost regression model as a JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    model.save_model(file_path)
    print(f"Saved regression model to {file_path}")

def load_regression_model(file_path="Derived_Data/model_params/model_params.json") -> xgb.XGBRegressor:
    """
    Load an XGBoost regression model from a JSON file.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Model file not found at {file_path}; run training first.")
    model = build_model()
    model.load_model(file_path)
    print(f"Loaded regression model from {file_path}")
    return model

# ================================================
# Main Predict Function (for a single prediction)
# ================================================

def predict(pitcher: float, batter: float, model: xgb.XGBRegressor, scaler: StandardScaler, df: pd.DataFrame):
    """
    Predict PitchofPA for a given PitcherId and BatterId.
    """
    row = df[(df["PitcherId"] == pitcher) & (df["BatterId"] == batter)]
    if row.empty:
        print(f"No aggregated data found for Pitcher {pitcher} vs Batter {batter}.")
        return None
    feature_cols = [col for col in df.columns if col != "PitchofPA"]
    X_new = row[feature_cols].values
    X_new_scaled = scaler.transform(X_new)
    pred = model.predict(X_new_scaled)
    prediction = int(np.ceil(pred[0]))
    print(f"Predicted PitchofPA for Pitcher {pitcher} vs Batter {batter}: {prediction}")
    return prediction

# -----------------------------------------------
# Director Function: Training Pipeline
# -----------------------------------------------

def model_train():
    """
    Loads data, prepares it, trains the model, and validates it.
    Returns the trained model, scaler, data, and feature names.
    """
    data_path = "Derived_Data/feature/nDate_feature.parquet"  
    if not os.path.exists(data_path):
        print(f"Data file not found at: {data_path}")
        return None, None, None, None

    table = pq.read_table(source=data_path)
    df = table.to_pandas()
    
    df_agg = prepare_data(df)
    
    reg_model = build_model()
    reg_model, scaler, X_valid, y_valid, feature_cols = train_model(df_agg, reg_model)
    validate_model(reg_model, X_valid, y_valid)
    
    return reg_model, scaler, df_agg, feature_cols

# -----------------------------------------------
# Main Execution
# -----------------------------------------------

if __name__ == "__main__":
    # Train the model
    reg_model, scaler, df_agg, feature_cols = model_train()
    if reg_model is not None and scaler is not None and df_agg is not None:
        # Save the trained regression model as JSON
        save_regression_model(reg_model, file_path="Derived_Data/model_params/model_params.json")
        
        # Load the regression model from JSON
        loaded_model = load_regression_model(file_path="Derived_Data/model_params/model_params.json")
        
        # Example prediction using the loaded model
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        predict(pitcher_id, batter_id, loaded_model, scaler, df_agg)
