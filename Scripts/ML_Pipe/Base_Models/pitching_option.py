import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==================================
# Scripting Functions
# ==================================

def build_model():
    """
    Build a TabNet classification model.
    """
    model = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type='sparsemax'  # other option: 'entmax'
    )
    return model

def train(df, model, encoders, feature_cols, target_col):
    """
    Train the TabNet classifier.
    - Encodes categorical features.
    - Splits, scales, and fits the model.
    """
    df_train = df.copy(deep=True)
    
    # Encode categorical features using provided encoders.
    for col, encoder in encoders.items():
        if col in df_train.columns:
            df_train[col] = encoder.transform(df_train[col])
    
    # Separate features and target.
    X = df_train[feature_cols].values
    y = df_train[target_col].values
    
    # Split into training and validation sets.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    
    # Train the model.
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['accuracy'],
        max_epochs=100,
        patience=10,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    return model, scaler, X_valid, y_valid

def validate(model, X_valid, y_valid):
    """
    Evaluate the trained model on the validation set and print accuracy.
    """
    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc

# ================================================
# Main Predict Function
# ================================================
def predict(pitcher, batter, model, scaler, encoders, df, feature_cols, target_col):
    """
    For a given pitcher and batter:
      - Determines candidate pitch types based solely on what the pitcher has thrown.
      - Gathers multiple rows: uses matchup data (if available) or all pitcher data otherwise.
      - For each candidate pitch, overrides 'CleanPitchType' in all rows, obtains predictions,
        and averages the probabilities across rows.
      - Recommends the pitch that maximizes the average probability for the "StrikeCalled" outcome.
      
    Returns:
      best_pitch (str): Recommended pitch type.
      results_df (DataFrame): Table of candidate pitches and their averaged predicted probabilities.
    """
    # Get all data for this pitcher.
    pitcher_data = df[df["PitcherId"] == pitcher]
    if pitcher_data.empty:
        print(f"No data found for Pitcher {pitcher}.")
        return None, None

    # Candidate pitch types: only those the pitcher has thrown.
    candidate_pitches = pitcher_data["CleanPitchType"].unique()
    if len(candidate_pitches) == 0:
        print(f"No pitch types available for Pitcher {pitcher}.")
        return None, None

    # Try to get matchup data; if none, use all pitcher data.
    matchup_data = df[(df['PitcherId'] == pitcher) & (df['BatterId'] == batter)]
    if not matchup_data.empty:
        base_data = matchup_data.copy()
    else:
        base_data = pitcher_data.copy()
        print(f"No matchup data found for Pitcher {pitcher} vs Batter {batter}; using general pitcher data.")
    
    # Get target class names.
    target_encoder = encoders[target_col]
    target_classes = target_encoder.classes_
    
    results = []
    for pitch in candidate_pitches:
        # Simulate candidate pitch by replacing CleanPitchType in all rows of base_data.
        simulated_data = base_data.copy()
        simulated_data["CleanPitchType"] = pitch
        
        # Encode categorical features for all rows.
        for col, encoder in encoders.items():
            if col in simulated_data.columns:
                simulated_data[col] = encoder.transform(simulated_data[col])
        
        # Select features and scale.
        X_sim = simulated_data[feature_cols].values
        X_sim_scaled = scaler.transform(X_sim)
        
        # Get predictions for all rows and average the probabilities.
        proba = model.predict_proba(X_sim_scaled)  # shape (n_rows, n_classes)
        avg_proba = np.mean(proba, axis=0)
        avg_proba_percent = (avg_proba * 100).round(2)
        
        # Build a result dictionary.
        result = {'PitchType': pitch}
        for cls, prob in zip(target_classes, avg_proba_percent):
            result[cls] = prob
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Determine best pitch based on maximum average probability for "StrikeCalled".
    if "StrikeCalled" in results_df.columns:
        best_idx = results_df["StrikeCalled"].idxmax()
        best_pitch = results_df.loc[best_idx, "PitchType"]
        print(f"Recommended pitch: {best_pitch} with {results_df.loc[best_idx, 'StrikeCalled']}% chance for 'StrikeCalled'")
    else:
        best_pitch = None
    
    print(f"\nPrediction results for Pitcher {pitcher} vs Batter {batter}:")
    print(results_df.to_string(index=False))
    
    return best_pitch, results_df

# =====================================
# Director Function
# =====================================
def model_train(data_path="data.parquet"):
    """
    Director function:
      - Loads data from a parquet file.
      - Uses all columns except the target column as features.
      - Automatically detects and encodes categorical columns.
      - Fills missing values robustly and drops rows with missing target.
      - Trains the TabNet classifier, validates it, and returns the trained model along with related objects.
    """
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None, None, None, None, None, None
    
    # Load data.
    table = pq.read_table(source=data_path)
    df = table.to_pandas()
    
    # Define target column.
    target_col = 'PitchCall'
    
    # Fill missing values for every feature column.
    for col in df.columns:
        if col == target_col:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            median_val = df[col].median()
            if pd.isna(median_val):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna("Missing")
    
    # Drop rows where the target is missing.
    df = df.dropna(subset=[target_col])
    
    # Use all columns except the target as features.
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Automatically detect categorical columns (assume object-type).
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if df[target_col].dtype == 'object' and target_col not in categorical_cols:
        categorical_cols.append(target_col)
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        encoders[col] = le
    
    model = build_model()
    model, scaler, X_valid, y_valid = train(df, model, encoders, feature_cols, target_col)
    
    validate(model, X_valid, y_valid)
    
    return model, scaler, encoders, feature_cols, target_col, df

# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    data_path = "Derived_Data/filter/filtered_20250301_000033.parquet"
    model, scaler, encoders, feature_cols, target_col, df = model_train(data_path=data_path)
    
    if model is not None:
        # Replace these with actual PitcherId and BatterId values from your data.
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        best_pitch, results_df = predict(pitcher_id, batter_id, model, scaler, encoders, df, feature_cols, target_col)
        print("\nFinal Recommended Pitch:", best_pitch)
        print("\nDetailed Prediction Table:")
        print(results_df)
