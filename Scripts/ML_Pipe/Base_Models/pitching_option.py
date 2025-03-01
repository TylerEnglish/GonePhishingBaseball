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
# Scripting Function
# ==================================

def build_model():
    """
    tabnet class model
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
    train the model
    """
    df_train = df.copy(deep=True)
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df_train.columns:
            df_train[col] = encoder.transform(df_train[col])
    
    # Drop rows with missing values in the selected feature and target columns
    df_train = df_train.dropna(subset=feature_cols + [target_col])
    
    # Separate features and target
    X = df_train[feature_cols].values
    y = df_train[target_col].values
    
    # Split into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    
    # Train the model
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
    scores the model to see how well
    """
    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc

# ================================================
# Main predict
# ================================================

def predict(pitcher, batter, model, scaler, encoders, df, feature_cols, target_col):
    """
    predict num of pitches a pitcher gonna throw
    """
    filtered = df[(df['PitcherId'] == pitcher) & (df['BatterId'] == batter)]
    if filtered.empty:
        print(f"No data found for Pitcher {pitcher} and Batter {batter}.")
        return None, None
    base_row = filtered.iloc[0].copy()
    
    # Get candidate pitch types from the 'CleanPitchType' encoder
    pitch_encoder = encoders['CleanPitchType']
    candidate_pitches = pitch_encoder.classes_
    
    # Get target class names from the target encoder
    target_encoder = encoders[target_col]
    target_classes = target_encoder.classes_
    
    results = []
    for pitch in candidate_pitches:
        row = base_row.copy()
        row['CleanPitchType'] = pitch  
        
        row_df = pd.DataFrame([row])
        for col, encoder in encoders.items():
            if col in row_df.columns:
                row_df[col] = encoder.transform(row_df[col])
                
        # Prepare feature vector and scale it
        X_new = row_df[feature_cols].values
        X_new_scaled = scaler.transform(X_new)
        
        # Get prediction probabilities
        proba = model.predict_proba(X_new_scaled)
        # Convert probabilities to percentages and round
        proba_percent = (proba[0] * 100).round(2)
        
        # Build a result dictionary
        result = {'PitchType': pitch}
        for cls, prob in zip(target_classes, proba_percent):
            result[cls] = prob
        results.append(result)
    
    # Convert results list 
    results_df = pd.DataFrame(results)
    
    # Determine best pitch 
    if "StrikeCalled" in target_classes:
        best_idx = results_df["StrikeCalled"].idxmax()
        best_pitch = results_df.loc[best_idx, "PitchType"]
        print(f"Recommended pitch: {best_pitch} with {results_df.loc[best_idx, 'StrikeCalled']}% chance for 'Ball'")
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
    basically the director function up to training
    """
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None, None, None, None, None, None
    
    # Load data from a parquet file
    table = pq.read_table(source=data_path)
    df = table.to_pandas()
    
    # Define feature and target columns; adjust these as needed.
    feature_cols = ['PitcherId', 'BatterId', 'PitcherThrows', 'BatterSide', 'CleanPitchType']
    target_col = 'PitchCall'
    
    # Initialize LabelEncoders for categorical columns and target
    encoders = {}
    for col in ['PitcherThrows', 'BatterSide', 'CleanPitchType', target_col]:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # ensure string type for encoding
        le.fit(df[col])
        encoders[col] = le
    
    # Build and train the TabNet classifier
    model = build_model()
    model, scaler, X_valid, y_valid = train(df, model, encoders, feature_cols, target_col)
    
    # Validate the model
    validate(model, X_valid, y_valid)
    
    return model, scaler, encoders, feature_cols, target_col, df

if __name__ == "__main__":
    """
    run the director
    run a prediction
    """
    data_path = "Derived_Data/filter/filtered_20250228_221118.parquet"
    model, scaler, encoders, feature_cols, target_col, df = model_train(data_path=data_path)
    # table = pq.read_table(source=data_path)
    # df = table.to_pandas()
    # print(df.columns)
    # print(df['PitcherId'][0],"\n", df['BatterId'][0])
    if model is not None:
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        pitch, result = predict(pitcher_id, batter_id, model, scaler, encoders, df, feature_cols, target_col)
        print(pitch)
        print(result)


