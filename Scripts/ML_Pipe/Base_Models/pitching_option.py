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
from sklearn.metrics import mean_squared_error

# ==================================
# Scripting Function
# ==================================

def model():
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
        mask_type='sparsemax'  # Other options include 'entmax'
    )
    return model


def train(df, model, encoders, feature_cols, target_col):
    """
    train the model
    """
    # Create a copy so as not to modify the original DataFrame
    df_train = df.copy()
    
    # Apply encoding for each categorical column we need
    for col, encoder in encoders.items():
        if col in df_train.columns:
            df_train[col] = encoder.transform(df_train[col])
    
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

def validate():
    """
    scores the model to see how well
    """
    pass

# ================================================
# Main predict
# ================================================

def predict(pitcher, batter, model, df):
    """
    predict num of pitches a pitcher gonna throw
    """
    pass

# =====================================
# Director Function
# =====================================

def model_train():
    """
    basically the director function up to training
    """
    pass

if __name__ == "__main":
    """
    run the director
    run a prediction
    """