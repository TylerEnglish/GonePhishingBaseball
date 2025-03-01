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
    tabnet regression model
    """
    pass

def train(df, model):
    """
    train the model
    """
    pass

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