import pyarrow.parquet as pq
import pandas as pd
import pandas as pd
import os
from datetime import datetime


# =================================
# Scripting Methods
# =================================

def save_models(models):
    """
    save model to derieved_data/model_params
    if not made make the folder to input in
    """
    pass

def save_prediction(pred_results):
    """
    save prediction to derieved_data/models_pred/
    if not made make the folder to input in
    """
    pass

# ===========================
# Director function
# ===========================

def train():
    """
    train both 
    """
    pass

def pred():
    """
    run the regression model for expected num of pitches to out
    run the classification model base on cummulative of 10 (or SD) for pitches
    label the report based on exepected to out then to the rest to fill out the 10 or SD
    """
    pass