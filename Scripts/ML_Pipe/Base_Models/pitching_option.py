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
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from pytorch_tabnet.metrics import Metric
import numpy as np
from sklearn.metrics import f1_score
import pickle
import zipfile

#====================================
# Helper: Save Object as Zip-Pickle
#====================================
def save_pickle_zip(obj, zip_path, internal_filename):
    """
    Pickle the object and save it as a compressed zip file.
    """
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        data = pickle.dumps(obj)
        zf.writestr(internal_filename, data)

#==========================
# Metrics
#==========================
class MacroF1(Metric):
    '''
    Macro F1 computes the F1-score for each class individually—calculating 
    the harmonic mean of precision and recall—and then averages these scores 
    equally across all classes. This ensures that the performance of each 
    class is weighted equally regardless of its frequency, making it especially 
    useful for imbalanced datasets. It effectively penalizes the model if even 
    one class has poor precision or recall. Overall, Macro F1 offers a balanced 
    view that emphasizes both false positives and false negatives.
    '''
    def __init__(self):
        self._name = "macro_f1"
        self._maximize = True 
    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1) 
        return f1_score(y_true, y_pred, average="macro")

class MCC(Metric):
    '''
    MCC is a robust metric that evaluates the quality of predictions by taking into 
    account true positives, true negatives, false positives, and false negatives. 
    It produces a value between –1 and 1, where 1 indicates perfect prediction, 
    0 indicates no better than random guessing, and –1 signifies total disagreement. 
    This metric is particularly valuable in imbalanced settings because it reflects 
    the overall structure of the confusion matrix. MCC thus provides a holistic 
    assessment of model performance that is not skewed by class imbalance.
    '''
    def __init__(self):
        self._name = "mcc"
        self._maximize = True  # maximize correlation
    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return matthews_corrcoef(y_true, y_pred)
class CohenKappa(Metric):
    '''
    Cohen’s Kappa measures the agreement between predicted labels and true labels while 
    accounting for the possibility of chance agreement. It compares the observed accuracy 
    to the expected accuracy, yielding a score between –1 and 1, where higher values indicate 
    better-than-chance agreement. This metric is beneficial for multi-class problems as it 
    penalizes models that merely capture the prevalence of the majority class. By adjusting 
    for chance, Cohen’s Kappa provides a more nuanced evaluation than raw accuracy.
    '''
    def __init__(self):
        self._name = "cohen_kappa"
        self._maximize = True  # maximize agreement
    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return cohen_kappa_score(y_true, y_pred)
class PrecisionScoreMetric(Metric):
    def __init__(self):
        self._name = "PrecisionScore"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        return precision_score(y_true, y_pred_labels, average="macro", zero_division=0)

class RecallScoreMetric(Metric):
    def __init__(self):
        self._name = "RecallScore"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        return recall_score(y_true, y_pred_labels, average="macro", zero_division=0)

class F1ScoreMetric(Metric):
    def __init__(self):
        self._name = "F1Score"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        # y_pred is assumed to be an array of probabilities. Get predicted labels:
        y_pred_labels = np.argmax(y_pred, axis=1)
        return f1_score(y_true, y_pred_labels)
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
        eval_metric=[MacroF1, RecallScoreMetric, PrecisionScoreMetric, MCC, CohenKappa],
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
    Evaluate the trained model on the validation set and print Accuracy, F1-score, and Precision.
    
    Returns a dictionary with keys: "accuracy", "f1", and "precision".
    """
    # Generate predictions
    preds = model.predict(X_valid)
    
    # Calculate accuracy
    acc = accuracy_score(y_valid, preds)
    
    # Calculate F1-score (macro average by default)
    f1 = f1_score(y_valid, preds, average="macro")
    
    # Calculate precision (macro average)
    precision = precision_score(y_valid, preds, average="macro")
    
    # Print the results
    print(f"Validation Accuracy: {acc:.4f}, F1-score (macro): {f1:.4f}, Precision (macro): {precision:.4f}")
    
    # Return as a dictionary
    return {"accuracy": acc, "f1": f1, "precision": precision}

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
    if "Strike" in results_df.columns:
        best_idx = results_df["Strike"].idxmax()
        best_pitch = results_df.loc[best_idx, "PitchType"]
        print(f"Recommended pitch: {best_pitch} with {results_df.loc[best_idx, 'Strike']}% chance for 'Strike'")
    else:
        best_pitch = None
    
    # print(f"\nPrediction results for Pitcher {pitcher} vs Batter {batter}:")
    # print(results_df.to_string(index=False))
    
    return best_pitch, results_df

def predict_single_pitch(
    pitcher_id: float,
    batter_id: float,
    pitch_type: str,
    model,              # your TabNetClassifier
    scaler,             # your TabNet scaler
    encoders,           # your TabNet encoders
    df_tabnet: pd.DataFrame,
    feature_cols: list,
    target_col: str
):
    """
    Computes the average predicted probabilities for ONE forced pitch_type
    across all rows matching (pitcher_id, batter_id) in df_tabnet.
    
    Returns:
      Dictionary of average probabilities {class_name: avg_prob},
      or None if no data found.
    """
    # Filter data for this pitcher
    pitcher_df = df_tabnet[df_tabnet["PitcherId"] == pitcher_id].copy()
    if pitcher_df.empty:
        print(f"[TabNet] No data found for Pitcher {pitcher_id}.")
        return None

    # If there's matchup data for (pitcher, batter), use that. Otherwise, pitcher-level data
    matchup_df = pitcher_df[pitcher_df["BatterId"] == batter_id].copy()
    if matchup_df.empty:
        # fallback
        print(f"[TabNet] No matchup data for Pitcher {pitcher_id}, Batter {batter_id}; using all pitcher data.")
        matchup_df = pitcher_df

    # Force the pitch
    if "CleanPitchType" in matchup_df.columns:
        matchup_df["CleanPitchType"] = pitch_type
    else:
        print("[TabNet] 'CleanPitchType' column missing in df_tabnet. Aborting.")
        return None

    # Encode any categorical columns
    for col, encoder in encoders.items():
        if col in matchup_df.columns:
            matchup_df[col] = matchup_df[col].astype(str)
            matchup_df[col] = encoder.transform(matchup_df[col])

    # Scale features
    X_sim = matchup_df[feature_cols].values
    X_sim_scaled = scaler.transform(X_sim)

    # Predict probabilities for each row, then average
    proba = model.predict_proba(X_sim_scaled)
    avg_proba = proba.mean(axis=0)  # shape = (n_classes,)

    # Get class names from encoders[target_col]
    class_labels = encoders[target_col].classes_  # e.g. ["Ball", "Strike", ...]
    prob_dict = {cls: p for cls, p in zip(class_labels, avg_proba)}

    return prob_dict
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
    target_col = 'CleanPitchCall'
    
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
    
    scores = validate(model, X_valid, y_valid)
    
    # Define the paths to save each object.
    TABNET_MODEL_ZIP    = "Derived_Data/model_params/"   # TabNet model
    TABNET_SCALER_ZIP   = "Derived_Data/extra/"
    TABNET_ENCODERS_ZIP = "Derived_Data/extra/"
    TABNET_FEATS_ZIP    = "Derived_Data/extra/"
    TABNET_TARGET_ZIP   = "Derived_Data/extra/"
    TABNET_DATA_ZIP     = "Derived_Data/extra/"

    # Save each object as a zip-compressed pickle file.
    save_pickle_zip(model, TABNET_MODEL_ZIP, "cls_model.pkl.zip")
    save_pickle_zip(scaler, TABNET_SCALER_ZIP, "cls_scaler.pkl.zip")
    save_pickle_zip(encoders, TABNET_ENCODERS_ZIP, "encoders.pkl.zip")
    save_pickle_zip(feature_cols, TABNET_FEATS_ZIP, "feature_cols_cls.pkl.zip")
    save_pickle_zip(target_col, TABNET_TARGET_ZIP, "target_col.pkl.zip")
    save_pickle_zip(df, TABNET_DATA_ZIP, "df_cls.pkl.zip")
    
    return model, scaler, encoders, feature_cols, target_col, df, scores

# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    data_path = "Derived_Data/feature/nDate_feature.parquet"  
    model, scaler, encoders, feature_cols, target_col, df, scores = model_train(data_path=data_path)
    
    if model is not None:
        # Replace these with actual PitcherId and BatterId values from your data.
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0
        best_pitch, results_df = predict(pitcher_id, batter_id, model, scaler, encoders, df, feature_cols, target_col)
        print("\nFinal Recommended Pitch:", best_pitch)
        print("\nDetailed Prediction Table:")
        print(results_df)
