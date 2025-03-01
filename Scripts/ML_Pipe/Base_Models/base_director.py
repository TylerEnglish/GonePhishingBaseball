import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
import pickle
import glob

# Import modules depending on context
if __name__ == "__main__":
    from num_pitches import model_train as train_reg, predict as predict_reg
    from pitching_option import model_train as train_cls, predict as predict_cls
else:
    from ML_Pipe.Base_Models.num_pitches import model_train as train_reg, predict as predict_reg
    from ML_Pipe.Base_Models.pitching_option import model_train as train_cls, predict as predict_cls


# =================================
# Scripting Methods
# =================================

def save_models(models, extras):
    """
    Save model weights to Derived_Data/model_params.
    Also save 'extras' (scalers, data, encoders, etc.) to Derived_Data/extra.
    """
    # 1) Save the trained models
    model_dir = "Derived_Data/model_params"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if models.get("reg_model") is not None:
        reg_path = os.path.join(model_dir, f"reg_model.pkl")
        models["reg_model"].save_model(reg_path)
        print(f"Saved regression model to {reg_path}")
    
    if models.get("cls_model") is not None:
        cls_path = os.path.join(model_dir, f"cls_model.pkl")
        models["cls_model"].save_model(cls_path)
        print(f"Saved classification model to {cls_path}")
    
    # 2) Save extras (pickled) in Derived_Data/extra
    extras_dir = "Derived_Data/extra"
    if not os.path.exists(extras_dir):
        os.makedirs(extras_dir)

    # Save each relevant extra object with a standard naming scheme
    # so your `predict` function knows where to find them.
    with open(os.path.join(extras_dir, "reg_scaler.pkl"), "wb") as f:
        pickle.dump(extras["reg_scaler"], f)

    with open(os.path.join(extras_dir, "df_reg.pkl"), "wb") as f:
        pickle.dump(extras["df_reg"], f)

    with open(os.path.join(extras_dir, "feature_cols_reg.pkl"), "wb") as f:
        pickle.dump(extras["feature_cols_reg"], f)

    with open(os.path.join(extras_dir, "cls_scaler.pkl"), "wb") as f:
        pickle.dump(extras["cls_scaler"], f)

    with open(os.path.join(extras_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(extras["encoders"], f)

    with open(os.path.join(extras_dir, "feature_cols_cls.pkl"), "wb") as f:
        pickle.dump(extras["feature_cols_cls"], f)

    with open(os.path.join(extras_dir, "target_col.pkl"), "wb") as f:
        pickle.dump(extras["target_col"], f)

    with open(os.path.join(extras_dir, "df_cls.pkl"), "wb") as f:
        pickle.dump(extras["df_cls"], f)

    print("Saved extras (scalers, encoders, data) to Derived_Data/extra")


def save_prediction(pred_df: pd.DataFrame, name="prediction"):
    """
    Save a single prediction DataFrame to Derived_Data/model_pred/.
    """
    save_dir = "Derived_Data/model_pred"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(save_dir, f"{name}_report_{timestamp}.csv")
    
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved {name} DataFrame to {pred_path}")


# ===========================
# Director function
# ===========================

def train():
    """
    Train both regression and classification models, return them plus extras.
    """
    print("Training regression model...")
    reg_model, reg_scaler, df_reg, feature_cols_reg = train_reg()
    
    print("Training classification model...")
    data_path = "Derived_Data/feature/feature_20250301_055043.parquet"
    cls_model, cls_scaler, encoders, feature_cols_cls, target_col, df_cls = train_cls(data_path)
    
    models = {
        "reg_model": reg_model,
        "cls_model": cls_model
    }
    extras = {
        "reg_scaler": reg_scaler,
        "df_reg": df_reg,
        "feature_cols_reg": feature_cols_reg,
        "cls_scaler": cls_scaler,
        "encoders": encoders,
        "feature_cols_cls": feature_cols_cls,
        "target_col": target_col,
        "df_cls": df_cls
    }
    return models, extras

def predict(pitcher_ids=None, batter_ids=None, n_pitches=10):
    """
    Load saved regression and classification models (and extras) from disk,
    then run predictions for each (pitcher, batter) pair.

    Produces two DataFrames:
      df_reg - one row per pitcher-batter (with 'RegressionPrediction')
      df_cls - up to n_pitches repeated classification outputs (all pitch types, probabilities, etc.)

    Saves two CSV files in Derived_Data/model_pred:
      - reg_prediction_report_*.csv
      - cls_prediction_report_*.csv

    Returns:
      (df_reg, df_cls)
    """
    
    if pitcher_ids is None:
        pitcher_ids = [1000066910.0]
    if batter_ids is None:
        batter_ids = [1000032366.0]

    if not isinstance(pitcher_ids, list):
        pitcher_ids = [pitcher_ids]
    if not isinstance(batter_ids, list):
        batter_ids = [batter_ids]

    # Load the .zip TabNet models
    model_dir = "Derived_Data/model_params"
    reg_model_path = os.path.join(model_dir, "reg_model.pkl.zip")
    cls_model_path = os.path.join(model_dir, "cls_model.pkl.zip")
    
    if not os.path.exists(reg_model_path) or not os.path.exists(cls_model_path):
        raise ValueError(f"No saved model zip files found in {model_dir}; run training first.")

    from num_pitches import build_model as build_reg_model, predict as predict_reg
    reg_model = build_reg_model()
    reg_model.load_model(reg_model_path)

    from pitching_option import build_model as build_cls_model, predict as predict_cls
    cls_model = build_cls_model()
    cls_model.load_model(cls_model_path)

    # Load extras from Derived_Data/extra
    extras_dir = "Derived_Data/extra"
    with open(os.path.join(extras_dir, "reg_scaler.pkl"), "rb") as f:
        reg_scaler = pickle.load(f)
    with open(os.path.join(extras_dir, "df_reg.pkl"), "rb") as f:
        df_reg_data = pickle.load(f)
    with open(os.path.join(extras_dir, "feature_cols_reg.pkl"), "rb") as f:
        feature_cols_reg = pickle.load(f)

    with open(os.path.join(extras_dir, "cls_scaler.pkl"), "rb") as f:
        cls_scaler = pickle.load(f)
    with open(os.path.join(extras_dir, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(extras_dir, "feature_cols_cls.pkl"), "rb") as f:
        feature_cols_cls = pickle.load(f)
    with open(os.path.join(extras_dir, "target_col.pkl"), "rb") as f:
        target_col = pickle.load(f)
    with open(os.path.join(extras_dir, "df_cls.pkl"), "rb") as f:
        df_cls_data = pickle.load(f)

    print(f"\nLoaded TabNet regression model from: {reg_model_path}")
    print(f"Loaded TabNet classification model from: {cls_model_path}")
    print(f"Running predictions for Pitchers={pitcher_ids} x Batters={batter_ids}, n_pitches={n_pitches}")

    # ========== 1) Regression DataFrame ==========
    reg_records = []
    for p_id in pitcher_ids:
        for b_id in batter_ids:
            reg_pred = predict_reg(p_id, b_id, reg_model, reg_scaler, df_reg_data)
            reg_records.append({
                "PitcherId": p_id,
                "BatterId": b_id,
                "RegressionPrediction": reg_pred
            })
    df_reg = pd.DataFrame(reg_records)

    # ========== 2) Classification DataFrame ==========
    cls_records = []
    for p_id in pitcher_ids:
        for b_id in batter_ids:
            for pitch_number in range(1, n_pitches + 1):
                best_pitch, results_df = predict_cls(
                    p_id, b_id,
                    cls_model, cls_scaler, encoders,
                    df_cls_data, feature_cols_cls, target_col
                )
                if results_df is None or results_df.empty:
                    print(f"No classification data for (Pitcher={p_id}, Batter={b_id}) at pitch# {pitch_number}")
                    continue

                # Add extra columns
                temp_df = results_df.copy()
                temp_df["PitcherId"] = p_id
                temp_df["BatterId"] = b_id
                temp_df["PitchNumber"] = pitch_number
                temp_df["RecommendedPitch"] = best_pitch
                cls_records.append(temp_df)

    df_cls = pd.concat(cls_records, ignore_index=True) if cls_records else pd.DataFrame()

    # Save them in Derived_Data/model_pred
    out_dir = "Derived_Data/model_pred"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reg_path = os.path.join(out_dir, f"reg_prediction_report_{timestamp}.csv")
    cls_path = os.path.join(out_dir, f"cls_prediction_report_{timestamp}.csv")
    df_reg.to_csv(reg_path, index=False)
    df_cls.to_csv(cls_path, index=False)

    print(f"\nSaved regression results to {reg_path}")
    print(f"Saved classification results to {cls_path}\n")

    return df_reg, df_cls

# ====================
# Director
# ====================

def training_pipe():
    """
    Full training pipeline:
      1) Train both models
      2) Save the models (and extras) 
      3) Run predictions with default pitcher/batter IDs
    """
    print("Starting training pipeline...")
    models, extras = train()
    save_models(models, extras)  # also saves the pickled 'extras'
    
    print("\n--- Running a quick test prediction ---\n")
    df_reg, df_cls = predict()  # with default IDs
    print("\n--- Regression results sample ---")
    print(df_reg.head())
    print("\n--- Classification results sample ---")
    print(df_cls.head())
    print("\nTraining pipeline completed.")


if __name__ == "__main__":
    # Example 1: Run the full pipeline (train + save + predict)
    training_pipe()

    # Example 2: If models & extras are already saved, just do predictions
    # (comment out the training_pipe if you only want to do inference)
    df_reg, df_cls = predict()
    print(df_reg.head())
    print(df_cls.head())
