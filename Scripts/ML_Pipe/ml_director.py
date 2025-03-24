import os
import logging
import zipfile
import pickle
import pandas as pd
import numpy as np
from joblib import dump, load
import pyarrow.parquet as pq

if __name__ == "__main__": 
    from Recommend.model import (
        load_data,
        compute_features,
        PitchSequencingMDP,
        prepare_game_state,
        recommend_pitch,
        simulate_next_pitches,
        main as mdp_train_main  # used during training (if needed)
    )
    from Base_Models.num_pitches import (
        model_train as train_reg,         # returns (reg_model, scaler, df_agg, feature_cols)
        predict as predict_reg,
        build_model as build_reg_model,
        load_regression_model
    )
    from Base_Models.pitching_option import (
        model_train as train_cls,         # returns (model, scaler, encoders, feature_cols, target_col, df, scores)
        predict as predict_cls,
        build_model as build_cls_model,
        predict_single_pitch
    )
    from Base_Models.pitching_option import build_model as build_tabnet_model
else:
    from Scripts.ML_Pipe.Recommend.model import (
        load_data,
        compute_features,
        PitchSequencingMDP,
        prepare_game_state,
        recommend_pitch,
        simulate_next_pitches,
        main as mdp_train_main
    )
    from Scripts.ML_Pipe.Base_Models.num_pitches import (
        model_train as train_reg,
        predict as predict_reg,
        build_model as build_reg_model,
        load_regression_model
    )
    from Scripts.ML_Pipe.Base_Models.pitching_option import (
        model_train as train_cls,
        predict as predict_cls,
        build_model as build_cls_model,
        predict_single_pitch
    )
    from Scripts.ML_Pipe.Base_Models.pitching_option import build_model as build_tabnet_model
# -----------------------------------
#  Loading
# -----------------------------------
def load_pickle_zip(file_path):
    """
    Loads a single pickled object from a .zip archive.
    Assumes there's exactly one file inside the zip.
    """
    with zipfile.ZipFile(file_path, 'r') as zipf:
        inner = zipf.namelist()[0]
        with zipf.open(inner) as f:
            return pickle.load(f)

# -------------------------------------------------------------------
#   GLOBAL PATHS (Adjust as needed)
# -------------------------------------------------------------------
REG_JSON_PATH      = "Derived_Data/model_params/model_params.json"  # XGB JSON
REG_SCALER_ZIP     = "Derived_Data/extra/reg_scaler.pkl.zip"
REG_DF_ZIP         = "Derived_Data/extra/df_reg.pkl.zip"

TABNET_MODEL_ZIP   = "Derived_Data/model_params/cls_model.pkl.zip"  # TabNet
TABNET_SCALER_ZIP  = "Derived_Data/extra/cls_scaler.pkl.zip"
TABNET_ENCODERS_ZIP= "Derived_Data/extra/encoders.pkl.zip"
TABNET_FEATS_ZIP   = "Derived_Data/extra/feature_cols_cls.pkl.zip"
TABNET_TARGET_ZIP  = "Derived_Data/extra/target_col.pkl.zip"
TABNET_DATA_ZIP    = "Derived_Data/extra/df_cls.pkl.zip"

MDP_ZIP_PATH       = "models/recommend/model.zip"                  # MDP-based pipeline
MDP_PARQUET        = "Derived_Data/feature/feature_20250301_105232.parquet"

# =========================
# Utilities
# =========================
def grab_load():
    """
    Checks if key model files exist. 
    """
    missing = False
    if not os.path.exists(REG_JSON_PATH):
        missing = True
        logging.warning(f"Missing Regression model: {REG_JSON_PATH}")
    if not os.path.exists(TABNET_MODEL_ZIP):
        missing = True
        logging.warning(f"Missing TabNet model: {TABNET_MODEL_ZIP}")
    if not os.path.exists(MDP_ZIP_PATH):
        missing = True
        logging.warning(f"Missing MDP pipeline: {MDP_ZIP_PATH}")

    if missing:
        logging.warning("Some model files are missing; now training all models...")
        train()
    else:
        logging.info("All model files found. Ready to predict.")

def train():
    """
    Trains all models if they are not already present.
    """
    # (A) Train Regression model.
    logging.info("Training XGB regression model for # of pitches...")
    reg_model, scaler, df_agg, feature_cols = train_reg()
    if reg_model is None:
        logging.error("Regression training failed.")
        return
    else:
        # Save the regression model if not already saved.
        # (Assuming your train_reg() saves the JSON; if not, you can call reg_model.save_model(REG_JSON_PATH) here.)
        logging.info(f"Regression model should be saved at {REG_JSON_PATH}.")
        # Save scaler and aggregated data:
        os.makedirs(os.path.dirname(REG_SCALER_ZIP), exist_ok=True)
        with zipfile.ZipFile(REG_SCALER_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('reg_scaler.pkl', pickle.dumps(scaler))
        with zipfile.ZipFile(REG_DF_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('df_reg.pkl', pickle.dumps(df_agg))
        logging.info("Regression scaler and data stored.")

    # (B) Train TabNet classification model.
    logging.info("Training TabNet classification model for CleanPitchType prediction...")
    train_cls()  
    logging.info(f"TabNet model should be produced at {TABNET_MODEL_ZIP} along with its extras.")

    # (C) Train MDP-based pipeline.
    logging.info("Training MDP-based pipeline (RF + MDP)...")
    mdp_train_main()  # This function should save model.pkl in models/recommend/
    logging.info("MDP-based pipeline training complete.")

# =========================
# Prediction pipeline
# =========================
def predict(
    pitcher_id=1000066910.0,
    batter_id=1000032366.0,
    count: str = None,
    outs: int = None,
    inning: int = None,
    override_n_pitches: int = None
):
    """
    Loads pre-trained models and runs the full prediction pipeline:
      1) Uses the regression model to predict number of pitches.
      2) Uses the MDP-based pipeline to simulate that many pitches.
      3) For each recommended pitch, forces that pitch in the TabNet model
         to obtain a dictionary of probabilities.
      4) Returns a DataFrame with PitchNumber, Pitch, and Probabilities.
    """
    # A) Ensure models exist.
    grab_load()

    # B) Load Regression Model
    
    reg_model = load_regression_model(REG_JSON_PATH)
    reg_scaler = load_pickle_zip(REG_SCALER_ZIP)
    df_reg_data = load_pickle_zip(REG_DF_ZIP)

    if override_n_pitches is not None and override_n_pitches > 0:
        predicted_n = override_n_pitches
    else:
        predicted_n = predict_reg(pitcher=pitcher_id, batter=batter_id,
                                   model=reg_model, scaler=reg_scaler, df=df_reg_data)
        if not isinstance(predicted_n, int) or predicted_n < 1:
            predicted_n = 1
    logging.info(f"[Regression] Simulating {predicted_n} pitch(es).")

    # C) Load MDP-based pipeline from zip.
    if not os.path.exists(MDP_ZIP_PATH):
        raise FileNotFoundError(f"Missing MDP pipeline zip at {MDP_ZIP_PATH}. Please ensure models are trained.")
    with zipfile.ZipFile(MDP_ZIP_PATH, 'r') as zipf:
        with zipf.open('model.pkl') as f:
            mdp_rf_model = load(f)

    df_raw = load_data(MDP_PARQUET)
    df_features = compute_features(df_raw)

    mdp = PitchSequencingMDP(df_features)
    mdp.estimate_transition_probabilities()
    mdp_policy = mdp.solve_mdp()

    final_count = count if count else "1-1"
    final_outs = outs if outs is not None else 1
    final_inning = inning if inning is not None else 6
    gs = prepare_game_state(count=final_count, outs=final_outs, inning=final_inning,
                            batter_id=int(batter_id), pitcher_id=int(pitcher_id),
                            hist_df=df_features)

    recommended_pitches = simulate_next_pitches(initial_state=gs,
                                                 supervised_model=mdp_rf_model,
                                                 mdp_policy=mdp_policy,
                                                 mdp=mdp,
                                                 df_features=df_features,
                                                 n=predicted_n)
    logging.info(f"[MDP] Sequence: {recommended_pitches}")

    # D) Load TabNet model and compute forced single-pitch probabilities.
    if not os.path.exists(TABNET_MODEL_ZIP):
        logging.warning(f"No TabNet model found at {TABNET_MODEL_ZIP}, skipping probability calculation.")
        return pd.DataFrame({
            "PitchNumber": range(1, len(recommended_pitches) + 1),
            "Pitch": recommended_pitches,
            "Probabilities": [None] * len(recommended_pitches)
        })
    tabnet_model = build_tabnet_model()
    tabnet_model.load_model(TABNET_MODEL_ZIP)

    cls_scaler   = load_pickle_zip(TABNET_SCALER_ZIP)
    cls_encoders = load_pickle_zip(TABNET_ENCODERS_ZIP)
    cls_features = load_pickle_zip(TABNET_FEATS_ZIP)
    cls_target   = load_pickle_zip(TABNET_TARGET_ZIP)
    df_tabnet    = load_pickle_zip(TABNET_DATA_ZIP)

    results = []
    for i, pitch in enumerate(recommended_pitches, start=1):
        prob_dict = predict_single_pitch(pitcher_id=pitcher_id,
                                         batter_id=batter_id,
                                         pitch_type=pitch,
                                         model=tabnet_model,
                                         scaler=cls_scaler,
                                         encoders=cls_encoders,
                                         df_tabnet=df_tabnet,
                                         feature_cols=cls_features,
                                         target_col=cls_target)
        if prob_dict is None:
            formatted = None
        else:
            # Format each probability to 4 decimals.
            formatted = {k: float(f"{v:.4f}") for k, v in prob_dict.items()}
        results.append({
            "PitchNumber": i,
            "Pitch": pitch,
            "Probabilities": formatted
        })

    pitch_stats_df = pd.DataFrame(results)
    logging.info("=== Final Pitch Sequence with Probabilities ===")
    logging.info(pitch_stats_df.to_string(index=False))
    return pitch_stats_df

# =============================
# Recommendation only
# =============================
def predict_sequence(
    pitcher_id=1000066910.0,
    batter_id=1000032366.0,
    count: str = "1-1",
    outs: int = 1,
    inning: int = 6,
    n: int = 5
):
    """
    Ignores regression & TabNet; loads the MDP-based pipeline and simulates n pitches.
    """
    grab_load()
    if not os.path.exists(MDP_ZIP_PATH):
        raise FileNotFoundError(f"Missing MDP model zip at {MDP_ZIP_PATH}")
    with zipfile.ZipFile(MDP_ZIP_PATH, 'r') as zipf:
        with zipf.open('model.pkl') as f:
            mdp_rf_model = load(f)
    df_raw = load_data(MDP_PARQUET)
    df_features = compute_features(df_raw)
    mdp = PitchSequencingMDP(df_features)
    mdp.estimate_transition_probabilities()
    mdp_policy = mdp.solve_mdp()
    gs = prepare_game_state(count=count, outs=outs, inning=inning,
                            batter_id=int(batter_id), pitcher_id=int(pitcher_id),
                            hist_df=df_features)
    recommended_seq = simulate_next_pitches(gs, mdp_rf_model, mdp_policy, mdp, df_features, n=n)
    logging.info(f"[MDP] Only Sequence: {recommended_seq}")
    return recommended_seq

# =============================
# MAIN 
# =============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Instead of training, we assume the models are already trained and stored.
    grab_load()  # This should log "All model files found. Ready to predict."

    '''
    Full intergrated
    '''
    # Run the combined prediction pipeline.
    df = predict(
        pitcher_id=1000066910.0,
        batter_id=1000032366.0,
        count="2-1",  # Balls - Strikes
        outs=2,
        inning=7,
        override_n_pitches=None  # Use regression result
    )
    print("\n\n\n\n=== Combined Predict Results ===") # -> outputs a dictionary of [{'pitch':{'ball':0.1231, 'strike':...,...}}]
    for idx, row in df.iterrows():
        print(f"Pitch {row['PitchNumber']} ({row['Pitch']}):")
        if row['Probabilities'] is not None:
            for cls, prob in row['Probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
        else:
            print("  No probability data.")
    print("\n=== End Combined Predict Results ===\n\n\n\n\n")

    '''
    Leave out regression
    '''
    df = predict(
        pitcher_id=1000066910.0,
        batter_id=1000032366.0,
        count="2-1",
        outs=2,
        inning=7,
        override_n_pitches=10  # Doesnt use Regression Result (can use a SD instead)
    )
    print("\n\n\n\n=== Combined Predict Results (no regression)===")
    for idx, row in df.iterrows():
        print(f"Pitch {row['PitchNumber']} ({row['Pitch']}):")
        if row['Probabilities'] is not None:
            for cls, prob in row['Probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
        else:
            print("  No probability data.")
    print("\n=== End Combined Predict Results ===\n\n\n\n\n")

    '''
    Only Recommendation sequence
    '''
    # Recommend-only sequence.
    seq = predict_sequence(
        pitcher_id=1000066910.0,
        batter_id=1000032366.0,
        count="3-2",
        outs=2,
        inning=8,
        n=10
    )
    print(f"\n=== MDP-only Sequence: {seq} ===\n")
