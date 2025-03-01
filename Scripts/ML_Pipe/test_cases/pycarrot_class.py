import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pycaret.classification import (
    setup, compare_models, finalize_model, predict_model
)

# ==================================
# PyCaret Training 
# ==================================

def model_train(data_path="data.parquet"):
    """
    Director function:
      - Loads data from a parquet file.
      - Cleans data (fill missing values, drop rows missing target).
      - Uses PyCaret to set up classification, compare multiple models,
        and select the best based on Accuracy.
      - Finalizes (re-trains on entire data) that best model.
      - Returns the finalized model, the cleaned DataFrame, and the target column.
    """
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None, None, None

    # 1) Load data
    table = pq.read_table(source=data_path)
    df = table.to_pandas()

    # 2) Define target column
    target_col = 'PitchCall'

    # 3) Fill missing values
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

    # Drop rows where the target is missing
    df = df.dropna(subset=[target_col])

    # 4) Set up PyCaret
    # Remove 'silent=True'. You can also remove 'html=False' if you prefer the new PyCaret UI.
    clf_setup = setup(
        data=df,
        target=target_col,
        session_id=42,
        html=False,
        log_experiment=False
        # ignore_features=['PitcherId','BatterId'],  # if you prefer NOT to train on IDs
    )

    # 5) Compare multiple models and pick the best one
    best_model = compare_models(sort='Accuracy', n_select=1)

    # 6) Finalize (retrain on full data)
    final_model = finalize_model(best_model)

    return final_model, df, target_col


# =====================================
# Prediction / Recommender
# =====================================

def predict(pitcher, batter, model, df, target_col="PitchCall"):
    """
    For a given pitcher and batter:
      - Determines candidate pitch types from the pitcher's historical data.
      - Finds the relevant subset of the dataframe (matchup if possible, else overall pitcher data).
      - For each candidate pitch, overrides 'CleanPitchType' in the relevant rows,
        applies predict_model, and averages the Probability of "StrikeCalled".
      - Recommends the pitch with the highest average Probability of "StrikeCalled".
      
    Returns
    -------
    best_pitch : str
        The recommended pitch type.
    results_df : pd.DataFrame
        A table with candidate pitches and their average predicted probabilities for each class
        (especially "StrikeCalled").
    """

    # 1) Filter pitcher data
    pitcher_data = df[df["PitcherId"] == pitcher]
    if pitcher_data.empty:
        print(f"No data found for Pitcher {pitcher}.")
        return None, None

    # Candidate pitch types
    candidate_pitches = pitcher_data["CleanPitchType"].unique()
    if len(candidate_pitches) == 0:
        print(f"No pitch types available for Pitcher {pitcher}.")
        return None, None

    # 2) Try matchup data, else fallback to all pitcher data
    matchup_data = df[(df["PitcherId"] == pitcher) & (df["BatterId"] == batter)]
    if not matchup_data.empty:
        base_data = matchup_data.copy()
    else:
        base_data = pitcher_data.copy()
        print(f"No matchup data found for Pitcher {pitcher} vs Batter {batter}; using general pitcher data.")

    # 3) For each candidate pitch, override 'CleanPitchType', run predictions, average Probability_StrikeCalled
    results = []
    desired_col = "Probability_StrikeCalled"  # adapt if your class name for "StrikeCalled" is different

    for pitch in candidate_pitches:
        # Simulate data
        simulated_data = base_data.copy()
        simulated_data["CleanPitchType"] = pitch

        # Use PyCaret's predict_model to get predicted probabilities
        pred_df = predict_model(model, data=simulated_data, raw_score=True)

        # Verify that Probability_StrikeCalled exists
        if desired_col not in pred_df.columns:
            # If your target classes are different, adjust accordingly.
            print(
                f"Could not find desired probability column '{desired_col}' "
                f"in predict_model output. Available columns: {list(pred_df.columns)}"
            )
            continue

        # Average Probability of StrikeCalled
        avg_prob_strike = pred_df[desired_col].mean() * 100

        result = {
            "PitchType": pitch,
            "AvgProbability_StrikeCalled": round(avg_prob_strike, 2)
        }
        results.append(result)

    # 4) Compile results
    if not results:
        print("No valid pitch simulations returned probabilities.")
        return None, None

    results_df = pd.DataFrame(results)

    # 5) Determine best pitch based on maximum average Probability_StrikeCalled
    best_idx = results_df["AvgProbability_StrikeCalled"].idxmax()
    best_pitch = results_df.loc[best_idx, "PitchType"]
    best_prob = results_df.loc[best_idx, "AvgProbability_StrikeCalled"]

    print(f"Recommended pitch: {best_pitch} with {best_prob}% chance for 'StrikeCalled'")

    return best_pitch, results_df


# =====================================
# Main Execution
# =====================================
if __name__ == "__main__":
    data_path = "Derived_Data/filter/filtered_20250301_000033.parquet"

    # 1) Train using PyCaret (automatically compares multiple models)
    model, df, target_col = model_train(data_path=data_path)

    # 2) Example usage of 'predict' if model is trained
    if model is not None:
        # Replace these IDs with real ones from your data
        pitcher_id = 1000066910.0
        batter_id = 1000032366.0

        best_pitch, results_df = predict(
            pitcher=pitcher_id,
            batter=batter_id,
            model=model,
            df=df,
            target_col=target_col
        )
        
        print("\nFinal Recommended Pitch:", best_pitch)
        print("\nDetailed Prediction Table:")
        print(results_df)
