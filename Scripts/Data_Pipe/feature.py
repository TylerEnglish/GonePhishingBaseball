import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
from scipy.stats import entropy  # Entropy helps measure unpredictability in pitch selection

def compute_basic_features(df):
    basic_features = df.groupby(["Date", "PitchofPA", "PitcherId", "BatterId"]).agg(
        Avg_Pitch_Speed=("RelSpeed", "mean"),
        Avg_Vertical_Release_Angle=("VertRelAngle", "mean"),
        Avg_Horizontal_Release_Angle=("HorzRelAngle", "mean"),
        Avg_Spin_Rate=("SpinRate", "mean"),
        Avg_Spin_Axis=("SpinAxis", "mean"),
        Strike_Percentage=("PitchCall", lambda x: (x == "Strike").sum() / len(x)),
        Ball_Percentage=("PitchCall", lambda x: (x == "Ball").sum() / len(x)),
        Outs_Created=("OutsOnPlay", "sum"),
        Avg_PlateLocHeight=("PlateLocHeight", "mean"),
        Avg_PlateLocSide=("PlateLocSide", "mean")
    ).reset_index()

    # merge the basic_features back with original df modified by the same groupby
    basic_features = df.merge(basic_features, on=["Date", "PitchofPA", "PitcherId", "BatterId"])

    return basic_features

def compute_advanced_features(df):
    # Function to compute Shannon entropy for pitch sequences
    from scipy.stats import entropy

    def pitch_entropy(pitches):
        pitch_counts = pitches.value_counts(normalize=True)
        return entropy(pitch_counts)

    advanced_features = df.groupby(["Date", "PitchofPA", "PitcherId", "BatterId"]).agg(
        Pitch_Type_Diversity=("CleanPitchType", lambda x: x.nunique()),
        Max_Effective_Velocity=("EffectiveVelo", "max"),
        Avg_Velocity_Drop=("SpeedDrop", "mean"),
        Breaking_Ball_Ratio=(
            "CleanPitchType",
            lambda x: (x.str.contains("Curve|Slider", na=False)).sum() / len(x)
        ),
        Pitch_Sequencing_Entropy=("CleanPitchType", pitch_entropy),
        Pitch_Zonal_Targeting=(
            "PlateLocHeight",
            lambda x: (x < x.median()).sum() / len(x)
        ),
        Fastball_to_Offspeed_Ratio=(
            "CleanPitchType",
            # Use na=False so ~ operator works correctly
            lambda x: (x.str.contains("Fastball", na=False)).sum() 
                    / ((~x.str.contains("Fastball", na=False)).sum() + 1)
        ),
        Vertical_vs_Horizontal_Break_Ratio=(
            "InducedVertBreak",
            lambda x: x.mean() / (df.loc[x.index, "HorzBreak"].mean() + 1e-6)
        ),
        Release_Extension_Deviation=("Extension", "std"),
        Avg_Hit_Exit_Velocity=("ExitSpeed", "mean")
    ).reset_index()

    # merge the advance_features back with original df modified by the same groupby
    advanced_features_df = df.merge(advanced_features, on=["Date", "PitchofPA", "PitcherId", "BatterId"])

    return advanced_features_df

def compute_more_advanced_features(df):
    """
    Computes advanced features for pitch-level data, including smoothed metrics for 
    hit probability, strikeout likelihood, and batter strikeout tendency.
    Smoothed Pobanility: alpha = 1, beta = 2 often
    """
    
    # Helper functions for pitch-level features
    def pitch_variance(pitches):
        return pitches.nunique() / len(pitches) if len(pitches) > 0 else 0

    def speed_consistency(speeds):
        return speeds.std()

    def breaking_vs_fastball_ratio(pitches):
        breaking_balls = pitches.str.contains("Curve|Slider", na=False).sum()
        fastballs = pitches.str.contains("Fastball", na=False).sum()
        return breaking_balls / (fastballs + 1)  # Avoid division by zero

    # New helper for Laplace smoothing
    def smoothed_probability(series, event, alpha=1, beta=2):
        count = (series == event).sum()
        return (count + alpha) / (len(series) + beta) if len(series) > 0 else 0

    # ----------------------------
    # Compute pitch-level features
    # ----------------------------
    pitch_level_features = df.groupby(["Date", "PitchofPA", "PitcherId", "BatterId"]).agg(
        Pitch_Type_Variance=("CleanPitchType", pitch_variance),
        Speed_Consistency=("RelSpeed", speed_consistency),
        Breaking_vs_Fastball_Ratio=("CleanPitchType", breaking_vs_fastball_ratio),
        Avg_Horizontal_Movement=("HorzBreak", "mean"),
        Avg_Vertical_Movement=("InducedVertBreak", "mean"),
        Change_in_Speed_Per_Pitch=("RelSpeed", lambda x: x.diff().mean(skipna=True)),
        Pitcher_Aggressiveness=("Balls", lambda x: (x == 0).sum() / len(x))
    ).reset_index()

    # ----------------------------------------
    # Compute matchup-level features with smoothing
    # (aggregated over all pitches between a pitcher and batter)
    # ----------------------------------------
    matchup_features = df.groupby(["PitcherId", "BatterId"]).agg(
        Hit_Probability=("PlayResult", lambda x: smoothed_probability(x, "Hit", alpha=1, beta=2)),
        Strikeout_Likelihood=("PlayResult", lambda x: smoothed_probability(x, "Strikeout", alpha=1, beta=2))
    ).reset_index()

    # ----------------------------------------
    # Compute batter-level features with smoothing
    # (aggregated over all pitches faced by the batter)
    # ----------------------------------------
    batter_features = df.groupby("BatterId").agg(
        Batter_Strikeout_Tendency=("PlayResult", lambda x: smoothed_probability(x, "Strikeout", alpha=1, beta=2))
    ).reset_index()

    # ----------------------------------------
    # Merge the advanced features back to the pitch-level DataFrame
    # ----------------------------------------
    combined_features = pitch_level_features.merge(matchup_features, on=["PitcherId", "BatterId"], how="left")
    combined_features = combined_features.merge(batter_features, on="BatterId", how="left")
    final_df = df.merge(combined_features, on=["Date", "PitchofPA", "PitcherId", "BatterId"], how="left")

    return final_df

# ======================================
# Main Pipe
# ====================================

def feature_pipe():
    input_file = "Derived_Data/filter/filtered_20250301_031659.parquet"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    df = pq.read_table(source=input_file).to_pandas()
    
    df = df.copy(deep=True)
    df = compute_basic_features(df)

    df = df.copy(deep=True)
    df = compute_advanced_features(df)

    output_dir = "derived_data/feature"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"feature_{timestamp}.parquet")
    
    df.to_parquet(output_file, index=False)
    
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    feature_pipe()