import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
from scipy.stats import entropy  # Entropy helps measure unpredictability in pitch selection

def compute_basic_features(df):
    basic_features = df.groupby(["PitcherId", "BatterId"]).agg(
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

    return basic_features

def compute_advanced_features(df):
    # Function to compute Shannon entropy for pitch sequences
    from scipy.stats import entropy

    def pitch_entropy(pitches):
        pitch_counts = pitches.value_counts(normalize=True)
        return entropy(pitch_counts)

    advanced_features = df.groupby(["PitcherId", "BatterId"]).agg(
        Pitch_Type_Diversity=("TaggedPitchType", lambda x: x.nunique()),
        Max_Effective_Velocity=("EffectiveVelo", "max"),
        Avg_Velocity_Drop=("SpeedDrop", "mean"),
        Breaking_Ball_Ratio=("TaggedPitchType", lambda x: (x.str.contains("Curve|Slider")).sum() / len(x)),
        Pitch_Sequencing_Entropy=("TaggedPitchType", pitch_entropy),
        Pitch_Zonal_Targeting=("PlateLocHeight", lambda x: (x < x.median()).sum() / len(x)),
        Fastball_to_Offspeed_Ratio=("TaggedPitchType", lambda x: (x.str.contains("Fastball")).sum() / ((~x.str.contains("Fastball")).sum() + 1)),  # Avoid div by 0
        Vertical_vs_Horizontal_Break_Ratio=("InducedVertBreak", lambda x: x.mean() / (df.loc[x.index, "HorzBreak"].mean() + 1e-6)),  # Avoid div by 0
        Release_Extension_Deviation=("Extension", "std"),
        Avg_Hit_Exit_Velocity=("ExitSpeed", "mean")
    ).reset_index()

    return advanced_features
def compute_more_advanced_features(df):

    def pitch_variance(pitches):
        """Measures how diverse the pitch selection is for a specific pitcher-batter matchup.
        If the pitcher keeps throwing the same pitch, variance will be low."""
        return pitches.nunique() / len(pitches) if len(pitches) > 0 else 0

    def speed_consistency(speeds):
        """Measures how consistent the pitch speeds are. A lower standard deviation means a pitcher is keeping their speed steady."""
        return speeds.std()

    def breaking_vs_fastball_ratio(pitches):
        """Calculates the ratio of breaking balls (curveballs, sliders) to fastballs.
        Some pitchers rely more on off-speed pitches, while others dominate with heat."""
        breaking_balls = pitches.str.contains("Curve|Slider", na=False).sum()
        fastballs = pitches.str.contains("Fastball", na=False).sum()
        return breaking_balls / (fastballs + 1)  # Adding 1 to avoid division by zero

    def strikeout_likelihood(play_results):
        """Estimates the likelihood of a strikeout happening in a given pitcher-batter matchup."""
        return (play_results == "Strikeout").sum() / len(play_results) if len(play_results) > 0 else 0

    more_advanced_features = df.groupby(["PitcherId", "BatterId"]).agg(
        # Measures the unpredictability of the pitch sequence—higher values mean the pitcher mixes things up.
        Pitch_Type_Variance=("TaggedPitchType", pitch_variance),

        # Captures how consistent the pitcher's velocity is—low values mean they keep their speed steady.
        Speed_Consistency=("RelSpeed", speed_consistency),

        # Tracks how often a pitcher throws breaking balls vs. fastballs—important for deception.
        Breaking_vs_Fastball_Ratio=("TaggedPitchType", breaking_vs_fastball_ratio),

        # Looks at how often this specific pitcher-batter matchup results in a strikeout.
        Strikeout_Likelihood=("PlayResult", strikeout_likelihood),

        # Measures the average horizontal movement of pitches—big movers are harder to hit.
        Avg_Horizontal_Movement=("HorzBreak", "mean"),

        # Measures the average vertical drop on pitches—key for curveballs and fastballs with sink.
        Avg_Vertical_Movement=("InducedVertBreak", "mean"),

        # Checks how much velocity changes from pitch to pitch—some pitchers mix speeds well.
        Change_in_Speed_Per_Pitch=("RelSpeed", lambda x: x.diff().mean(skipna=True)),

        # Estimates the probability of a batter making solid contact and getting a hit.
        Hit_Probability=("PlayResult", lambda x: (x == "Hit").sum() / len(x)),

        # Measures how often this specific batter strikes out in general.
        Batter_Strikeout_Tendency=("BatterId", lambda x: (df.loc[x.index, "PlayResult"] == "Strikeout").sum() / len(x)),

        # Looks at how aggressive a pitcher is—do they attack the zone early or nibble?
        Pitcher_Aggressiveness=("Balls", lambda x: (x == 0).sum() / len(x))

    ).reset_index()

    return more_advanced_features




# ======================================
# Main Pipe
# ====================================

def feature_pipe():
    input_file = "Derived_Data/filter/filtered_20250228_231053.parquet"
    
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