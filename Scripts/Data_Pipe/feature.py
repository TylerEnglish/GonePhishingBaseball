import pyarrow.parquet as pq
import pandas as pd
import pandas as pd
import os
from datetime import datetime

def features(df):
    df = df.copy(deep=True)
    valuable_columns = [
        'Date'
        , 'Time'
        , 'PAofInning'
        , "PitchofPA"
        , 'PitcherId'
        , 'BatterId'
        , "PitcherThrows" # left or right
        , "BatterSide" # left or right
        , 'BatterTeam'
        ,'Inning'
        ,'Top/Bottom'
        ,'Outs'
        ,'Balls'
        ,'Strikes'
        ,'PitchCall'
        ,'KorBB' # might take out but Bracken is using
        , 'CleanPitchType'
        , 'TaggedHitType'
        ,'PlayResult'
        ,'OutsOnPlay'
        ,'RunsScored'
        ,'RelSpeed'
        ,'VertRelAngle'
        ,'HorzRelAngle'
        ,'SpinRate'
        ,'SpinAxis'
        ,'Tilt'
        ,'RelHeight'
        ,'RelSide'
        ,'Extension'
        ,'VertBreak'
        ,'InducedVertBreak'
        ,'HorzBreak'
        ,'PlateLocHeight'
        ,'PlateLocSide'
        ,'ZoneSpeed'
        ,'VertApprAngle'
        ,'HorzApprAngle'
        ,'ZoneTime'
        ,'ExitSpeed'
        ,'Angle'
        ,'Direction'
        ,'HitSpinRate'
        ,'PositionAt110X'
        ,'PositionAt110Y'
        ,'PositionAt110Z'
        ,'Distance'
        ,'LastTrackedDistance'
        ,'Bearing'
        ,'HangTime'
        ,'pfxx'
        ,'pfxz'
        ,'x0'
        ,'y0'
        ,'z0'
        ,'vx0'
        ,'vy0'
        ,'vz0'
        ,'ax0'
        ,'ay0'
        ,'az0'
        ,'GameID'
        ,'PitchUID'
        ,'EffectiveVelo'
        ,'MaxHeight'
        ,'MeasuredDuration'
        ,'SpeedDrop'
        ,'AutoHitType'
        #,'PitchTrajectoryXc0'
        # ,'PitchTrajectoryXc1'
        # ,'PitchTrajectoryXc2'
        # ,'PitchTrajectoryYc0'
        # ,'PitchTrajectoryYc1'
        # ,'PitchTrajectoryYc2'
        # ,'PitchTrajectoryZc0'
        # ,'PitchTrajectoryZc1'
        # ,'PitchTrajectoryZc2'
        ,'HitSpinAxis'
        # ,'HitTrajectoryXc0'
        # ,'HitTrajectoryXc1'
        # ,'HitTrajectoryXc2'
        # ,'HitTrajectoryXc3'
        # ,'HitTrajectoryXc4'
        # ,'HitTrajectoryXc5'
        # ,'HitTrajectoryXc6'
        # ,'HitTrajectoryXc7'
        # ,'HitTrajectoryXc8'
        # ,'HitTrajectoryYc0'
        # ,'HitTrajectoryYc1'
        # ,'HitTrajectoryYc2'
        # ,'HitTrajectoryYc3'
        # ,'HitTrajectoryYc4'
        # ,'HitTrajectoryYc5'
        # ,'HitTrajectoryYc6'
        # ,'HitTrajectoryYc7'
        # ,'HitTrajectoryYc8'
        # ,'HitTrajectoryZc0'
        # ,'HitTrajectoryZc1'
        # ,'HitTrajectoryZc2'
        # ,'HitTrajectoryZc3'
        # ,'HitTrajectoryZc4'
        # ,'HitTrajectoryZc5'
        # ,'HitTrajectoryZc6'
        # ,'HitTrajectoryZc7'
        # HitTrajectoryZc8
        # ,'HitLaunchConfidence' # bc it has so many nulls
        # ,'HitLandingConfidence' # bc it has so many nulls
        ]
    df = df[valuable_columns]
    output_dir = "derived_data/filter"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"filtered_{timestamp}.parquet")
    
    df.to_parquet(output_file, index=False)
    
    print(f"Data saved to {output_file}")

def filter_pipe():
    input_file = "Derived_Data/clean/cleaned_20250228_210153.parquet"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    df = pq.read_table(source=input_file).to_pandas()
    
    features(df)


if __name__ == "__main__":
    filter_pipe()