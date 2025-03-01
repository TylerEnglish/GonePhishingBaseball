import pyarrow.parquet as pq
import pandas as pd
import os
from datetime import datetime
import numpy as np

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
        ,'CleanPitchCall'
        ,'KorBB' # might take out but Bracken is using
        , 'CleanPitchType'
        , 'CleanHitType'
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
        ,'SpeedDrop'
        ,'PitcherTeam'
        ,'HitSpinAxis'
        ]
    df = df[valuable_columns]
    df = df.copy(deep=True)
    return df


def filter_pipe():
    input_file = "Derived_Data/clean/cleaned_20250301_102622.parquet"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    df = pq.read_table(source=input_file).to_pandas()
    
    df = features(df)
    df = df.copy(deep=True)

    output_dir = "derived_data/filter"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"filtered_{timestamp}.parquet")
    
    df.to_parquet(output_file, index=False)
    
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    filter_pipe()