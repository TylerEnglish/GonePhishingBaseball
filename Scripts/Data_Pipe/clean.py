import pandas as pd
import os
from datetime import datetime
import regex as re
import numpy as np

def join(base_path="Raw_Data/GameData"):
    all_csv_files = []
    
    # Walk through the directory and find all CSV files
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                all_csv_files.append(os.path.join(root, file))
    
    if not all_csv_files:
        print("No CSV files found.")
        return pd.DataFrame()
    
    # Load all CSVs into a list of DataFrames
    dataframes = []
    for csv_file in all_csv_files:
        try:
            df = pd.read_csv(csv_file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not dataframes:
        print("Dataframe Couldnt Be read")
        return pd.DataFrame()
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def clean_baseball(df):
  # remove names and TaggedPitchType
  clean_df = df.drop(columns=['Batter', 'Pitcher', 'Catcher'])
  # replace '' values with NULL
  clean_df = clean_df.replace('', pd.NA)
  # turn "Date" into a date type instead of object
  clean_df['Date'] = pd.to_datetime(clean_df['Date'])
  # turn "Time" into a date type instead of object
  clean_df['Time'] = pd.to_datetime(clean_df['Time'], format='%H:%M:%S.%f')
  # clean and replace undefined for PitchType
  clean_df['TaggedPitchType'] = clean_df['TaggedPitchType'].replace(r'(?i)^undefined', pd.NA, regex=True)
  clean_df['CleanPitchType'] = clean_df['TaggedPitchType'].fillna(clean_df['AutoPitchType'])
  # clean and replace undefined for HitType
  clean_df['TaggedHitType'] = clean_df['TaggedHitType'].replace(r'(?i)^undefined', pd.NA, regex=True)
  clean_df['CleanHitType'] = clean_df['TaggedHitType'].fillna(clean_df['AutoHitType'])

  clean_df["CleanPitchCall"] = clean_df["PitchCall"].apply(lambda x: 
      "Strike" if "strike" in str(x).lower() else 
      "Ball" if "ball" in str(x).lower() else 
      "Hit" if "inplay" in str(x).lower() else 
      "Walk" if "hitbypitch" in str(x).lower() else                                       
      "Foul" if "foul" in str(x).lower() else x
  )
  clean_df["CleanPitchType"] = clean_df["CleanPitchType"].apply(lambda x:  
     "ChangeUp" if re.search(r'(?i)^changeup$', str(x)) else  # Case insensitive match for Changeup
    "TwoSeamFastBall" if re.search(r'(?i)(twoseamfastball|oneseamfastball)', str(x)) else x 
  )


  return clean_df

def fill_missing_values(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = ["PitchofPA", "PitchCall", "PitchType", "CleanPitchType", "PitcherId", "BatterId"]

    # # Identify columns that should be modified
    # columns_to_fill = [col for col in df.columns if col not in exclude_columns]

    # # Optionally convert selected columns to object dtype for string replacements
    # df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].astype(object)

    # # Replace a list of possible missing-value strings with np.nan
    # df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].replace(
    #     to_replace=['NaN', 'nan', 'NULL', 'null', 'None'],
    #     value=np.nan
    # )

    # # Fill all genuine missing values (np.nan, None, pd.NA) with "no recorded value"
    # df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].fillna('no recorded value')
    df.fillna('no recorded value')

    return df

def clean_pipe():
    df = join()
    df = df.copy(deep=True)
    df = clean_baseball(df=df)
    df = df.copy(deep=True)
    df = fill_missing_values(df)
    df = df.copy(deep=True)
    output_dir = "derived_data/clean"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_format = "parquet"
    output_file = os.path.join(output_dir, f"cleaned_{timestamp}.{output_format}")
    if output_format == "csv":
        df.to_csv(output_file, index=False)
    elif output_format == "parquet":
        df.to_parquet(output_file, index=False)
    
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    clean_pipe()