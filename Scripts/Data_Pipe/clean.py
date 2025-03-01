import pandas as pd
import os
from datetime import datetime

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
  # set 'undefined' in TaggedPitchType to NA then fill it with the value from AutoPitchType
  clean_df['TaggedPitchType'] = clean_df['TaggedPitchType'].replace('undefined', pd.NA)
  clean_df['CleanPitchType'] = clean_df['TaggedPitchType'].fillna(clean_df['AutoPitchType'], inplace=True)

  return clean_df

def clean_pipe():
    df = join()
    df.copy(deep=True)
    df = clean_baseball(df=df)
    df.copy(deep=True)
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