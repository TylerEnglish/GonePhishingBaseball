import os
import glob
import re
import argparse
from datetime import datetime
if __name__ == "__main__":
    from clean import clean_pipe
    from filter import filter_pipe
    from feature import feature_pipe
else:
    from Scripts.Data_Pipe.clean import clean_pipe
    from Scripts.Data_Pipe.filter import filter_pipe
    from Scripts.Data_Pipe.feature import feature_pipe

def get_most_recent_file(directory, prefix):
    """
    Searches for files in 'directory' that start with 'prefix' and have a timestamp
    in the format YYYYMMDD_HHMMSS. Returns the file with the latest timestamp.
    """
    pattern = os.path.join(directory, prefix + "*")
    files = glob.glob(pattern)
    if not files:
        return None

    def extract_timestamp(file_path):
        base = os.path.basename(file_path)
        # Expecting a file name like prefixYYYYMMDD_HHMMSS.ext
        match = re.search(rf"{prefix}(\d{{8}}_\d{{6}})", base)
        if match:
            timestamp_str = match.group(1)
            try:
                return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                return datetime.min
        return datetime.min

    files.sort(key=extract_timestamp, reverse=True)
    return files[0]

def run_clean():
    print("Running clean pipeline...")
    clean_pipe()
    recent_clean = get_most_recent_file("derived_data/clean", "cleaned_")
    if recent_clean:
        print("Most recent clean file:", recent_clean)
    else:
        print("No clean file found.")

def run_filter():
    print("Running filter pipeline...")
    filter_pipe()
    recent_filter = get_most_recent_file("derived_data/filter", "filtered_")
    if recent_filter:
        print("Most recent filter file:", recent_filter)
    else:
        print("No filter file found.")

def run_feature():
    print("Running feature pipeline...")
    feature_pipe()
    recent_feature = get_most_recent_file("derived_data/feature", "feature_")
    if recent_feature:
        print("Most recent feature file:", recent_feature)
    else:
        print("No feature file found.")

def main():
    parser = argparse.ArgumentParser(
        description="Data Director: Run pipelines based on changes."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "clean", "filter", "feature"],
        default="all",
        help=("Pipeline stage to run: "
              "'all' or 'clean' to run clean -> filter -> feature; "
              "'filter' to run filter -> feature; "
              "'feature' to run only feature.")
    )
    args = parser.parse_args()

    if args.stage in ("all", "clean"):
        run_clean()
        run_filter()
        run_feature()
    elif args.stage == "filter":
        run_filter()
        run_feature()
    elif args.stage == "feature":
        run_feature()

if __name__ == "__main__":
    main()
