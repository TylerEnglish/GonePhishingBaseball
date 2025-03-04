# Data Pipeline Overview for Baseball Data

This document explains the structure and functionality of the data pipeline for processing baseball game data. The pipeline is composed of multiple stages, each encapsulated in its own Python file. These stages include data cleaning, filtering, feature engineering, and a director script to coordinate the entire workflow.

## Overview

The data pipeline is designed to process raw game data and progressively transform it into enriched datasets suitable for further analysis or modeling. The key stages are:

- **Data Cleaning:**  
  Raw CSV files are read and cleaned. This stage handles tasks such as concatenating multiple CSV files, removing unnecessary columns, correcting data types (e.g., dates and times), and handling missing or undefined values. The cleaned data is saved in a structured format.

- **Data Filtering:**  
  The cleaned data is then filtered to retain only the most valuable columns for analysis. This step simplifies the dataset, ensuring that subsequent operations focus on the key features required for feature engineering.

- **Feature Engineering:**  
  Advanced computations are performed on the filtered data to extract both basic and complex features. Statistical summaries, ratios, and entropy calculations are examples of the transformations applied to capture deeper insights into pitch characteristics and game events.

- **Pipeline Coordination:**  
  A dedicated director script orchestrates the execution of the entire pipeline. Depending on which component has been updated, the director can run the entire workflow or only specific stages (e.g., running only the filtering and feature engineering steps if the filtering logic has changed).

## Table of Contents

- [Data Pipeline Overview for Baseball Data](#data-pipeline-overview-for-baseball-data)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Data Cleaning Stage](#data-cleaning-stage)
  - [Data Filtering Stage](#data-filtering-stage)
  - [Feature Engineering Stage](#feature-engineering-stage)
  - [Pipeline Coordination](#pipeline-coordination)
  - [How to Run the Pipeline](#how-to-run-the-pipeline)
  - [Conclusion](#conclusion)

## Data Cleaning Stage

The **data cleaning** component processes raw input data from game CSV files. It performs several key functions:
- Reads and merges CSV files from a designated raw data directory.
- Cleans the data by removing unwanted columns (such as player names) and handling undefined or missing values.
- Converts columns like "Date" and "Time" into appropriate datetime formats.
- Saves the cleaned data as a parquet file, timestamped to track different processing runs.

## Data Filtering Stage

The **data filtering** component builds upon the cleaned dataset by:
- Selecting only a subset of columns that are considered valuable for analysis.
- Preparing the data in a more compact and efficient format for further processing.
- Outputting the filtered dataset as a parquet file, again with a timestamp embedded in the filename.

## Feature Engineering Stage

The **feature engineering** stage takes the filtered data and applies a series of computations to extract enhanced features. This includes:
- Calculating basic statistical measures (such as average pitch speed and angles).
- Deriving advanced metrics like pitch sequencing entropy and ratios related to pitch types.
- Combining computed features back with the original data to produce enriched datasets.
- Producing final outputs that are saved in a structured format for use in analysis or predictive modeling.

## Pipeline Coordination

The **pipeline coordination** script manages the execution order of the individual stages. It offers flexibility by allowing selective execution:
- **Full Pipeline:** Runs the complete sequence (cleaning → filtering → feature engineering) when new data is added.
- **Selective Runs:** Executes only the filtering and feature stages if changes are made in filtering logic, or just the feature stage if modifications occur in the feature engineering process.

This approach streamlines the workflow, ensuring that only the necessary steps are re-run after modifications, thereby saving time and computational resources.

## How to Run the Pipeline

You can control which parts of the pipeline run using command-line arguments in the director script (`data_director.py`). Below are some example commands:

- **Run everything (clean → filter → feature):**
    ```bash
    python data_director.py --stage all
    ```
  - This option will process raw data through all stages.
- **Run only the clean pipeline (which will then run filter and feature too):**
    ```bash
    python data_director.py --stage clean
    ```
    - Useful when the raw data or cleaning logic changes.
- **Run only the filter pipeline (then feature):**
    ```bash
    python data_director.py --stage filter
    ```
    - Ideal for when the filtering logic is updated.
- **Run only the feature pipeline:**
    ```bash
    python data_director.py --stage feature
    ```
    - For updates related only to feature computation.

By adjusting the command-line flags, you can easily integrate the pipeline into automated workflows or CI/CD systems like GitHub Actions.

## Conclusion

This multi-stage data pipeline provides an efficient and flexible framework for processing baseball game data. Each stage is modular, allowing for easy updates and targeted re-runs, which is critical for maintaining a robust data processing system. The director script ties all components together, ensuring that changes propagate correctly through the pipeline.
