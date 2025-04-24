# HackUSU: Team Gone Phishing Pioneer Baseball AI/ML Project

This project processes baseball game data to train machine learning models for predicting optimal pitching sequences and provides an interactive dashboard for data exploration and predictions.

## Table of Contents

- [HackUSUS: Team Gone Phishing Pioneer Baseball AI/ML Project](#hackusus-team-gone-phishing-pioneer-baseball-aiml-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [File Structure](#file-structure)
  - [How to Run](#how-to-run)
  - [Contributors](#contributors)

## Overview

The project has two main components:

- **Machine Learning Pipeline:**  
  - Trains both regression and classification models using historical game data.
  - Saves models and supporting artifacts (scalers, encoders, feature columns, etc.) in organized directories.
  - Includes an advanced PyTorch model (a multi-level transformer) with custom training, validation, and prediction routines.
  - Produces CSV reports with prediction results.

- **Interactive Dashboard:**  
  - Built with Streamlit for an easy-to-use, web-based interface.
  - Allows users to filter and visualize game data.
  - Supports file uploads to run model predictions on custom data.

## Requirements

- **Python 3.12**  
- All required packages are listed in the `requirement.txt` file.
  - [Requirements File](requirements.txt)

## File Structure

- **Data Directories:**
  - `Raw_Data/GameData/` – Contains raw CSV files.
  - `Derived_Data/` – Contains cleaned, filtered, and feature-engineered data, plus saved models, extra artifacts, and prediction reports.

- **Scripts:**
  - **Data Pipeline (Scripts/Data_Pipe/):**
    - `clean.py` – Cleans and merges raw data.
    - `filter.py` – Filters the cleaned data.
    - `feature.py` – Generates advanced features.
    - `data_director.py` – Orchestrates the execution of the data processing pipeline. Supports running the full pipeline or only specific stages if changes are detected.
  
  - **Machine Learning Pipeline (`Scripts/ML_Pipe/`):**

  - **Regression Model (`Scripts/ML_Pipe/Base_Models/num_pitches.py`):**  
    - Models tested: **PyCaret**, **TabNetRegressor**, **XGBoost**.
    - Final selection: **XGBoost**.
    - Originally intended to use TabNet for consistency across regression and classification tasks but chose XGBoost based on PyCaret performance evaluations.

  - **Recommendation System (`Scripts/ML_Pipe/Recommend/model.py`):**
    - Models used: **RandomForestClassifier** and **Markov Decision Process (MDP)**.
    - **RandomForestClassifier**:
      - Predicts pitch types using real-time game data (count, pitch speed, spin rate, pitcher/batter handedness).
      - Outputs pitch probabilities.
    - **Markov Decision Process**:
      - Uses historical pitch sequences to determine optimal pitches for different game scenarios.
      - Combines predictive probabilities from RandomForest with strategic MDP decisions and slight randomness to enhance realism and flexibility.

  - **Scoring Model (`Scripts/ML_Pipe/Base_Models/pitching_options.py`):**
    - Model: **TabNetClassifier**.
    - Metrics used:
      - **Accuracy** – standard predictive performance measure.
      - **F1 Score** – ensures a balanced evaluation of precision and recall.
      - **Precision** – specifically emphasized to minimize false positives, avoiding scenarios where expected strikes or outs result unexpectedly in balls in play or home runs.

  - **Pipeline Coordination (`Scripts/ML_Pipe/ml_director.py`):**
    - Manages the structured execution of the pipeline:
      1. **Regression Model** predicts the expected number of pitches a pitcher will throw.
      2. **Recommendation System** identifies the optimal next pitch based on the current game situation.
      3. **Scoring Model** evaluates and scores the quality of the recommended pitch.

- **Dashboard:**
  - `streamlit_app.py` – Launches the interactive dashboard for data visualization and model predictions.

## How to Run

1. **Install Dependencies:**  
   Ensure you are using Python 3.12 and install all packages from `requirement.txt`.
   ```bash
    pip install -r requirement.txt
   ```

2. **Run the Data Pipeline:**  
   Execute the data director script located in `Scripts/Data_Pipe/` to run data cleaning, filtering, and feature engineering. You can choose to run the entire pipeline or only the modified stages.
   ```bash
    cd Scripts/Data_Pipe
    # Run the full data pipeline (clean -> filter -> feature)
    python data_director.py --stage all
   ```

3. **Run the Machine Learning Pipeline:**  
   Use the training script (for example, `base_director.py` in `Scripts/ML_Pipe/`) to train models, save artifacts, and generate prediction reports.
   ```bash
    cd ../../Scripts/ML_Pipe
    # Run the base director to start training and prediction generation
    python ml_director.py
   ```

4. **Launch the Dashboard:**  
   Run the dashboard script (`app.py`) to start the Streamlit application, which opens in your browser for interactive exploration and prediction.
   ```bash
    cd ../../
    streamlit run app.py
   ```

## Contributors

- **Team Gone Phishing (HackUSU):**
  - Annaka McClelland
  - Bracken Sant
  - Logan Ray
  - Tyler English
