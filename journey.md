## Hackathon Journey
### Background
- Data from Pioneer Baseball League
- We were asked to make a Machine Learning Model to predict the best pitching sequence to strike out an opposing batter
### Summary of Machine Learning
- **Data Cleaning**: Cleaned the data by merging columns, resolving Nulls, filtering out data, and fixing data types
- **Feature Engineering**:
    - **Basic Feature Engineering**:
        - Grouping: Data is grouped by Date, PitchNo, PitcherId, and BatterId.
        - **Key Metrics:**
            - Avg_Pitch_Speed: Mean release speed (RelSpeed).
            - Release Angles & Spin: Average vertical (VertRelAngle) and horizontal (HorzRelAngle) release angles, spin rate, and spin axis.
            - Count-Based Rates: Strike and ball percentages from PitchCall.
            - Outcome Metrics: Total Outs Created and average plate location (PlateLocHeight/PlateLocSide).
        - **Purpose:** Provides foundational insights into pitch quality and release mechanics on a per-plate appearance basis.
    - **Advanced Feature Engineering**:"
        - Aggregation of Pitch Sequencing:"
            - Pitch_Type_Diversity & Sequencing Entropy: Measures the variety and unpredictability of pitch types using Shannon entropy.
            - Movement Ratios: Vertical vs. Horizontal break ratios and Pitch Zonal Targeting based on PlateLocHeight.
            - Velocity Metrics: Maximum Effective Velocity and Avg_Velocity_Drop.
            - Pitch Mix Ratios: Breaking Ball Ratio and Fastball-to-Offspeed Ratio.
            - Release Consistency: Deviation in release extension, reflecting mechanical consistency.
            - Avg_Hit_Exit_Velocity: Average exit speed of batted balls.
        - **Purpose:** Captures deeper strategic elements and pitch movement profiles that influence batter performance.
    - **More Advanced Feature Engineering**:
        - Enhanced Metrics with Smoothing:
            - Combining pitch-level, matchup-level, and batter-level data with Laplace smoothing (alpha=1, beta=2) to ensure robustness:
            - Pitch-Level: Includes Pitch_Type_Variance, Speed_Consistency, Change_in_Speed_Per_Pitch, and Pitcher_Aggressiveness.
            - Matchup-Level: Computes smoothed Hit_Probability and Strikeout_Likelihood for each pitcher-batter pair.
            - Batter-Level: Determines Batter_Strikeout_Tendency.
        - **Purpose:** Integrates individual pitch details with overall matchup trends to create a comprehensive feature set for predicting optimal pitch sequences.
- **Baseline Models Tested**: TabNetClassifier, PyCaretClassifier, TabNetRegressor, PyCaretRegressor, XGBoost
- **Findings**: TabNet performed best for classification, while XGBoost was slightly superior for regression.
- **Test Case Model**: PyCaretClassifier performed well with basic features but switching to TabNetClassifier yielded even better results.
  PyCaretRegressor underperformed on base features; TabNetRegressor showed slight improvement, but XGBoost emerged as the best regressor.
### Streamlit Implementation
- Built framework and filled in fields about our project
- Data exploration while creating graphs and looking for outliers.
- Made charts after data exploration and found good insights into the data.
- Made ML integration - Allows for file upload and insertion of data
