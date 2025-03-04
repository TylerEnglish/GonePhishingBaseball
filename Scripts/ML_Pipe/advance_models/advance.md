# Transformer Model Pipeline for Tabular Data

This document explains the structure and functionality of a Python file that implements a multi-level transformer model for tabular data. The file includes several components ranging from data preprocessing to model training, evaluation, and prediction.

## Overview

The file sets up an end-to-end pipeline for training a deep learning model on tabular data. It leverages popular libraries for data manipulation, machine learning, and deep learning. Key elements include utility functions for data cleaning and augmentation, a custom dataset class, a transformer-based model architecture, a training wrapper, cross-validation routines, and a prediction function.

## Table of Contents

- [Transformer Model Pipeline for Tabular Data](#transformer-model-pipeline-for-tabular-data)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Utility Functions](#utility-functions)
  - [Custom Dataset Class](#custom-dataset-class)
  - [Multi-Level Transformer Model](#multi-level-transformer-model)
  - [Training Wrapper](#training-wrapper)
  - [Cross-Validation Routine](#cross-validation-routine)
  - [Main Pipeline](#main-pipeline)
  - [Prediction Function](#prediction-function)
  - [How to Run](#how-to-run)
    - [Running the Pipeline](#running-the-pipeline)
    - [Using the Trained Model for Predictions](#using-the-trained-model-for-predictions)
    - [Running the Script](#running-the-script)
  - [Example Usage](#example-usage)
  - [Conclusion](#conclusion)

## Utility Functions

Several utility functions are provided to support the main training and evaluation tasks:

- **Computing Class Weights:**  
  A function computes weights for each class based on the distribution of the training labels. This is useful for handling class imbalance during training.

- **Advanced Data Cleaning:**  
  This function performs cleaning by imputing missing numerical values using the median and filling missing categorical values with a placeholder. It also ensures that rows missing the target value are removed.

- **Encoding Categorical Columns:**  
  A dedicated function converts categorical variables into numeric form using label encoding. It can also reuse existing encoders to transform new data consistently.

- **Feature Selection with Mutual Information:**  
  Mutual information is used to rank and optionally select a subset of features that are most informative for predicting the target variable.

- **Data Augmentation with Noise:**  
  Noise is added to the input data by scaling randomly generated Gaussian noise with the standard deviation of each feature. This technique can help prevent overfitting by making the model more robust to small variations in the data.

- **Model State Saving and Loading:**  
  Utility functions are provided for saving and loading the model state. The state is first saved to a temporary file and then compressed into a zip file, making it easier to manage model checkpoints.

## Custom Dataset Class

A custom dataset class is implemented to wrap the tabular data for use with a deep learning framework. This class converts the input data into tensors and makes them available via standard dataset indexing. It supports both supervised and unsupervised scenarios (with or without labels).

## Multi-Level Transformer Model

The core model is a multi-level transformer designed for tabular data. Key components include:

- **Categorical Embeddings:**  
  Each categorical feature is embedded into a vector space. The embeddings are then processed with a transformer encoder that treats each categorical feature as a token in a sequence.

- **Processing Numeric Features:**  
  A linear projection followed by a non-linearity and dropout is used to transform numeric features. These features are also passed through a separate transformer encoder.

- **Fusion Transformer Block:**  
  The model fuses the processed categorical and numeric features by concatenating their pooled representations. This concatenated vector is passed through a fusion layer and an additional transformer encoder to capture interactions between the different feature types.

- **Final Classification:**  
  A fully connected layer converts the final representation into logits for the target classes.

## Training Wrapper

A training wrapper class encapsulates the training process. This wrapper provides:

- **Feature Splitting:**  
  A helper method separates numeric from categorical features in the input data.

- **Training Routine:**  
  The training process iterates over batches of training data, performs forward and backward passes, and updates model parameters using an optimizer.

- **Evaluation Methods:**  
  Functions are provided to evaluate the model's performance using metrics such as accuracy, F1 score, and precision. These functions aggregate predictions over a validation set.

- **Prediction Methods:**  
  Separate methods are implemented for obtaining class predictions, probability estimates, and cumulative predictions (which update over a sequence).

- **Model Saving and Loading:**  
  The wrapper includes methods for saving the trained model state to disk and loading it back for later use.

## Cross-Validation Routine

The cross-validation function splits the data into several folds using stratified sampling. For each fold:

- Data is augmented with noise.
- Numeric features are scaled.
- The model is trained and evaluated.
- Metrics such as accuracy, F1 score, and precision are reported.

This approach helps ensure that the model’s performance is robust and not overly dependent on any single train–validation split.

## Main Pipeline

The main pipeline function orchestrates the complete workflow:

1. **Data Loading:**  
   The pipeline starts by reading a Parquet file containing the data.

2. **Data Cleaning and Encoding:**  
   Advanced cleaning is applied, and categorical features are encoded into numeric values.

3. **Feature Selection:**  
   Mutual information is used to optionally select the most informative features.

4. **Data Splitting and Scaling:**  
   The data is split into training and validation sets, and numeric features are standardized.

5. **Cross-Validation:**  
   The model is evaluated using a cross-validation routine before final training.

6. **Final Training:**  
   The model is retrained on the full training data, and its performance on a held-out validation set is reported.

7. **Model Saving:**  
   The trained model is saved for future inference.

The pipeline function returns a set of objects that includes the trained model, scaler, feature columns, encoders, and processed data.

## Prediction Function

A separate prediction function generates cumulative recommendations. It works by:

- Extracting data for a specific pitcher and batter.
- Generating candidate pitch types.
- Iteratively predicting the outcome for a sequence of pitches while updating a cumulative metric.
- Returning a summary of predictions and probabilities for each candidate pitch type.

This function is designed to support real-time recommendations based on the evolving state of a pitch sequence.

## How to Run

### Running the Pipeline

Create a Python script (for example, `run_pipeline.py`) with the following code:

```python
import os
from your_module_filename import main_transformer_pipeline 

# Set the path to your data file
data_path = "Derived_Data/feature/nDate_feature.parquet"

# Run the main pipeline function
pipeline_objs = main_transformer_pipeline(data_path=data_path)

# Check if the pipeline ran successfully
if pipeline_objs is not None:
    print("Pipeline executed successfully!")
else:
    print("Pipeline execution failed. Check data path and settings.")
```

### Using the Trained Model for Predictions

Once the pipeline runs and returns the necessary objects, you can use the trained model to generate predictions. Create a script (for example, predict.py) with the following code:

```python

import os
from your_module_filename import main_transformer_pipeline, prediction  

# Set the path to your data file and run the pipeline
data_path = "Derived_Data/feature/nDate_feature.parquet"
pipeline_objs = main_transformer_pipeline(data_path=data_path)

if pipeline_objs is not None:
    # Extract necessary objects from the pipeline output
    final_trainer = pipeline_objs["model_trainer"]
    scaler = pipeline_objs["scaler"]
    numeric_cols = pipeline_objs["numeric_cols"]
    cat_cols = pipeline_objs["cat_cols"]
    feature_cols = pipeline_objs["feature_cols"]
    df_proc = pipeline_objs["df_processed"]
    encoders = pipeline_objs["encoders"]
    target_col = "CleanPitchCall"

    # Define sample pitcher and batter IDs
    pitcher_id = 1000066910.0
    batter_id = 1000032366.0

    # Generate cumulative predictions for the specified pitcher and batter
    cumulative_results = prediction(
        pitcher=pitcher_id,
        batter=batter_id,
        model=final_trainer,
        scaler=scaler,
        encoders=encoders,
        df=df_proc,
        feature_cols=feature_cols,
        target_col=target_col,
        pitch_type_col="CleanPitchType",  # Adjust if needed
        n_pitches=3,
        alpha=0.5
    )

    # Display the prediction results
    if not cumulative_results.empty:
        print("Cumulative Prediction Results:")
        print(cumulative_results)
    else:
        print("No cumulative prediction data found.")
else:
    print("Pipeline execution failed. Check data path and settings.")


```

### Running the Script

```bash
python run_pipeline.py
python predict.py
```

## Example Usage

The file concludes with an example usage section that demonstrates how to run the main pipeline and then use the trained model for making predictions. In this section:

- A sample of processed data is used to generate predictions.
- The prediction function is called with a specific pitcher and batter to generate cumulative predictions.
- The results are printed and saved to a CSV file.

## Conclusion

This Python file provides a comprehensive pipeline for training a transformer-based model on tabular data. It includes robust preprocessing, feature engineering, model architecture, training routines, and prediction logic. The modular design allows for easy modifications and extensions, making it suitable for a range of classification tasks on structured datasets.
