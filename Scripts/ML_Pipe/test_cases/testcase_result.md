# Test Cases

## TabNetRegressor Num Pitches

### Results

#### Training Metrics per Epoch

| Epoch |   Loss  | Val_0_RMSE |   Time    |
|:-----:|:-------:|:----------:|:---------:|
| 0     | 5.21954 | 1.59332    | 0:00:01s |
| 1     | 1.35037 | 1.04760    | 0:00:03s |
| 2     | 0.82212 | 0.89682    | 0:00:05s |
| 3     | 0.76892 | 0.84848    | 0:00:07s |
| 4     | 0.72083 | 0.86697    | 0:00:08s |
| 5     | 0.71774 | 0.81945    | 0:00:10s |
| 6     | 0.66293 | 0.84573    | 0:00:12s |
| 7     | 0.66084 | 0.85495    | 0:00:13s |
| 8     | 0.65583 | 0.80163    | 0:00:15s |
| 9     | 0.65046 | 0.80322    | 0:00:17s |
| 10    | 0.63066 | 0.79384    | 0:00:18s |
| 11    | 0.62199 | 0.81318    | 0:00:20s |
| 12    | 0.60792 | 0.79767    | 0:00:22s |
| 13    | 0.62017 | 0.80916    | 0:00:23s |
| 14    | 0.57938 | 0.82291    | 0:00:25s |
| 15    | 0.57622 | 0.74923    | 0:00:27s |
| 16    | 0.56667 | 0.75331    | 0:00:28s |
| 17    | 0.56995 | 0.75477    | 0:00:30s |
| 18    | 0.53811 | 0.76835    | 0:00:32s |
| 19    | 0.57260 | 0.76263    | 0:00:33s |
| 20    | 0.54184 | 0.74713    | 0:00:35s |
| 21    | 0.53484 | 0.74929    | 0:00:37s |
| 22    | 0.51718 | 0.73803    | 0:00:38s |
| 23    | 0.51069 | 0.72303    | 0:00:40s |
| 24    | 0.49415 | 0.71785    | 0:00:41s |
| 25    | 0.49819 | 0.72442    | 0:00:43s |
| 26    | 0.49017 | 1.09474    | 0:00:45s |
| 27    | 0.49180 | 1.24811    | 0:00:46s |
| 28    | 0.47588 | 0.71052    | 0:00:48s |
| 29    | 0.48217 | 1.44880    | 0:00:50s |
| 30    | 0.49413 | **0.69054**| 0:00:51s |
| 31    | 0.46650 | 0.73037    | 0:00:53s |
| 32    | 0.46199 | 0.71578    | 0:00:55s |
| 33    | 0.46017 | 0.73879    | 0:00:56s |
| 34    | 0.45323 | 0.76336    | 0:00:58s |
| 35    | 0.45452 | 2.44407    | 0:00:59s |
| 36    | 0.46324 | 2.44623    | 0:01:01s |
| 37    | 0.45467 | 1.61233    | 0:01:03s |
| 38    | 0.45430 | 1.04815    | 0:01:04s |
| 39    | 0.45463 | 1.11796    | 0:01:06s |
| 40    | 0.46302 | 0.80728    | 0:01:08s |

**Early stopping** occurred at epoch **40** with:
- **Best Epoch:** 30
- **Best Validation RMSE:** 0.69054

Additional logs:
- *UserWarning: Best weights from best epoch are automatically used!*
- **Validation RMSE:** 0.6905
- **Predicted PitchofPA for Pitcher 1000066910.0 vs Batter 1000032366.0:** 2.91

### Explanation

The `TabNetRegressor` was trained for **40 epochs**, with early stopping selecting **epoch 30** as the best checkpoint. The model showed steady improvement during the initial epochs, reducing the `RMSE` from **1.59332** at epoch 0 to **0.69054** at epoch 30—the lowest validation RMSE achieved. However, subsequent epochs revealed significant fluctuations (e.g., spikes at epoch 29 with **1.44880** and epoch 36 with **2.44623**), suggesting potential overfitting or training instability.

Despite these fluctuations, the final selected model produced a validation `RMSE` of **0.6905**, indicating that further training is unlikely to yield improvements. The model's prediction for the given Pitcher and Batter resulted in a PitchofPA of **2.91**, which aligns with the trained performance. To further lower the RMSE, consider exploring options like additional feature engineering, hyperparameter tuning, data augmentation, or alternative models (e.g., XGBoost, LightGBM) to enhance generalization and reduce prediction error.



## Pycarrot Class

### Results

#### Summary
| Description                 | Value                                     |
|-----------------------------|-------------------------------------------|
| Session id                  | 42                                        |
| Target                      | PitchCall                                 |
| Target type                 | Multiclass                                |
| Target mapping              | BallCalled: 0, BallIntentional: 1, BallinDirt: ... |
| Original data shape         | (167081, 70)                              |
| Transformed data shape      | (167081, 132)                             |
| Transformed train set shape | (116956, 132)                             |
| Transformed test set shape  | (50125, 132)                              |
| Numeric features            | 54                                        |
| Categorical features        | 15                                        |
| Preprocess                  | True                                      |
| Imputation type             | simple                                    |
| Numeric imputation          | mean                                      |
| Categorical imputation      | mode                                      |
| Maximum one-hot encoding    | 25                                        |
| Encoding method             | None                                      |
| Fold Generator              | StratifiedKFold                           |
| Fold Number                 | 10                                        |
| CPU Jobs                    | -1                                        |
| Use GPU                     | False                                     |
| Log Experiment              | False                                     |
| Experiment Name             | clf-default-name                          |
| USI                         | 5b14                                      |

#### Model Performance
| Model                           | Accuracy | AUC  | Recall | Prec.  | F1     | Kappa  | MCC    | TT (Sec) |
|---------------------------------|----------|------|--------|--------|--------|--------|--------|----------|
| Ridge Classifier                | 0.6383   | 0.0  | 0.6383 | 0.6369 | 0.5676 | 0.4854 | 0.5501 | 0.801    |
| Extra Trees Classifier          | 0.3804   | 0.0  | 0.3804 | 0.6497 | 0.3053 | 0.2537 | 0.3685 | 1.532    |
| K Neighbors Classifier          | 0.3697   | 0.0  | 0.3697 | 0.3510 | 0.3226 | 0.1101 | 0.1225 | 2.596    |
| Random Forest Classifier        | 0.3659   | 0.0  | 0.3659 | 0.5997 | 0.3188 | 0.2488 | 0.3133 | 2.042    |
| Logistic Regression             | 0.3595   | 0.0  | 0.3595 | 0.1292 | 0.1901 | 0.0000 | 0.0000 | 9.409    |
| Dummy Classifier                | 0.3595   | 0.0  | 0.3595 | 0.1292 | 0.1901 | 0.0000 | 0.0000 | 0.749    |
| Light Gradient Boosting Machine | 0.2754   | 0.0  | 0.2754 | 0.4206 | 0.2581 | 0.1625 | 0.1880 | 2.571    |
| Naive Bayes                     | 0.2583   | 0.0  | 0.2583 | 0.1290 | 0.1711 | -0.0010| -0.0015| 1.022    |
| SVM - Linear Kernel             | 0.1925   | 0.0  | 0.1925 | 0.1090 | 0.1239 | 0.0007 | 0.0009 | 2.810    |
| Decision Tree Classifier        | 0.1759   | 0.0  | 0.1759 | 0.1856 | 0.1745 | 0.1490 | 0.3055 | 0.891    |
| Extreme Gradient Boosting       | 0.1759   | 0.0  | 0.1759 | 0.1744 | 0.1744 | 0.1490 | 0.3055 | 2.380    |
| Quadratic Discriminant Analysis | 0.1625   | 0.0  | 0.1625 | 0.3064 | 0.1613 | 0.1263 | 0.2035 | 1.034    |
| Ada Boost Classifier            | 0.1560   | 0.0  | 0.1560 | 0.0264 | 0.0451 | 0.0000 | 0.0000 | 4.052    |
| Gradient Boosting Classifier    | 0.0018   | 0.0  | 0.0018 | 0.0892 | 0.0006 | 0.0002 | 0.0063 | 125.729  |
| Linear Discriminant Analysis    | 0.0015   | 0.0  | 0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.240    |

#### Explanation
Takes forever for mediorcer results.

---

## Pycarrot Reg

### Results

#### Summary
| Description                 | Value                 |
|-----------------------------|-----------------------|
| Session id                  | 42                    |
| Target                      | PitchofPA             |
| Target type                 | Regression            |
| Original data shape         | (16979, 70)           |
| Transformed data shape      | (16979, 132)          |
| Transformed train set shape | (13583, 132)          |
| Transformed test set shape  | (3396, 132)           |
| Numeric features            | 55                    |
| Categorical features        | 14                    |
| Rows with missing values    | 100.0%                |
| Preprocess                  | True                  |
| Imputation type             | simple                |
| Numeric imputation          | mean                  |
| Categorical imputation      | mode                  |
| Maximum one-hot encoding    | 25                    |
| Encoding method             | None                  |
| Fold Generator              | KFold                 |
| Fold Number                 | 10                    |
| CPU Jobs                    | -1                    |
| Use GPU                     | False                 |
| Log Experiment              | False                 |
| Experiment Name             | reg-default-name      |
| USI                         | f374                  |

#### Model Performance
| Metric    | lr         | en         | omp        | lasso      | llar       | et         | ridge      | dummy      | lightgbm | gbr        | dt         | rf         | ada        | knn        | xgboost    | par        | br           | lar                |
|-----------|------------|------------|------------|------------|------------|------------|------------|------------|----------|------------|------------|------------|------------|------------|------------|------------|--------------|--------------------|
| RMSE      | 1.6292e+00 | 1.7150e+00 | 1.7616e+00 | 1.7624e+00 | 1.7624e+00 | 1.7882e+00 | 1.8314e+00 | 1.8456e+00 | 1.8491e+00 | 1.8492e+00 | 1.8492e+00 | 1.8492e+00 | 1.8492e+00 | 1.9474e+00 | 2.0465e+00 | 2.2375e+00 | 1.09693e+01  | 1.853328e+19       |
| R2        | 2.198e-01  | 1.362e-01  | 8.86e-02   | 8.78e-02   | 8.78e-02   | 6.07e-02   | 1.50e-02   | -4.00e-04  | -4.20e-03| -4.30e-03  | -4.30e-03  | -4.30e-03  | -4.30e-03  | -1.146e-01 | -2.303e-01 | -5.054e-01 | -3.60877e+01 | -1.013586e+39      |
| RMSLE     | 0.3109     | 0.3406     | 0.3484     | 0.3493     | 0.3493     | 0.3514     | 0.3625     | 0.3649     | 0.3685   | 0.3685     | 0.3685     | 0.3685     | 0.3685     | 0.3770     | 0.3754     | 0.4254     | 0.9214       | 11.1333            |
| MAPE      | 3.774e-01  | 4.360e-01  | 4.484e-01  | 4.505e-01  | 4.505e-01  | 4.499e-01  | 4.706e-01  | 4.747e-01  | 4.809e-01| 4.810e-01  | 4.810e-01  | 4.810e-01  | 4.810e-01  | 4.788e-01  | 4.266e-01  | 5.223e-01  | 2.1829e+00   | 8.382214e+16       |
| TT (Sec)  | 0.282      | 0.212      | 0.064      | 0.274      | 0.069      | 0.175      | 0.229      | 0.065      | 0.155    | 1.473      | 0.081      | 0.450      | 0.263      | 0.084      | 0.126      | 0.063      | 0.086       | 0.062              |

#### Explanations
Just takes too long for mediocre results.
