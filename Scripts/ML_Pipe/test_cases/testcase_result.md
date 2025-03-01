# Test Cases

## TabNetRegressor Num Pitches

### Results 

| Epoch  | Loss    | Val_0_RMSE | Time     |
|--------|--------|-----------|---------|
| 0      | 5.21954 | 1.59332   | 0:00:01s |
| 1      | 1.35037 | 1.04760   | 0:00:03s |
| 2      | 0.82212 | 0.89682   | 0:00:05s |
| 3      | 0.76892 | 0.84848   | 0:00:07s |
| 4      | 0.72083 | 0.86697   | 0:00:08s |
| 5      | 0.71774 | 0.81945   | 0:00:10s |
| 6      | 0.66293 | 0.84573   | 0:00:12s |
| 7      | 0.66084 | 0.85495   | 0:00:13s |
| 8      | 0.65583 | 0.80163   | 0:00:15s |
| 9      | 0.65046 | 0.80322   | 0:00:17s |
| 10     | 0.63066 | 0.79384   | 0:00:18s |
| 11     | 0.62199 | 0.81318   | 0:00:20s |
| 12     | 0.60792 | 0.79767   | 0:00:22s |
| 13     | 0.62017 | 0.80916   | 0:00:23s |
| 14     | 0.57938 | 0.82291   | 0:00:25s |
| 15     | 0.57622 | 0.74923   | 0:00:27s |
| 16     | 0.56667 | 0.75331   | 0:00:28s |
| 17     | 0.56995 | 0.75477   | 0:00:30s |
| 18     | 0.53811 | 0.76835   | 0:00:32s |
| 19     | 0.57260 | 0.76263   | 0:00:33s |
| 20     | 0.54184 | 0.74713   | 0:00:35s |
| 21     | 0.53484 | 0.74929   | 0:00:37s |
| 22     | 0.51718 | 0.73803   | 0:00:38s |
| 23     | 0.51069 | 0.72303   | 0:00:40s |
| 24     | 0.49415 | 0.71785   | 0:00:41s |
| 25     | 0.49819 | 0.72442   | 0:00:43s |
| 26     | 0.49017 | 1.09474   | 0:00:45s |
| 27     | 0.49180 | 1.24811   | 0:00:46s |
| 28     | 0.47588 | 0.71052   | 0:00:48s |
| 29     | 0.48217 | 1.44880   | 0:00:50s |
| 30     | 0.49413 | 0.69054   | 0:00:51s |
| 31     | 0.46650 | 0.73037   | 0:00:53s |
| 32     | 0.46199 | 0.71578   | 0:00:55s |
| 33     | 0.46017 | 0.73879   | 0:00:56s |
| 34     | 0.45323 | 0.76336   | 0:00:58s |
| 35     | 0.45452 | 2.44407   | 0:00:59s |
| 36     | 0.46324 | 2.44623   | 0:01:01s |
| 37     | 0.45467 | 1.61233   | 0:01:03s |
| 38     | 0.45430 | 1.04815   | 0:01:04s |
| 39     | 0.45463 | 1.11796   | 0:01:06s |
| 40     | 0.46302 | 0.80728   | 0:01:08s |


Early stopping occurred at epoch 40 with best_epoch = 30 and best_val_0_rmse = 0.69054
 UserWarning: Best weights from best epoch are automatically used!
Validation RMSE: 0.6905
Predicted PitchofPA for Pitcher 1000066910.0 vs Batter 1000032366.0: 2.91


### Explanation

The `TabNetRegressor` was trained for **40 epochs**, with early stopping selecting **epoch 30** as the best checkpoint. The model showed steady improvement in the initial epochs, reducing `RMSE` from `1.59332` at epoch 0 to `0.69054` at epoch 30, which became the lowest validation RMSE achieved. However, after this point, `RMSE` values fluctuated significantly, with extreme spikes such as `1.4488` at epoch 29 and `2.44623` at epoch 36, suggesting potential `overfitting` or `instability` in training.

Despite these fluctuations, the final selected model produced a validation `RMSE` of `0.6905`, which indicates a plateau where further training is **unlikely to yield improvements**. The prediction for Pitcher 1000066910.0 vs Batter 1000032366.0 resulted in a PitchofPA of **2.91**, aligning with the trained model’s performance. To lower RMSE further, improvements such as **feature engineering, hyperparameter tuning, data augmentation, or alternative models (e.g., XGBoost, LightGBM) should be explored** to enhance generalization and reduce prediction error.


## Pycarrot Class

### Results
 Description                                              Value
0                    Session id                                                 42
1                        Target                                          PitchCall
2                   Target type                                         Multiclass
3                Target mapping  BallCalled: 0, BallIntentional: 1, BallinDirt:...
4           Original data shape                                       (167081, 70)
5        Transformed data shape                                      (167081, 132)
6   Transformed train set shape                                      (116956, 132)
7    Transformed test set shape                                       (50125, 132)
8              Numeric features                                                 54
9          Categorical features                                                 15
10                   Preprocess                                               True
11              Imputation type                                             simple
12           Numeric imputation                                               mean
13       Categorical imputation                                               mode
14     Maximum one-hot encoding                                                 25
15              Encoding method                                               None
16               Fold Generator                                    StratifiedKFold
17                  Fold Number                                                 10
18                     CPU Jobs                                                 -1
19                      Use GPU                                              False
11              Imputation type                                             simple
12           Numeric imputation                                               mean
13       Categorical imputation                                               mode
14     Maximum one-hot encoding                                                 25
15              Encoding method                                               None
16               Fold Generator                                    StratifiedKFold
11              Imputation type                                             simple
12           Numeric imputation                                               mean
13       Categorical imputation                                               mode
11              Imputation type                                             simple
12           Numeric imputation                                               mean
13       Categorical imputation                                               mode
14     Maximum one-hot encoding                                                 25
15              Encoding method                                               None
16               Fold Generator                                    StratifiedKFold
17                  Fold Number                                                 10
18                     CPU Jobs                                                 -1
19                      Use GPU                                              False
20               Log Experiment                                              False
21              Experiment Name                                   clf-default-name
22                          USI                                               5b14



Model  Accuracy  AUC  Recall   Prec.  \
ridge                    Ridge Classifier    0.6383  0.0  0.6383  0.6369
et                 Extra Trees Classifier    0.3804  0.0  0.3804  0.6497
knn                K Neighbors Classifier    0.3697  0.0  0.3697  0.3510
rf               Random Forest Classifier    0.3659  0.0  0.3659  0.5997
lr                    Logistic Regression    0.3595  0.0  0.3595  0.1292
dummy                    Dummy Classifier    0.3595  0.0  0.3595  0.1292
lightgbm  Light Gradient Boosting Machine    0.2754  0.0  0.2754  0.4206
nb                            Naive Bayes    0.2583  0.0  0.2583  0.1290
svm                   SVM - Linear Kernel    0.1925  0.0  0.1925  0.1090
dt               Decision Tree Classifier    0.1759  0.0  0.1759  0.1856
xgboost         Extreme Gradient Boosting    0.1759  0.0  0.1759  0.1744
qda       Quadratic Discriminant Analysis    0.1625  0.0  0.1625  0.3064
ada                  Ada Boost Classifier    0.1560  0.0  0.1560  0.0264
gbc          Gradient Boosting Classifier    0.0018  0.0  0.0018  0.0892
lda          Linear Discriminant Analysis    0.0015  0.0  0.0015  0.0000

              F1   Kappa     MCC  TT (Sec)
ridge     0.5676  0.4854  0.5501     0.801
et        0.3053  0.2537  0.3685     1.532
knn       0.3226  0.1101  0.1225     2.596
rf        0.3188  0.2488  0.3133     2.042
lr        0.1901  0.0000  0.0000     9.409
dummy     0.1901  0.0000  0.0000     0.749
lightgbm  0.2581  0.1625  0.1880     2.571
nb        0.1711 -0.0010 -0.0015     1.022
svm       0.1239  0.0007  0.0009     2.810
dt        0.1745  0.1490  0.3055     0.891
xgboost   0.1744  0.1490  0.3055     2.380
qda       0.1613  0.1263  0.2035     1.034
ada       0.0451  0.0000  0.0000     4.052
gbc       0.0006  0.0002  0.0063   125.729
lda       0.0000  0.0000  0.0000     1.240

### Explanation
Takes forever for mediorcer results.


## Pycarrot Reg

### Results

                    Description             Value
0                    Session id                42
1                        Target         PitchofPA
2                   Target type        Regression
3           Original data shape       (16979, 70)
4        Transformed data shape      (16979, 132)
5   Transformed train set shape      (13583, 132)
6    Transformed test set shape       (3396, 132)
7              Numeric features                55
8          Categorical features                14
9      Rows with missing values            100.0%
10                   Preprocess              True
11              Imputation type            simple
12           Numeric imputation              mean
13       Categorical imputation              mode
14     Maximum one-hot encoding                25
15              Encoding method              None
16               Fold Generator             KFold
17                  Fold Number                10
18                     CPU Jobs                -1
19                      Use GPU             False
20               Log Experiment             False
21              Experiment Name  reg-default-name
22                          USI              f374

               RMSE            R2       RMSLE          MAPE  TT (Sec)
lr        1.629200e+00  2.198000e-01   0.3109  3.774000e-01     0.282
en        1.715000e+00  1.362000e-01   0.3406  4.360000e-01     0.212
omp       1.761600e+00  8.860000e-02   0.3484  4.484000e-01     0.064
lasso     1.762400e+00  8.780000e-02   0.3493  4.505000e-01     0.274
llar      1.762400e+00  8.780000e-02   0.3493  4.505000e-01     0.069
et        1.788200e+00  6.070000e-02   0.3514  4.499000e-01     0.175
ridge     1.831400e+00  1.500000e-02   0.3625  4.706000e-01     0.229
dummy     1.845600e+00 -4.000000e-04   0.3649  4.747000e-01     0.065
lightgbm  1.849100e+00 -4.200000e-03   0.3685  4.809000e-01     0.155
gbr       1.849200e+00 -4.300000e-03   0.3685  4.810000e-01     1.473
dt        1.849200e+00 -4.300000e-03   0.3685  4.810000e-01     0.081
rf        1.849200e+00 -4.300000e-03   0.3685  4.810000e-01     0.450
ada       1.849200e+00 -4.300000e-03   0.3685  4.810000e-01     0.263
knn       1.947400e+00 -1.146000e-01   0.3770  4.788000e-01     0.084
xgboost   2.046500e+00 -2.303000e-01   0.3754  4.266000e-01     0.126
par       2.237500e+00 -5.054000e-01   0.4254  5.223000e-01     0.063
br        1.096930e+01 -3.608770e+01   0.9214  2.182900e+00     0.086
lar       1.853328e+19 -1.013586e+39  11.1333  8.382214e+16     0.062

### Explanations

Just takes to long for mediorce results.