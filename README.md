# Kickstarter Success Prediction

This project reproduces a study on predicting whether a Kickstarter campaign will succeed or fail using information available around launch. It also tests whether a text-derived feature from the project blurb helps, and compares XGBoost with LightGBM.

## Main result

On one 70/30 train/test split, LightGBM reached **80.8% accuracy** using metadata features and **82.1%** after adding a Naive Bayes feature derived from the blurb. XGBoost reached **82.3%** with the added text feature. The 0.2 percentage-point difference between XGBoost and LightGBM is small and was not tested across repeated splits, so it does not establish that either model is better.

## Data and method

The notebook collects monthly Kickstarter CSV snapshots from Web Robots covering May 2019 through April 2021. The May 2019 archive did not download, leaving 23 snapshots. After removing duplicate campaigns and retaining only successful and failed projects, the notebook reports **220,003 projects**. It uses a random 70/30 train/test split with seed 42: 154,002 training projects and 66,001 test projects.

Two feature sets are compared:

- **Sample 1 — metadata:** funding goal in USD, campaign duration, launch month and year, lengths of the project name and blurb, and encoded project, category, and location fields. It uses blurb length but not the words in the blurb.
- **Sample 2 — metadata + text:** Sample 1 plus one feature: a Naive Bayes estimate of success based on TF-IDF features from the blurb. The notebook creates training estimates out-of-fold to reduce leakage.

The models are LightGBM (listed as Gradient Boosting in several tables), Random Forest, a neural network, SVM, Logistic Regression, Ridge, OLS, and Lasso. XGBoost is added as an extra model. Class weighting is used for several classifiers.

## Results at a glance

| Feature set | Best accuracy among the paper's models | Best F1 among the paper's models | XGBoost accuracy |
|---|---:|---:|---:|
| Sample 1 — metadata | LightGBM: 0.808 | LightGBM / Neural Net: 0.832 | 0.811 |
| Sample 2 — metadata + blurb feature | LightGBM: 0.821 | Neural Net: 0.845 | 0.823 |

Adding the Naive Bayes text feature increased accuracy for all eight reproduced models by about 1.0 to 1.9 percentage points. XGBoost and LightGBM were nearly tied on Sample 2 accuracy and had the same F1 score (0.844).

## How to read the metrics

- **Accuracy (A):** the fraction of all test projects classified correctly.
- **F1:** combines precision and recall for the positive class, which is “successful.” It is useful when a single accuracy score does not show the balance between false alarms and missed successes.
- **Precision (P1 / P0):** among projects predicted successful / failed, the fraction that truly were successful / failed.
- **Recall (R1 / R0):** among truly successful / failed projects, the fraction predicted as successful / failed.
- **pp:** percentage points. For example, 82.3% minus 82.1% is a 0.2 percentage-point difference.

## Notebook graphs

The first four charts compare the notebook’s reproduced results with the paper. The next two rank the reproduced models, including XGBoost. The final chart compares LightGBM and XGBoost across several metrics.

### 1. Accuracy — Sample 1

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/4fdcf168-22ae-447e-ba56-fbdf789a3e26" />


Each pair of bars shows a model’s reported paper accuracy and reproduced accuracy with metadata features. LightGBM is close to the paper’s result (0.808 vs. 0.810). SVM and Ridge/OLS are lower; the adjusted Lasso result is higher. The vertical axis begins at 0.60, not zero, so the visual gaps are magnified.

### 2. Accuracy — Sample 2

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/69163f8d-2d89-4142-9824-a7beaf83ae8a" />


This uses metadata plus the Naive Bayes blurb feature. Among the original eight models, LightGBM has the highest reproduced accuracy at 0.821. The chart compares paper values with reproduced values; XGBoost is not included here.

### 3. F1 — Sample 1

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/f1e50a44-b47e-40a3-a394-8d741f039ab0" />


The bars compare F1 for the successful-project class. LightGBM and the neural network are tied at 0.832 in the reproduction. SVM and Ridge/OLS are notably below the paper’s reported F1 values. As with the accuracy chart, the vertical axis starts at 0.60.

### 4. F1 — Sample 2

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/522cd4dd-1859-41a6-bb1d-cd33fc453e37" />

With the blurb feature, the reproduced neural network has the highest F1 at 0.845, just above LightGBM at 0.844. Several models still fall short of the paper’s reported F1, even though their accuracy improved from Sample 1.

### 5. All models by accuracy — Sample 1

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/aeda5d2b-6262-4953-af1f-d3234fc3f899" />


This chart ranks all nine models, including XGBoost, by reproduced accuracy. XGBoost leads at 0.811, followed closely by LightGBM at 0.808 and the neural network at 0.805. Ridge and OLS are lowest at 0.734.

### 6. All models by accuracy — Sample 2

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/c20e6eb2-cb41-4d7f-8f02-546858b68ab7" />


With the text feature added, XGBoost reaches 0.823 and LightGBM 0.821. The neural network follows at 0.817. All models score higher than their Sample 1 accuracy, though this is one fixed train/test split.

### 7. LightGBM vs XGBoost — Sample 2 metrics

<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/10c44a6d-54cd-4eb9-995b-c2565ea886b0" />


The paired bars compare accuracy (A), F1, precision for success and failure (P1/P0), and recall for success and failure (R1/R0). The models are close across the board. XGBoost has slightly higher accuracy and success precision; LightGBM has slightly higher success recall. The plotted axis begins at 0.60, so small differences can look larger than they are.

## Limitations

- Only one random train/test split and seed were used; results may change on other splits.
- The May 2019 snapshot was unavailable, so the project count is smaller than the paper’s.
- The neural network used different settings from the paper and did not converge within 30 iterations.
- Lasso with the paper’s alpha of 0.75 predicted every project as failed in this reproduction. The reported Lasso results instead use alpha 0.001, so they are not a direct comparison.
- The results measure performance on this historical held-out set. They do not show that the model causes campaign success or guarantee performance on future campaigns.

## Running the notebook

The notebook imports NumPy, pandas, scikit-learn, LightGBM, XGBoost, Matplotlib, and `requests`. Install these packages in the Python environment used by Jupyter, then open `ML_MINIPROJECT.ipynb` and run cells from top to bottom. The download step needs internet access when the required archives are not already in `zips/`. The notebook reads and writes `zips/`, `data/`, CSV result files, and graph PNGs relative to the current working directory; start Jupyter from the project folder if you want the generated files there.

The notebook’s final cell uses `google.colab.files.download`, so that cell is specific to Google Colab. In a local Jupyter session, the result CSVs and charts are already written to the working directory by earlier cells; skip the final download cell.
