# Arya.ai assignment

## Introduction 

This is the assignment solution for the datascience role at Arya.ai. I have attempted a binary classification problem given the data, and have attempted feature selection, training (with validation) and presented the predictions

## Data

The data has been visualized in these ways:
- Normally; no transforms on the data, pick just 3 dimensions
- PCA: The standardized data is reduced to 3 dimensions and visualized
- T-SNE plot: The data is projected into 3-D and visualized. Perplexity had a minimal impact on the visualization.  

The classes are imbalanced (n(0):n(1) correspond to approximately 60:40), so the data has been balanced using SMOTE, a synthetic oversampling technique

## Feature Selection  

The feature selection has been performed in 3 ways
- Manually: Checking and eliminating sparse features
- PCA: Finding the number of features for the explained variance to be 95% 
    - This means that the features account for 95% of variance in data, can prevent overfitting, and are good enough for our model training
- Intrinsic selection: L1-Regularization has been imposed for LogisticRegression (can deal with sparse features), and Bagging and max-depth set for Random Forest.

## Model training
- 3 models have been trained: LogisticRegression, SVM, RandomForest
- Validation split (train:test = 80:20 or 4:1)
- Grid Search has been performed and the parameters have been considered whenever they show a decent improvement
- The best model found during the runs (Support Vector Classifier) has been saved

## Scripts and notebooks

### Create a venv

`python -m venv /path/to/venv/`   

On Windows: `/path/to/venv/Scripts/activate`  
On Linux: `source /path/to/venv/bin/activate`  

### Install dependencies

`pip install -r requirements.txt`  

###  Deliverables  

- `data_fit.ipynb` for the feature selection and model training (run all cells for outputs)
- `validate.py` for validation accuracy (run `python validate.py`)
- `data_pred.py` for the `predictions.csv` file (run `python data_pred.py`)


## Other data
- `training_set.csv` for the training data provided
- `test_set.csv` for the test set provided
- This README.md for descriptions of the project
-  `predictions.csv` for the predictions WITH the data
- `model.pkl` for the saved model and pca fit
- `requirements.txt` for the libraries to be installed 


