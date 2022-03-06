# Mod12_challenge - Evaluating Credit Risk

# Overview
This analysis uses a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. It leverages logistic regression to predict the loan default. The analysis evaluates original data (with imbalanced data due to fewer defaults than current loans) and resampled data to balance the default component of the data. For both dataset, the analysis measures accuracy and generates confusion matrices.

# Data
This analysis is based on the dataset 'leanding_data.csv' which can be found at the following location: Resources/lending_data.csv

# Imports and libraries
This analysis was done using Jupyter Labs and requires the following libraries:

'''
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import balanced_accuracy_score

import warnings
warnings.filterwarnings('ignore')
'''

# Output

# Section 1: Original Data

To test the models the data is divided into test data and training data.

A logistic regression model is fit using the training data.

Using the testing data, predictions are made on the model that was created by the training data.

The quality of the model is evalauted using an accuracy score, a confusion matrix and a classification report. 

# Section 2: Resampled Data
The steps from section 1 are repeated but using a new version of the data.

The data is resampled using the RandomSampler function to even out the imbalanced data 
(impbalance exists because there are fewer defaulted loans than current loans)




# Created by
Created by Ann Howell with help from the Rice University FinTech Bootcamp

# License
MIT