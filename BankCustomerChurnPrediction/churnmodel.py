# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import numpy as np 
import pandas as pd
import numpy as np 
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder,LabelBinarizer
from sklearn.compose import ColumnTransformer,make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix, classification_report

from imblearn.pipeline import make_pipeline as imbl_pipe
from imblearn.over_sampling import SMOTE

#hyper-parameter tuning
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

#importing models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import plot_confusion_matrix
import seaborn





import pickle

loaded_model = pickle.load(open(r'/Users/iqrabismi/Desktop/machine_learning_files/deploy_model.sav', 'rb'))

input_data = (619.0, 'France','Female',42.0,2,0.00,1,1,101348.88 )

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('Customer will not churn')
else:
  print('Customer will churn')
  

