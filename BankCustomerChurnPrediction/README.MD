 # :credit_card: :dollar: Bank Customers Churn Prediction Model

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)


## Introduction 
__Banking Sector__ is a network or group of financial institutions that engages in providing banking
services to corporate and individual customers through direct or a third party client service. 
A bank is a financial institution licensed to receive deposits and make loans. 
Banks may also provide financial services such as wealth management, currency exchange. The banking sector
has become one of the main industries in developed countries. 
The technical progress and the increasing number of banks raised the level of competition among them. 
Banks are working hard to survive in this competitive market by implementing multiple strategies.
Churn is the measure of how many customers stop using the product or service. It can be based on actual usage or failure to renew.
Churn analysis involves analyzing historical customer data to make churn prediction possible.

  so that bank can take preventive measures  by giving certain benefits
such as offering disount, free membership etc to prevent
them from leaving. Detailed steps can be found in [Project Report.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/BankCustomerChurnPrediction/ProjectReport_Group6.pdf) 

## Challenges
One of the challenges that these banks face is to know the likelihood of customer attrition so that so that bank can 
take preventive measures in advance by giving certain benefits
such as offering disount, free membership etc to prevent
them from leaving.  __Customer churn__ has become a big issue in many banks because retaining existing customers costs less 
when compared the the cost spent to acquire a new customer.
Some of the bank customers in due course will stop their utilization or end their subscription this could be
because they switched to a competitor, no longer need the bank services, they are unhappy with their user experience, 
or they can no longer afford the cost.
Therefore, it becomes important to tackle this issues and understand which factors are responsible for customer attrition. 


## Data Science and Machine Learning 

With the help of __machine learning__, 
this problem could be addressed and possible churners in the bank can be identified. The data set used to built predictive model was taken from 
[Kaggle.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/BankCustomerChurnPrediction/Churn_Modelling.csv)

## Exploratory Data Analysis (EDA)

Visualised the distribution of churn rates within the dataset,
as well as examined the relationship between churn and various categorical variables (such as customer demographics or product usage). 
This was done using plots such as histograms, bar charts, or scatter plots.
__CRISP-DM__ Methodology was used to understand data and business problem, followed by data processing in which data quality report was prepared for both 
numeric and categorical variables. __Data Quality Report__ was used to identify any issues or 
problems that need to be addressed in order to ensure that the data is clean, consistent, and ready for analysis. 
However, this dataset didn't include any missing or incorrect values, duplicated records, or formatting issues. RowNumber and CustomerID was dropped as 
it was irrelevant in our analysis.

## Data Transformation 

After the data processing step, the dataset has to be transformed for the machine models 
to understand and to provide patterns that are easier to understand for the built models. 
This process involves converting non-numeric features into numeric. 
The categorical variables are encoded using __‘OneHotEncoder’__ that produces group of 
bits among which the combinations of values are only those with a single high (1) bit and all the others low (0). 
The column transformer does OneHotEncoder by passing the column numbers of the categorical variables. 
Also, drop parameter was set ‘first’ in onehotencoder to avoid __dummy variable trap__. 
The total instances with 10000 are encoded during the transformation step. 

## Metrics
Since this is a __classification__ problem, in which target variable is binary in nature. 
we consider following metrics that take into account __False Positives__ and __False Negative__. However, in this model, it is important to reduce False Negative
cases so that model can accurately detect a churn case.

* [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
* [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)
* [__F1_Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
* [__ROC-AUC Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
* [__ROC-Curve__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
* [__Accuracy Score__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [__Confusion Matrix__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

## Machine Learning Models

There were a large number of machine learning models used in the prediction of the __Churn Prediction__.
Below are the models that were used for prediction. Code for all these model can be found in [__Modelling Notebook__.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/BankCustomerChurnPrediction/Machine_Learning_Project_Modelling_Group6.ipynb)

* [__K Nearest Neighbors__](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* [__Decision Tree Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
* [__Gradient Boosting Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* [__Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [__Support Vector Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [__Gaussian Naive Baye's Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* [__Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [__XGBoost Classifier__](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
* [__Perceptron__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
* [__Ada Boost Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [__Ensemble Voting Classifer__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)

## Data Augmentation 
As there was class imbalance in the dataset i.e there were more number of non churn case as compared to churn case.
Hence, SMOTE(synthetic Minority Oversampling Technique) was used to make the target class balanced and there is no bias in the predictive model. 
It generates the synthetic records by linear interpolation methods for the minority class. For this [__Imblearn module__](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
was used.

## Hyper-Parameter Tuning 
To select the best parameters for each classifier Hyper-Parameter tuning was performed using __GridSearchCV__ with 5 Fold __Cross Validation__ technique. 

## Feature Importance
Following methods were implemented to see which features are most important and which features are least import
* [__Random Forest Feature Importance__](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
* [__SHAP__](https://shap.readthedocs.io/en/latest/)
* [__Two-Sample Independent T-test using Pingouin__](https://pingouin-stats.org/build/html/index.html)

From above analysis. it was oberseved that Age is most important feature responsible for Churn cases whereas HasCreditCard is least important feature. 

## Outcomes

The models that were able to perform the best for predicting the churn case are __Gradient Boosting Classifier__, __XGBoost Classifier__,
 __Random Forest Classifier__ and __Ada Boost Classifier__. These models were selected by calculating the score using cross validation method.
 Also, final ensemble voting classifier was implented using these four models and ROC AUC score 
 was highest for ensemble model(86%) but Recall was highest for Random Forest Model(61%). Also, Random Forest model had comparatively good AUC Score(85%).
 In churn prediction model, recall holds a higher importance as we want to accurately detect churn case. 
 In other words, we want to decrease the false negative so 
 that whenever any customer leaves the bank that effect can be detected accurately. 
 Therefore, for this reason random forest classifier was selected as 
 final model because its recall score is highest and f1 score is approximately close to ensemble model. This model was saved on local machine using [__Pickle__](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/BankCustomerChurnPrediction/deploy_model.sav). 
 
 ## Depolyment
 A web application using the best predictive model(RF Classifier) was created which can be used by bank to
 find out which customers are likely to leave the bank. This was done using __Streamlit and Spyder__. Detailed steps to create a web application can be
 found in [depolyment and evaluation notebook](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/BankCustomerChurnPrediction/Machine_Learning-Evaluation_Deployment_Group6.ipynb)
 Steps to create a .py file in [spyder](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/BankCustomerChurnPrediction/ChurnPredictiveModel.py) can be found here. 
 
 ## Future Scope

* Additional features such as __recent purchase date__ can be considered when training the model that can contribute on increasing the accuracy
* More number of customer records can be collected so that the amount of training data could also increase efficiency in a timely manner.

 
 
 
