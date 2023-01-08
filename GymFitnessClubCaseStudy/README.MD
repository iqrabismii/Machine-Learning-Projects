# :running_man: :weight_lifting_woman: : Attendance Prediction for Members of Gym Fitness Club

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)



## Introduction
This is a DataCamp Case Study. Detail about this case study and steps taken to implement machine learning  is given below. Link for DataCamp workspace can
be found [here.](https://app.datacamp.com/workspace/w/30507c3b-7079-44e8-9481-676a3631ae16/edit?file=notebook.ipynb)

#### Company Background
GoalZone is a fitness club chain providing five types of fitness classes in Canada.
Finally, the fitness classes schedule is back to normal after the COVID-19 restrictions are lifted.
However, they have received many complaints from the customers about having a hard time booking a fitness class.

From initial analysis, the program operation team found out that the fitness classes are always fully booked 
but are having a low attendance rate per class. To improve this situation, they would like to increase the class
capacity for customers to sign up if a low attendance rate is predicted.



## Challenges

The operation manager has asked  to predict the attendance rate for each fitness class?
  
## Data Preprocessing
The data you will use for this analysis can be accessed here: "data/fitness_class.csv"
![Screenshot 2023-01-07 at 5 33 44 PM](https://user-images.githubusercontent.com/108056063/211176798-9c5c0813-65b1-4d14-a8c4-146ae14b4a33.png)
 
The feature named 'attendance' is the target feature in this data.
Data types for all features are correct as per the crieterion mentioned. 
For class category the data type was initial object but that has been changed 
to category type using astype. Also for age, where age was less than 14 those 
instances have been dropped. Likewise, for class capacity any instance with capacity greater
than 25 has been changed to 25. I also validated the data with respect null values and there was 
no missing value in the data. Lastly, five point summary for continous features is calculated and 
for categorical features mode is calculated. Also, checked the cardinality for numeric features 
and in this cardinality for class capacity was only two. So this was converted  into category.

## Exploratory Data Analysis (EDA)
1. from the histogram above we can see that days before is not normally distributed. Infact it seems to be right skewed. Rest all features are normally distributed.
2. from the pair plot above and correlation coefficient above we can see that attendance is negatively 
related with age with correlation coefficient -0.734. Also, it is linearly correlated with new students and
over 6 months. with respect to multicollinearity, there is no such feature. A linear relationship was observed between over 6months and new students but from correlation table, they are not so related.
3.with respect to box plot above, we cna see that there is relationship between attendance and class_Capacity. with respect to class category the median values are almost similar and same goes for day_of_week also. To find that if there is any relationship between attendance and these two category we will also cross check using pairwise anova test.where;

H0: there is no difference in attendance wrt to different groups in categorical feature  <br>
Ha: there is difference in attendance wrt different groups <br>
significance level is set at : 0.05 <br>
After conducting the anova test for these two features we can see none of the p-value are less than significance level. from there we can say that we fail to reject the null hypothesis. That is there no such significant difference between two groups. Hence in modelling, we will not be considering day_of_week and class_category features

## Machine Learning
1. Descriptive Feature array X was defined by converting all relevant and numeric features into array. Likewise, y was defined by converting target feature into array
2. Split the train and test data using sklearn train_test_split
3. Also, scale the data as age had comparitively higher magnitude as compared to other features
4. First linear regression was implemented which gave 76% accuracy.
5. Decision tree gave less accuracy around 60% which was less.
6.Followed by gradient boosting which gave 76% accuracy. 7. Implemented gridsearch cv for Ridge regularisation and the accuracy was in the range 74-76% for alplha= 0.1 8. Implemented emsemble model which gave 76% , again similar to previous used models 9. To be sure on accuracy , I also performed 5 fold cross validation for different models to get the accurate results and the accuracy was around 74-76% only.

Hence, after implementing various models I have decided to select ridge regression with parameters('alpha': 0.01, 'fit_intercept': True, 'solver': 'saga') as it gives accurate results and also prevent model from overfitting. Also, by doing hyper parameter tuning (using GridSearchCv) we can get best parameters for the model.

## Evaluation 
1. for this analysis , i have selected coefficient of determination R2, mean squared error and mean absolute error. I have chosen this because for regression models mean squared error tell us about difference between actual and predicted values.the lesser the value of rmse is , the more efficient the model is. It tell us about the residuals.
2. By using ridge with best parameters and cross validation score,accuracy got increased to 77%.
3. hence, with this model the values can be predicted by the model approximately 77% and these values can help the concerned department to increase the capacity of the gym whenever low attendance is predicted. 
4. Also, as the rmse is a bit high in this model,so the better approach would be to define a threshold ie if the attendance is near or below that threshold then the office should increase the capacity so that more people can register. as the threshold is not given in the instructions manual. So, I have implemented threshold when the attendance is less than class capacity ie. below 12 then it should be consider low. This threshold can be modified after discussing with the gym people.
Further details about this cases study can be found here in the [notebook](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/GymFitnessClubCaseStudy/MLModels.ipynb).
