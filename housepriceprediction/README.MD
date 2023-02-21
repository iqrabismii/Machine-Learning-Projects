# :houses: :dollar: House Price Prediction
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Introduction
House price prediction is the process of forecasting the price of a house based on various factors such as location, size, number of rooms, and other features. This is an important task for buyers, sellers, and real estate agents who want to estimate the potential price of a house or property.

Predicting house prices can be done using various techniques such as statistical models, machine learning algorithms, and artificial neural networks. These techniques use historical data on house sales and other relevant factors to make predictions about future prices.

The accuracy of house price prediction models depends on the quality and quantity of data used, as well as the chosen model and its parameters. However, some common factors that are typically considered when predicting house prices include location, neighborhood, property age, house size, number of rooms, amenities, and local economic conditions.

House price prediction models are useful not only for buying and selling homes, but also for real estate investors, mortgage lenders, and government agencies. Accurate predictions can help these entities make informed decisions about property investments, lending policies, and economic development plans.



## Data Science and Machine Learning
Linear regression and polynomial regression are two popular techniques for house price prediction. Here's how each can help:

Linear regression:
Linear regression is a statistical method used to establish a relationship between a dependent variable (in this case, house price) and one or more independent variables (e.g. house size, number of rooms, location, etc.). In the context of house price prediction, a linear regression model can be used to estimate the impact of each independent variable on the house price. By analyzing historical sales data, the model can identify which factors have the most significant impact on the price of a house.

Polynomial regression:
Polynomial regression is an extension of linear regression that involves fitting a curve to the data instead of a straight line. This is useful when the relationship between the dependent variable and the independent variable is not linear but instead shows a more complex pattern. In the context of house price prediction, polynomial regression can help capture more complex relationships between house price and various factors, such as house size or the number of rooms. For example, it may reveal that the relationship between house size and price is not linear, but instead follows a quadratic or cubic pattern.

In summary, both linear and polynomial regression can be used to develop models that estimate house prices based on various factors. Linear regression can be a good starting point for simple models where the relationship between house price and the independent variables is approximately linear. Polynomial regression can be useful for capturing more complex relationships between house price and various factors, allowing for more accurate predictions.



## EDA 
 Dataset is available [here.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/housepriceprediction/data.xlsx)
 Some exploratory data analysis (EDA) on a dataset related to house prices. Here's a brief overview of the methods used:
 
 Pairplot:
A pairplot is a type of scatterplot matrix that allows you to visualize the relationship between pairs of variables in a dataset. In the context of house price prediction, you might use a pairplot to look for correlations between house price and other factors, such as house size, number of rooms, and location.

Spearman rank correlation:
Spearman rank correlation is a nonparametric method for measuring the strength and direction of the relationship between two variables. It works by ranking the values of each variable and then calculating the correlation coefficient based on the ranks. In the context of house price prediction, you might use Spearman rank correlation to look for correlations between house price and other factors, such as neighborhood quality or local economic conditions.

Skewness and kurtosis:
Skewness and kurtosis are measures of the shape of a distribution. Skewness measures the degree of asymmetry in the distribution, while kurtosis measures the degree of peakedness or flatness. In the context of house price prediction, you might use these measures to check the distribution of your data and ensure that it is approximately normal, which is a common assumption for many statistical models.

One-way ANOVA test:
One-way ANOVA (analysis of variance) is a statistical test used to compare the means of two or more groups. In the context of house price prediction, you might use a one-way ANOVA test to compare the mean house prices in different neighborhoods or regions.

Overall, these methods were useful for gaining insight into the relationships between variables in your dataset, identifying potential outliers or anomalies, and preparing  data for further analysis. By combining different EDA methods, helps in  more comprehensive understanding of the underlying patterns and structure in your data.
 
## Data Transformation 

Some data transformation on a dataset related to house prices. Here's a brief overview of the methods used:

Log transformation:
Log transformation is a common data transformation method used to normalize data that is not normally distributed. In the context of house price prediction, you might use a log transformation to transform the house price variable to make it more normally distributed. This can be useful for some statistical models, which assume a normal distribution of the data.

Minmax scaler:
MinMax scaler is a data normalization method that scales the values of a variable to a fixed range, typically between 0 and 1. In the context of house price prediction, you might use a MinMax scaler to normalize other variables such as house size or number of rooms, which might have different units or ranges of values. This can help to avoid biases in the analysis caused by differences in the scales of the variables.

Selecting relevant columns:
Selecting relevant columns is an important step in preparing data for analysis. In the context of house price prediction, you might select relevant columns that are likely to have a significant impact on house prices, such as house size, number of rooms, location, and neighborhood. By selecting only the most important columns, you can reduce noise in the data and improve the accuracy of your analysis.

Overall, data transformation and selection are important steps in preparing data for analysis in house price prediction. By transforming and normalizing the data, and selecting relevant columns, accuracy and efficiency of your analysis got improved and ensured that your models are robust and generalizable.

## Prediction
In the task of predicting house prices, the performance of different regression models was evaluated. The accuracy of the linear regression model was found to be 65%, while the polynomial regression model achieved an accuracy of 87%. The performance of a decision tree regressor was also evaluated, but the polynomial regression model was found to be the most effective in this task. Overall, the results suggest that the polynomial regression model was able to capture more complex relationships between the independent variables and the dependent variable, which led to more accurate predictions of house prices compared to the other models evaluated.
More details are available in this [notebook.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/housepriceprediction/data.xlsx) 





 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
