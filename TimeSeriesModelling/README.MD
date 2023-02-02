# :dollar: :chart_with_upwards_trend:  Amazon Stock Prediction using ARIMA and SARIMA Models

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)



## Introduction 
The prediction of __stock prices__ has always been a challenging task,
with numerous factors affecting the prices of stocks. The prediction of 
stock prices of one of the largest __e-commerce__ companies in the world, __Amazon__, 
is even more challenging, given its vast and diverse range of products and services. 
Despite the complexity of stock price prediction, it is a crucial aspect for investors 
and traders who are looking to make informed decisions about buying or selling stocks. 
In this context, __data science and machine learning__ have proven to be valuable tools for stock price prediction,
providing a comprehensive understanding of the underlying patterns in the data and enabling more accurate predictions.

## Challenges
There are several challenges in stock prediction, including:

Non-stationarity: Stock prices are often non-stationary, meaning that their statistical 
properties change over time. This makes it difficult to model and predict their behavior.

Volatility: Stock prices are often highly volatile, meaning that they can change rapidly and 
unpredictably. This makes it difficult to make accurate predictions about their future behavior.

Noise: Stock prices can be influenced by a variety of factors that are not easily predictable, 
including news events, economic indicators, and market sentiment. This noise can make it difficult
to distinguish meaningful patterns from random fluctuations in the data.

Complex relationships: There are many complex relationships between stocks, including relationships
between different industries, companies, and economic indicators. Modeling these relationships accurately can be difficult.

Modeling limitations: Stock prediction models are often limited by the quality and quantity of available 
data, the choice of model, and the assumptions made by the modeler. It is important to be aware of these limitations when making predictions.

Overall, stock prediction is a challenging task that requires a combination of statistical modeling techniques, 
domain knowledge, and a deep understanding of the financial markets.

## EDA

The data used for the prediction was obtained from Kaggle. 
Before implementing the prediction models, the data was thoroughly 
cleaned and preprocessed to ensure its quality. The data was checked 
for missing values and duplicates, which were removed, if any. Additionally, 
the data had some outliers that were detected and replaced with the median 
value of the stock to maintain the accuracy of the predictions.


## Data Science and Time Series Forecasting 
Data science and time series modeling can help in stock 
prediction by using statistical and machine learning algorithms 
to analyze past stock data, identify patterns and trends, and make predictions 
about future stock performance. Time series modeling is a powerful tool in stock 
prediction because it takes into account the sequential nature of stock data, allowing 
models to capture and analyze patterns in the data over time.

Some popular time series models used in stock prediction include ARIMA
(AutoRegressive Integrated Moving Average) and SARIMA (Seasonal AutoRegressive 
Integrated Moving Average) models. These models use mathematical equations to analyze 
the past trends and patterns in stock data and make predictions about future trends and 
patterns. __ARIMA__ models analyze univariate time series data, while __SARIMA__ models take
into account both the univariate time series data and the seasonal patterns in the data.


## ARIMA Model
The Auto-Regressive Integrated Moving Average 
(ARIMA) model is a statistical model that helps in 
understanding and predicting the behavior of a time series data.

Before using the ARIMA model, it is important to determine 
the order of the model, which is the number of lags in the
autoregression and moving average components. This is typically done 
using the Autocorrelation Function (ACF) and the Partial Autocorrelation 
Function (PACF) plots. The ACF and PACF plots help in identifying the patterns
and relationships in the data, which are then used to determine the order of the ARIMA model.

However, in order to use the ARIMA model effectively, 
the time series data must be stationary, meaning that the mean and 
variance of the data should be constant over time. If the data is not stationary,
it must be transformed to make it stationary. This was done by taking the
difference of the data. The stationarity of the data was  verified using the Dickey-Fuller test.

Once the data is stationary, the ARIMA model was fit to the data, and the model 
parameters can be estimated. After the model is fit, the last 20 days of data can be 
used to forecast the future 20 days. The ARIMA model helps in overcoming the challenges
of stock prediction by providing a structured approach to analyzing time series data and 
making predictions based on that analysis.

After the stock price predictions were made using the ARIMA model, the differenced values were transformed back to their original values.
More details are available [here.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/TimeSeriesModelling/TimeSeriesModelling.ipynb)

## SARIMA Model

a SARIMA (Seasonal Autoregressive Integrated Moving Average) model was 
implemented as the data showed a clear trend. The order of the SARIMA model 
was determined using the PMDARIMA technique. The PMDARIMA technique is an automated 
method that helps in finding the best hyperparameters for SARIMA models by passing in all 
required parameters. In this case, the order of the SARIMA model was identified by using PMDARIMA,
taking into account the trend and seasonality of the data. The SARIMA model was then fitted to the data 
and used to make predictions. This approach of using SARIMA models is 
useful in handling time-series data with trends and seasonality, and helps in achieving better predictions compared to other models.


## Prediction using Ridge Model

In order to make accurate predictions for the stock prices, 
various features were created using the past values of the stock 
at different time intervals. This was done because in time series analysis,
past values often serve as a useful indicator of future values. After creating 
the features, a ridge model was implemented 
which was cross-validated to assess its accuracy. The results showed that the 
model was highly accurate, with an accuracy of 99.5%.





