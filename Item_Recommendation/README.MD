# :woman_teacher: :dollar: :chart_with_upwards_trend: Recency, Frequency, Monetary Analysis and Customer Segmentation

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)




## Introduction 
__RFM (Recency, Frequency, Monetary) analysis__ is a __Customer Segmentation__ technique that can be used by 
retailers and grocery stores to identify their most valuable customers and to target their marketing efforts. 
It works by dividing customers into segments based on their past purchase behavior.



## Challenges
There are several challenges that businesses may face when trying to upsell or cross-sell products to their customers:

Identifying the right customers: It is important to target the right customers with upselling and cross-selling efforts, as trying to sell higher-value products or additional products and services to customers who are not interested or cannot afford them is likely to be unsuccessful.

Finding the right products or services: It is important to identify products or services that are a good fit for a customer's needs and interests, as customers are more likely to make additional purchases if they feel that the products or services being offered are relevant to them.

Maintaining customer trust: Upselling and cross-selling can be perceived as pushy or aggressive if not done in a respectful and transparent way, which can lead to a loss of customer trust.

Balancing the customer experience: Upselling and cross-selling efforts should not detract from the overall customer experience, as a negative experience can lead to a loss of customer loyalty.

RFM (Recency, Frequency, Monetary) analysis can help businesses overcome these challenges by providing insight into a customer's past purchase behavior, which can be used to identify the right customers to target with upselling and cross-selling efforts. By segmenting customers based on their recency, frequency, and monetary value, businesses can identify customers who are most likely to be interested in higher-value products or additional products and services. RFM analysis can also be used to optimize the timing of upselling and cross-selling efforts, helping to balance the customer experience and maintain customer trust.



## Exploratory Data Analysis (EDA)
This dataset consisted of 54k records and was taken from [Kaggle.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/Item_Recommendation/Online_Retail.xlsx)
__Data Preprocessing__ was done to remove nulls, duplicate values. There were no missing value in the dataset.However, there were around
5k duplicated which was dropped. UK has the highest number of shoppers as 
compared to other countries. Also, light bag,cake stand were the most sold products. 
Details about EDA can be found in this [notebook.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/MarketBasketAnalysis/MarketBasketAnalysis.ipynb)
Also, for UK Apriori algorithm was implemented and it was observed that mostly different types of cups are bought and this behaviour aligned with their rich and royal culture. 


## Recency, Frequency and Monetary(RFM) Analysis
Most recent, frequent and highest spenders was analysed using Pandas. In order to segment customer customers on the basis of thier, a new dataframe was 
created by merging all RFM values. As values differed in their magnititude, SKlearn [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
was implemeted to scale the values in the range 0 to +1. Also, 
outliers were clamped that were three deviations away from the mean. 

## K-Means Clustering and Silhouette Analysis

[__K-means Clustering__](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
is an unsupervised machine learning technique that was used 
to segment a customer base on the basis of RFM (Recency, Frequency, Monetary) values.
The technique works by dividing a dataset into a specified number of clusters, or groups,
based on the similarity of the data points within each cluster.
[__Silhouette Analysis__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
was used to evaluate the quality of the clusters produced by k-means clustering.
It works by calculating a silhouette score for each data point within a cluster, which is a measure of how 
similar the data point is to other data points within the same cluster compared to data points in other clusters.
The silhouette score ranges from -1 to 1, with higher values indicating a better fit within the cluster.
Details can be found in this [notebook.]
Silhouette Score was good when number of cluster was set 3. Hence, customers were segmented in three clusters. Lastly, scatter plot was drawn to 
visualise cluster wrt to recency, frequency and monetary. 


## Outcome
With the help of scatter plot, customers can be futher segmented into Best customer, High-spending new customers, Lowest-spending active loyal customers,
Churned Best Customers. Accordingly, the organisation can clearly identify groups  to cross sell, upsell their products. 

1. Best Customers – Frequent and recent shoppers with heavy spending. Potential customers for upselling. 
2. High-spending New Customers – Recent shoppers with low frequency and spending. Emphasize CRM to enhance shopping experience to increase engagement.
3. Lowest-Spending Active Loyal Customers – These repeat customers are active and loyal, but they are low spenders.
Marketers should create campaigns for this group that make them feel valued, and incentivize them to increase their
spend levels. 
4. Churned Best Customers – These are valuable customers who stopped transacting a long time ago. While it’s often 
challenging to re-engage churned customers, the high value of these customers makes it worthwhile trying. 

