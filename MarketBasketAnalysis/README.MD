 # :green_salad: :cookie: Market Basket Analysis for Grocery Store

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org) 
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org)
[![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) 
[![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)



## Introduction 
__Market Basket Analysis__ is used for __Cross Selling__ certain products/items. It is a 
technique used in __Data Mining__ to identify relationships between items in a large dataset of transactions. 
In the context of a grocery store, market basket analysis might
be used to identify which products are most commonly purchased together, such as milk and bread, or vegetable and rolls/buns.
This information can be used to create targeted promotions or to place related items near each other in the store to encourage additional sales.

## Challenges
Cross-selling refers to the practice of selling additional products or services to a 
customer who has already made a purchase. It can be a challenging task for retailers, as they need to identify the 
right products to offer to a particular customer and find a way to present those products in a way that is relevant and appealing.

Market basket analysis can help retailers overcome some of the challenges of cross-selling 
by providing insights into the products that are commonly purchased together. For example, 
if a customer has just purchased a grill, a retailer could use market basket analysis to identify
other products that are commonly purchased by customers who have bought grills, such as BBQ sauce or charcoal. 
This information can help the retailer create targeted recommendations for the customer and increase the chances of making additional sales.

Additionally, market basket analysis can help retailers understand the preferences of their customers and 
create more personalized recommendations. For example, if a customer has previously purchased vegetarian-friendly 
products, a retailer could use this information to recommend vegetarian options to the customer in the future.


## Apriori and Association Algorithm 


With the help of __Apriori and Association Algorithm__ , retailers can identify pattern in customer shopping behaivour i.e. which products are frequently 
purchased together. They are used to identify relationships between items in a large dataset of transactions.
Apriori is an algorithm that is used to identify frequent item sets in a dataset. It works by first identifying all of 
the single items that are present in the dataset, and then progressively building up to larger and 
larger item sets, only considering those that meet a minimum support threshold. For example, 
if the minimum support threshold is set at 50%, this means that an item set must be present in at 
least 50% of the transactions in the dataset in order to be considered a frequent item set.

Association rule learning is a technique used to identify relationships between variables in a dataset. 
It works by identifying the variables that are most frequently present in the same transactions and then 
expressing those relationships as "rules" in the form of "if this, then that". For example, an association
rule might be "if a customer purchases bread, they are also likely to purchase butter".

Both Apriori and association rule learning can be useful tools for market basket analysis, as they allow 
retailers to identify patterns and relationships in customer purchasing behavior that can inform marketing and sales efforts.


## Exploratory Data Analysis (EDA)
This dataset consisted of 34k records and was taken from [Kaggle.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/MarketBasketAnalysis/Groceriesdata.csv)
__Data Preprocessing__ was done to remove nulls, duplicate values. There were no missing value in the dataset. However, there were around 739 duplicates. 
Visualised shopping trend by plotting line graph wrt to date and number of items bought in a particulary day. Whole Milk, Vegetables, 
rolls and buns were the most sold products. Details about EDA can be found in this [notebook.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/MarketBasketAnalysis/MarketBasketAnalysis.ipynb)


## Data Transformation 

Identified the total number of rules by first creating a list for transaction wrt to memberid and transaction done on particular day. After this 
itertools permuations was leverage to create number of rules. There were total 167 unique items. Out of which 38006 transactions was done wrt to member and date. 
Also, numeber of total rules was 27722.
[__TransactionEncoder__](http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/)
was used to transform the data in appropriate format.
A transaction encoder is a tool that converts a dataset into a transaction dataset, which is a specific format that is used as input for the Apriori algorithm and other tools for market basket analysis.

Transaction datasets are used to represent transactions, where each row represents a single transaction and each column represents an item that was purchased in that transaction. The values in the cells of the transaction dataset are typically either 0 or 1, with a value of 1 indicating that the item was purchased in the corresponding transaction, and a value of 0 indicating that it was not.

A transaction encoder takes a dataset as input and converts it into a transaction 
dataset by identifying the unique items present in the dataset and encoding them as columns. 
For example, if a dataset includes a list of customer orders, with each row representing a single 
order and each column representing a different item, the transaction encoder would create a new dataset
where each row represents a single transaction and each column represents a different item. The values in the cells of the new dataset 
would be 1 if the item was purchased in the corresponding transaction, and 0 if it was not.


## Metrics used for Pruning
Following metrics was used to pruning
* [__Support__](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/)
* [__Confidence__](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/)
* [__Lift__](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/)
* [__Leverage__](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/)
* [__Zhang's Metric__](http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns/)

Details and calculation involved in each metric can be view in 
[notebook.](https://github.com/iqrabismii/Machine-Learning-Projects/blob/main/MarketBasketAnalysis/MarketBasketAnalysis.ipynb)


## Mxltend Library for Apriori and Association 
Mlxtend is a library of Python tools for data science 
and machine learning, and it includes implementations of the [__Apriori algorithm and association rule__](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
learning.Some of the key parameters include:

min_support: This parameter determines the minimum support threshold for identifying frequent item sets. It can be specified as a percentage or as a absolute value, depending on the format of the input data.

max_len: This parameter specifies the maximum length of the item sets that will be considered by the Apriori algorithm.

use_colnames: This parameter determines whether the column names of the input dataset will be used as the names of the items in the frequent item sets.

items_to_consider: This parameter allows you to specify a list of items that will be considered by the Apriori algorithm. If set, only item sets that contain one or more of the specified items will be considered.


## Visualising Rules
HeatMap was plotted to visualise the intensity of support between antecendent and consequent. To perform this, a pivot table was created with  
consequents as columns and index with antencendents. It was observed there was strong association 
1. other vegetable <-- whole milk
2. rolls/buns <-- whole milk 
Also, scatter plot was plotted to see min and max values for support, lift and relation between them. 
Lastly [__Parellel Coordinates Plot__](https://plotly.com/python/parallel-coordinates-plot/)
was plotted to visualise the rules between antencedent and consequent. 
It was observed that milk is bought with beer and citrus fruits and sausages are bought with other vegetables. 


 
 
 
