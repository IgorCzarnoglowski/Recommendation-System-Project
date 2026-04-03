# Recommendation System Project

## This is a project aimed to use multiple models in order to build tool for recommending products from any datatset.

This project is for learning purposes, to grasp the idea of build such recommendation system. It is coded by me, but I used many sources to take inspirations from, such as kaggle notebooks and stackoverflow. It is built with a help of LLM Claude.

In its origin, project was supposed to help me understand implicit library, but it developed into using multiple machine learning models, such as:
* Item-based KNN
* Popularity-Based Recommendation
* Basket-Based Recommendation
* Content-Based Recommendation
* Implicit ALS

## Approach

At this point only ALS model is done, in its approach we are focusing on using events csv, where we have information about
users, products and actions (events) done on items by clients. We assign weights to each action
* View - 1
* Add to cart - 3
* Transaction (buy) - 5

We assign them to be able to use them later when training model on multiple actions. We build matrix which is size of
```amount of users x amount of products``` and the values are ```weights```. Then the matrix is passed on to ALS model.
Later in evaluation process, as a ground truth we use only transaction events, because we are trying to predict "what customer will buy".
Since transaction has the biggest weight it obviously has the biggest impact.

## Results for Implicit ALS
Without filtering out users and products with few interactions

| Metric       | Value  |
|--------------|--------|
| Precision@10 | 0.005128205128205129 |
| Recall@10    | 0.042735042735042736 |

With filtering out users who have less than 3 interactions and products which have less than 5 interactions

| Metric       | Value  |
|--------------|--------|
| Precision@10 | 0.005309734513274337 |
| Recall@10    | 0.04424778761061947 |

We can clearly see that metrics are slightly better after removing abnormal activity, it is not a HUGE difference, but that's normal, since dataset is huge and even slight change is relevant.

## What I learned

I believe that key takeaways from this project is "power" of ctx matrix, ways to preprocess your data, the idea of recommendation systems and
problems that awaits, such as cold start for new products or sparsity of data. Also, I learned more about ALS method and math behind it.
I feel like, that I have some idea of how recommendation systems look like in big e-commerce shops.

## Dataset

Download the RetailRocket dataset from Kaggle:
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

## How to install and run this project

1. Clone the repo
2. Download dataset and place the files in the `data/` folder
3. Make sure to have installed at least Python 3.10, preferably use conda
4. In your project directory run command ```pip install -r requirements.txt```
5. To run project, all you need is run command ```python main.py```

