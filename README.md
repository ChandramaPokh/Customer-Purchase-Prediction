# Customer Purchase Prediction

## Objective

The objective of this project is to build a Machine Learning model that predicts whether a customer will purchase a product based on their Age, Income, and Spending Score.

## Dataset

The dataset contains information about 20 customers.

The columns are:

- Age
- Income
- SpendingScore
- Purchased

### Target Variable

The `Purchased` column is the target variable.

- `0` = Not Purchased
- `1` = Purchased

## Features

The following features are used to make predictions:

- Age
- Income
- SpendingScore

## Machine Learning Algorithm

Logistic Regression is used to predict whether a customer will purchase the product.

## Libraries Used

- Pandas
- Matplotlib
- Scikit-learn

## Project Steps

1. Created the customer dataset using Pandas.
2. Explored the dataset using `head()`, `shape`, and `isnull()`.
3. Visualized Income vs Spending Score.
4. Selected Age, Income, and SpendingScore as features.
5. Selected Purchased as the target.
6. Split the data into training and testing sets.
7. Trained a Logistic Regression model.
8. Made predictions.
9. Calculated the accuracy.
10. Visualized actual vs predicted values.
11. Predicted whether a new customer would purchase.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt

Then run the Python program:

python customer_purchase_prediction.py