# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Creating the DataFrame

data = {
    "Age": [
        22, 25, 30, 35, 40,
        45, 50, 23, 28, 33,
        38, 42, 48, 52, 27,
        31, 36, 41, 46, 55
    ],

    "Income": [
        25000, 30000, 35000, 45000, 50000,
        60000, 70000, 28000, 32000, 40000,
        48000, 55000, 65000, 75000, 33000,
        38000, 47000, 58000, 68000, 80000
    ],

    "SpendingScore": [
        75, 80, 65, 60, 55,
        50, 40, 85, 70, 65,
        55, 50, 45, 35, 75,
        70, 60, 50, 45, 30
    ],

    "Purchased": [
        1, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        0, 0, 0, 0, 1,
        1, 1, 0, 0, 0
    ]
}

df = pd.DataFrame(data)


# 2. Exploring the dataset


print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# 3. Visualizing Income vs Spending Score


plt.figure(figsize=(8, 5))

# Customers who did not purchase
plt.scatter(
    df[df["Purchased"] == 0]["Income"],
    df[df["Purchased"] == 0]["SpendingScore"],
    label="Not Purchased"
)

# Customers who purchased
plt.scatter(
    df[df["Purchased"] == 1]["Income"],
    df[df["Purchased"] == 1]["SpendingScore"],
    label="Purchased"
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Income vs Spending Score")
plt.legend()
plt.show()


# 4. Selecting the  features and the  target

X = df[["Age", "Income", "SpendingScore"]]
y = df["Purchased"]


# 5. Splitting data into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# 6. Training Logistic Regression model


model = LogisticRegression()

model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)

# --------------------------------------------------
# 8. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

# --------------------------------------------------
# 9. Visualize Actual vs Predicted values
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted"
)

plt.xlabel("Test Sample")
plt.ylabel("Purchased")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


# 10. Predict whether a new customer will purchase

# Example new customer:
# Age = 29
# Income = 42000
# Spending Score = 70

new_customer = [[29, 42000, 70]]

# if we were to add another example, we could do:
# new_customer = [[25, 40000, 70]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nNew Customer Prediction: Customer will purchase.")
else:
    print("\nNew Customer Prediction: Customer will not purchase.")