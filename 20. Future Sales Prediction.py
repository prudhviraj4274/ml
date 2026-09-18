import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Dataset
data = pd.DataFrame({
    "Month": np.arange(1, 13),

    "Advertising": [
        10, 12, 15, 18, 20, 22,
        25, 28, 30, 32, 35, 38
    ],

    "Sales": [
        100, 110, 125, 135, 150, 160,
        175, 190, 205, 220, 235, 250
    ]
})

X = data[["Month", "Advertising"]]
y = data["Sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Actual Sales:")
print(y_test.values)

print("\nPredicted Sales:")
print(y_pred)

print("\nMean Absolute Error:")
print(mean_absolute_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# Predict future month
future_data = [[13, 40]]

future_sales = model.predict(future_data)

print("\nPredicted Future Sales:",
      round(future_sales[0], 2))
