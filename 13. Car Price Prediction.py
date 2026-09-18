import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame({
    "Year": [2015, 2016, 2017, 2018, 2019,
             2020, 2021, 2022, 2023, 2024],

    "Kilometers": [80000, 70000, 65000, 55000, 50000,
                   40000, 30000, 20000, 15000, 10000],

    "Engine": [1200, 1200, 1500, 1500, 1600,
               1600, 1800, 1800, 2000, 2000],

    "Price": [350000, 400000, 450000, 520000, 600000,
              700000, 850000, 1000000, 1200000, 1400000]
})

X = data[["Year", "Kilometers", "Engine"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

print("\nMean Squared Error:",
      mean_squared_error(y_test, y_pred))

print("R2 Score:",
      r2_score(y_test, y_pred))

# New car prediction
new_car = [[2025, 5000, 2000]]

price = model.predict(new_car)

print("\nPredicted Price for New Car:",
      round(price[0], 2))
