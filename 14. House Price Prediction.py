import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.DataFrame({
    "Area": [800, 1000, 1200, 1500, 1800,
             2000, 2200, 2500, 2800, 3000],

    "Bedrooms": [2, 2, 3, 3, 3,
                 4, 4, 4, 5, 5],

    "Age": [20, 15, 12, 10, 8,
            7, 5, 4, 3, 2],

    "Price": [2500000, 3000000, 3800000, 4500000, 5200000,
              6000000, 6800000, 7500000, 8500000, 9500000]
})

X = data[["Area", "Bedrooms", "Age"]]
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(y_pred)

print("\nMAE:")
print(mean_absolute_error(y_test, y_pred))

print("\nR2 Score:")
print(r2_score(y_test, y_pred))

# New house
new_house = [[1800, 3, 5]]

price = model.predict(new_house)

print("\nPredicted House Price:",
      round(price[0], 2))
