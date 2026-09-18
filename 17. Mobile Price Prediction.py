import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    "RAM": [2, 3, 4, 4, 6, 6, 8, 8, 12, 12],
    "Storage": [32, 32, 64, 128, 128, 256, 128, 256, 256, 512],
    "Battery": [3000, 3500, 4000, 4500, 4500,
                5000, 5000, 5500, 6000, 6000],
    "Camera": [8, 12, 12, 16, 20, 24, 32, 48, 50, 64],

    "Price_Range": [
        "Low", "Low", "Low", "Medium", "Medium",
        "Medium", "High", "High", "High", "High"
    ]
})

X = data[["RAM", "Storage", "Battery", "Camera"]]
y = data["Price_Range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual:")
print(y_test.values)

print("\nPredicted:")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# New mobile
new_mobile = [[8, 256, 5000, 50]]

prediction = model.predict(new_mobile)

print("\nPredicted Price Range:",
      prediction[0])
