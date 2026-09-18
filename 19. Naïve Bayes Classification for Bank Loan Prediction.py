import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    "Age": [25,35,45,28,52,31,41,26,55,38,29,48,34,50,27],
    "Income": [25000,50000,75000,30000,90000,45000,65000,
               28000,95000,55000,27000,80000,48000,85000,32000],
    "LoanAmount": [10000,20000,30000,12000,25000,18000,22000,
                   10000,20000,25000,12000,30000,18000,25000,15000],
    "CreditScore": [600,700,750,580,800,680,720,590,820,710,570,780,690,790,610],
    "Loan_Status": [
        "No","Yes","Yes","No","Yes",
        "Yes","Yes","No","Yes","Yes",
        "No","Yes","Yes","Yes","No"
    ]
})

encoder = LabelEncoder()
data["Loan_Status"] = encoder.fit_transform(data["Loan_Status"])

X = data[["Age", "Income", "LoanAmount", "CreditScore"]]
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = GaussianNB()

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
