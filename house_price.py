import pandas as pd

import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("house_data.csv")

print("Columns:", df.columns)
print("Assumption: area in sqft, price in ₹")
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)
target_column = "price"
X = df.drop(target_column, axis=1)
y = df[target_column]

feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"MAE: ₹{mae:.2f}")
print(f"R2 Score: {r2:.2f}")

joblib.dump(model, "model.pkl")




print("\nEnter values for prediction:")

area = float(input("Enter area (in square feet): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

input_data = pd.DataFrame([[area, bedrooms, bathrooms]],
                          columns=["area", "bedrooms", "bathrooms"])

input_data = input_data.reindex(columns=feature_columns, fill_value=0)

prediction = model.predict(input_data)

print(f"\n Predicted House Price: ₹{prediction[0]:,.2f}")