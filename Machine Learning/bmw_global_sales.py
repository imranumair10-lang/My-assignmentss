import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("bmw_global_sales_2018_2025.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print(df.info())
print(df.describe())

print(df.isnull().sum())

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='Region', y='Units_Sold', data=df)
plt.title("Units Sold by Region")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Revenue_EUR'], bins=30, kde=True)
plt.title("Revenue Distribution")
plt.show()

le_region = LabelEncoder()
le_model = LabelEncoder()

df['Region'] = le_region.fit_transform(df['Region'])
df['Model'] = le_model.fit_transform(df['Model'])

features = [
    'Units_Sold',
    'Avg_Price_EUR',
    'BEV_Share',
    'Premium_Share',
    'GDP_Growth',
    'Fuel_Price_Index',
    'Region',
    'Model'
]

X = df[features]

y = df['Revenue_EUR']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Linear Regression Results")
print("R2 Score:", r2)
print("RMSE:", rmse)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()

median_sales = df['Units_Sold'].median()
df['High_Sales'] = (df['Units_Sold'] > median_sales).astype(int)

X_class = df[
    [
        'Avg_Price_EUR',
        'BEV_Share',
        'Premium_Share',
        'GDP_Growth',
        'Fuel_Price_Index',
        'Region',
        'Model'
    ]
]
y_class = df['High_Sales']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_c, y_train_c)

y_pred_class = log_model.predict(X_test_c)

accuracy = accuracy_score(y_test_c, y_pred_class)

print("Logistic Regression Results")
print("Accuracy:", accuracy)

print(confusion_matrix(y_test_c, y_pred_class))

print(classification_report(y_test_c, y_pred_class))

coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': linear_model.coef_
})

print(coefficients)

plt.figure(figsize=(8,5))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)

plt.title("Feature Impact on Revenue")
plt.show()
results = pd.DataFrame({
    "Actual_Revenue": y_test,
    "Predicted_Revenue": y_pred
})

results.to_csv("revenue_predictions.csv", index=False)

print("Prediction file saved: revenue_predictions.csv")
