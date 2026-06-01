import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv(r"c:/Users/HP/Documents/GitHub/My assignments/My Assignments/archive (1)/WineQT.csv")
df = df.drop(columns=["Id"])

print("Shape of dataset:", df.shape)
print()
print("First 5 rows:")
print(df.head())
print()
print("Checking for missing values:")
print(df.isnull().sum().sum(), "missing values found")
print()
print("Quality distribution:")
print(df["quality"].value_counts().sort_index())


df["quality_label"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

print()
print("Binary Class distribution (0 = Low/Medium, 1 = High Quality):")
print(df["quality_label"].value_counts())


plt.figure(figsize=(6, 4))
sns.countplot(x="quality", data=df)
plt.title("Wine Quality Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


X = df.drop(columns=["quality", "quality_label"])
y = df["quality_label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print()
print("Training size:", X_train.shape[0])
print("Testing size:", X_test.shape[0])


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print()
print("Logistic Regression Results:")
print("Accuracy:", round(accuracy_score(y_test, lr_pred) * 100, 2), "%")
print(classification_report(y_test, lr_pred))


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("Decision Tree Results:")
print("Accuracy:", round(accuracy_score(y_test, dt_pred) * 100, 2), "%")
print(classification_report(y_test, dt_pred))


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("Random Forest Results:")
print("Accuracy:", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")
print(classification_report(y_test, rf_pred))


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models = [("Logistic Regression", lr_pred), ("Decision Tree", dt_pred), ("Random Forest", rf_pred)]

for ax, (name, pred) in zip(axes, models):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()


accuracies = {
    "Logistic Regression": accuracy_score(y_test, lr_pred),
    "Decision Tree": accuracy_score(y_test, dt_pred),
    "Random Forest": accuracy_score(y_test, rf_pred)
}

plt.figure(figsize=(7, 4))
plt.bar(accuracies.keys(), [v * 100 for v in accuracies.values()], color=["steelblue", "orange", "green"])
plt.ylim([70, 102])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

print("Best Model:", max(accuracies, key=accuracies.get))
print("Best Accuracy:", round(max(accuracies.values()) * 100, 2), "%")
print()