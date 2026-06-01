import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


df1 = pd.read_csv(r"C:\Users\HP\Documents\GitHub\My assignments\My Assignments\archive\car data.csv")

df1["Price_Category"] = pd.cut(df1["Selling_Price"], bins=[0, 3, 7, 100], labels=["Low", "Medium", "High"])

le = LabelEncoder()
df1["Fuel_Type"] = le.fit_transform(df1["Fuel_Type"])
df1["Seller_Type"] = le.fit_transform(df1["Seller_Type"])
df1["Transmission"] = le.fit_transform(df1["Transmission"])
df1["Price_Category"] = le.fit_transform(df1["Price_Category"])

features1 = ["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"]
X1 = df1[features1]
y1 = df1["Price_Category"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = DecisionTreeClassifier()
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)
acc1 = accuracy_score(y1_test, y1_pred)

print("Dataset 1: car data.csv")
print("Accuracy:", round(acc1, 2))
print(classification_report(y1_test, y1_pred, target_names=["Low", "Medium", "High"]))
print()



df2 = pd.read_csv(r"C:\Users\HP\Documents\GitHub\My assignments\My Assignments\archive\CAR DETAILS FROM CAR DEKHO.csv")

df2 = df2.dropna()
df2["Price_Category"] = pd.cut(df2["selling_price"], bins=[0, 300000, 700000, 10000000], labels=["Low", "Medium", "High"])

le2 = LabelEncoder()
df2["fuel"] = le2.fit_transform(df2["fuel"])
df2["seller_type"] = le2.fit_transform(df2["seller_type"])
df2["transmission"] = le2.fit_transform(df2["transmission"])
df2["owner"] = le2.fit_transform(df2["owner"])
df2["Price_Category"] = le2.fit_transform(df2["Price_Category"])

features2 = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
X2 = df2[features2]
y2 = df2["Price_Category"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = DecisionTreeClassifier()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)
acc2 = accuracy_score(y2_test, y2_pred)

print("Dataset 2: CAR DETAILS FROM CAR DEKHO.csv")
print("Accuracy:", round(acc2, 2))
print(classification_report(y2_test, y2_pred, target_names=["Low", "Medium", "High"]))
print()



df3 = pd.read_csv(r"C:\Users\HP\Documents\GitHub\My assignments\My Assignments\archive\Car details v3.csv")

df3 = df3.dropna()

df3["mileage"] = df3["mileage"].str.replace(" kmpl", "").str.replace(" km/kg", "")
df3["engine"] = df3["engine"].str.replace(" CC", "")
df3["max_power"] = df3["max_power"].str.replace(" bhp", "")

df3["mileage"] = pd.to_numeric(df3["mileage"], errors="coerce")
df3["engine"] = pd.to_numeric(df3["engine"], errors="coerce")
df3["max_power"] = pd.to_numeric(df3["max_power"], errors="coerce")

df3 = df3.dropna()

df3["Price_Category"] = pd.cut(df3["selling_price"], bins=[0, 300000, 700000, 10000000], labels=["Low", "Medium", "High"])

le3 = LabelEncoder()
df3["fuel"] = le3.fit_transform(df3["fuel"])
df3["seller_type"] = le3.fit_transform(df3["seller_type"])
df3["transmission"] = le3.fit_transform(df3["transmission"])
df3["owner"] = le3.fit_transform(df3["owner"])
df3["Price_Category"] = le3.fit_transform(df3["Price_Category"])

features3 = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "seats"]
X3 = df3[features3]
y3 = df3["Price_Category"]

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model3 = DecisionTreeClassifier()
model3.fit(X3_train, y3_train)
y3_pred = model3.predict(X3_test)
acc3 = accuracy_score(y3_test, y3_pred)

print("Dataset 3: Car details v3.csv")
print("Accuracy:", round(acc3, 2))
print(classification_report(y3_test, y3_pred, target_names=["Low", "Medium", "High"]))
print()



df4 = pd.read_csv(r"C:\Users\HP\Documents\GitHub\My assignments\My Assignments\archive\car details v4.csv")

df4 = df4.dropna()

df4["Engine"] = df4["Engine"].str.replace(" cc", "").str.strip()
df4["Max Power"] = df4["Max Power"].str.split(" ").str[0]
df4["Engine"] = pd.to_numeric(df4["Engine"], errors="coerce")
df4["Max Power"] = pd.to_numeric(df4["Max Power"], errors="coerce")
df4 = df4.dropna()

df4["Price_Category"] = pd.cut(df4["Price"], bins=[0, 300000, 700000, 10000000], labels=["Low", "Medium", "High"])

le4 = LabelEncoder()
df4["Fuel Type"] = le4.fit_transform(df4["Fuel Type"])
df4["Transmission"] = le4.fit_transform(df4["Transmission"])
df4["Seller Type"] = le4.fit_transform(df4["Seller Type"])
df4["Owner"] = le4.fit_transform(df4["Owner"])
df4["Location"] = le4.fit_transform(df4["Location"])
df4["Color"] = le4.fit_transform(df4["Color"])
df4["Drivetrain"] = le4.fit_transform(df4["Drivetrain"])
df4["Price_Category"] = le4.fit_transform(df4["Price_Category"])

features4 = ["Year", "Kilometer", "Fuel Type", "Transmission", "Seller Type", "Owner", "Engine", "Max Power", "Seating Capacity"]
X4 = df4[features4]
y4 = df4["Price_Category"]

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42)

model4 = DecisionTreeClassifier()
model4.fit(X4_train, y4_train)
y4_pred = model4.predict(X4_test)
acc4 = accuracy_score(y4_test, y4_pred)

print("Dataset 4: car details v4.csv")
print("Accuracy:", round(acc4, 2))
print(classification_report(y4_test, y4_pred))
print()