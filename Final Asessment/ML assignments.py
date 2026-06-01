import numpy as np
import csv
import warnings
import os
warnings.filterwarnings('ignore')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Part 1 Numpy")
ids_list       = []
price_list     = []
maturity_list  = []
revenue_list   = []

with open('C:/Users/HP/Documents/GitHub/FULL-STACK-B-6-MON-to-TUE-7pm-to-9pm/FinalAssessment/AI-Adoption-in-Fortune-500-Companies/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', newline='') as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [col.strip() for col in reader.fieldnames]
    for row in reader:
        ids_list.append(row['Company'])
        price_list.append(float(row['AI_ROI_Percent']))
        maturity_list.append(int(row['AI_Maturity_Score']))
        revenue_list.append(float(row['Revenue_USD']))
ids      = np.array(ids_list)
price    = np.array(price_list)
maturity = np.array(maturity_list)
revenue  = np.array(revenue_list)

print("ids:      " , ids)
print("price:    " , price)
print("maturity: " , maturity)
print("revenue:  " , revenue)
print("AI Adoption Dataset - Price (ROI) mean:            " , np.mean(price))
print("AI Adoption Dataset - Price (ROI) average:         " , np.average(price))
print("AI Adoption Dataset - Price (ROI) std:             " , np.std(price))
print("AI Adoption Dataset - Price (ROI) mod:             " , np.median(price))
print("AI Adoption Dataset - Price (ROI) percentile - 25: " , np.percentile(price, 25))
print("AI Adoption Dataset - Price (ROI) percentile - 75: " , np.percentile(price, 75))
print("AI Adoption Dataset - Price (ROI) percentile -  3: " , np.percentile(price, 3))
print("AI Adoption Dataset - Price (ROI) min :            " , np.min(price))
print("AI Adoption Dataset - Price (ROI) max :            " , np.max(price))
print("AI Adoption Dataset - Price (ROI) square: " , np.square(price))
print("AI Adoption Dataset - Price (ROI) sqrt:   " , np.sqrt(price))
print("AI Adoption Dataset - Price (ROI) abs:    " , np.abs(price))


addition       = maturity + revenue
subtraction    = maturity - revenue
multiplication = maturity * revenue
division       = maturity / revenue

print(" AI Adoption Dataset - Maturity + Revenue - Addition:       " , addition)
print(" AI Adoption Dataset - Maturity + Revenue - Subtraction:    " , subtraction)
print(" AI Adoption Dataset - Maturity + Revenue - Multiplication: " , multiplication)
print(" AI Adoption Dataset - Maturity + Revenue - Division:       " , division)

pricePie = (price / np.pi) + 1
sine_values    = np.sin(pricePie)
cosine_values  = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("AI Adoption Dataset - Price div pie - Sine values:    " , sine_values)
print("AI Adoption Dataset - Price div pie - Cosine values:  " , cosine_values)
print("AI Adoption Dataset - Price div pie - Tangent values: " , tangent_values)
print("AI Adoption Dataset - Price div pie - Exponential values: " , np.exp(pricePie))

log_array   = np.log(pricePie)
log10_array = np.log10(pricePie)
print("AI Adoption Dataset - Price div pie - Natural logarithm values: " , log_array)
print("AI Adoption Dataset - Price div pie - Base-10 logarithm values: " , log10_array)

sinh_values = np.sinh(pricePie)
print("AI Adoption Dataset - Price div pie - Hyperbolic Sine values: " , sinh_values)
cosh_values = np.cosh(pricePie)
print("AI Adoption Dataset - Price div pie - Hyperbolic Cosine values: " , cosh_values)
tanh_values = np.tanh(pricePie)
print("AI Adoption Dataset - Price div pie - Hyperbolic Tangent values: " , tanh_values)
asinh_values = np.arcsinh(pricePie)
print("AI Adoption Dataset - Price div pie - Inverse Hyperbolic Sine values: " , asinh_values)
acosh_values = np.arccosh(pricePie)
print("AI Adoption Dataset - Price div pie - Inverse Hyperbolic Cosine values: " , acosh_values)

D2PriceMaturity = np.array([price, maturity])

print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - " , D2PriceMaturity)
print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - dimension:                         " , D2PriceMaturity.ndim)
print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - total number of elements:         " , D2PriceMaturity.size)
print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - gives size in each dimension:     " , D2PriceMaturity.shape)
print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - data type:                        " , D2PriceMaturity.dtype)

D2Slice1 = D2PriceMaturity[:1, :5]
print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - Slicing D2PriceMaturity[:1,:5]:         " , D2Slice1)

D2Slice2 = D2PriceMaturity[:1, 4:15:4]
print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - Slicing D2PriceMaturity[:1, 4:15:4]:    " , D2Slice2)

D2SliceItemOnly  = D2Slice1[0, 1]
print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - Index D2Slice1[0,1]:    " , D2SliceItemOnly)
D2Slice2ItemOnly = D2Slice2[0, 2]

print("AI Adoption Dataset - ROI Plus Maturity - 2 dimensional array - Index D2Slice2[0, 2]:   " , D2Slice2ItemOnly)

for elem in np.nditer(D2PriceMaturity):
    print(elem)
for index, elem in np.ndenumerate(D2PriceMaturity):
    print(index, elem)

D2Reshaped = np.reshape(D2PriceMaturity, (1, 12000))
print("AI Adoption Dataset - ROI Plus Maturity - np.reshape(D2PriceMaturity, (1, 12000)):        " , D2Reshaped)
print("AI Adoption Dataset - ROI Plus Maturity - np.reshape(D2PriceMaturity, (1, 12000)) Size:   " , D2Reshaped.size)
print("AI Adoption Dataset - ROI Plus Maturity - np.reshape(D2PriceMaturity, (1, 12000)) ndim:   " , D2Reshaped.ndim)
print("AI Adoption Dataset - ROI Plus Maturity - np.reshape(D2PriceMaturity, (1, 12000)) shape:  " , D2Reshaped.shape)
print()



import pandas as pd
print("Part 2 Pandas")

df = pd.read_csv('C:/Users/HP/Documents/GitHub/FULL-STACK-B-6-MON-to-TUE-7pm-to-9pm/FinalAssessment/AI-Adoption-in-Fortune-500-Companies/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv',
                 delimiter=",")

df.columns = df.columns.str.strip()

print(df)
print("df - data types: " , df.dtypes)
print("df.info():   " , df.info())

print('Last three Rows:')
print(df.tail(3))

print('First Three Rows:')
print(df.head(3))
print()

print(df.describe())
print( df.shape)
print()

industry = df['Industry']
print(industry)
print()

industry_company = df[['Industry', 'Company']]
print(industry_company)
print()

second_row = df.loc[1]
print(second_row)
print()

second_row2 = df.loc[[1, 3]]
print(second_row2)
print()

second_row3 = df.loc[1:5]
print(second_row3)
print()

second_row4 = df.loc[df['Industry'] == 'Technology']
print(second_row4)
print()

second_row5 = df.loc[:1, 'Industry']
print(second_row5)
print()

second_row6 = df.loc[:1, ['Industry', 'Company']]
print(second_row6)
print()

second_row7 = df.loc[:1, 'Industry':'Revenue_USD']
print(second_row7)
print()

second_row8 = df.loc[df['Industry'] == 'Technology', 'Industry':'Revenue_USD']
print(second_row8)
print()

df_index_col = pd.read_csv('C:/Users/HP/Documents/GitHub/FULL-STACK-B-6-MON-to-TUE-7pm-to-9pm/FinalAssessment/AI-Adoption-in-Fortune-500-Companies/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv',
                           delimiter=",", index_col=0)
df_index_col.columns = df_index_col.columns.str.strip()
df_index_col.index.name = 'Year'

print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())

row_2020 = df_index_col.loc[2020]
print(row_2020)
print()

row_multi = df_index_col.loc[[2020, 2022]]
print(row_multi)
print()

row_slice = df_index_col.loc[[2020, 2021, 2022]]
print(row_slice)
print()

row_cond = df_index_col.loc[df_index_col['Industry'] == 'Technology']
print(row_cond)
print()

iloc_row = df.iloc[0]
print(iloc_row)
print()

iloc_row2 = df.iloc[[1, 3, 5]]
print(iloc_row2)
print()

iloc_row3 = df.iloc[2:5]
print(iloc_row3)
print()

iloc_col = df.iloc[:, 2]
print(iloc_col)
print()

iloc_col2 = df.iloc[:, [2, 4]]
print(iloc_col2)
print()

iloc_col3 = df.iloc[:, 2:4]
print(iloc_col3)
print()

iloc_combined = df.iloc[[1, 3, 5], 2:4]
print(iloc_combined)
print()

df.loc[len(df.index)] = [9999, 'TestCompany', 'Technology', 'USA', 'Real',
                          'Enterprise', 999999999.99, 'Yes',
                          'Generative AI', 38.50, 95]
print(df)
print()


df.drop(0, axis=0, inplace=True)
df.drop(index=1, inplace=True)
df.drop([2, 3], axis=0, inplace=True)
print(df)

df.drop('Use_Case', axis=1, inplace=True)
df.drop(columns='Company_Type', inplace=True)
df.drop(['Country', 'Employee_Size'], axis=1, inplace=True)
print(df)


df.rename(columns={'Industry': 'Industry_Changed'}, inplace=True)
df.rename(mapper={'AI_ROI_Percent': 'ROI_Changed', 'AI_Maturity_Score': 'Maturity_Changed'}, axis=1, inplace=True)
print(df)

df.rename(index={4: 400}, inplace=True)
df.rename(mapper={5: 500, 6: 600}, axis=0, inplace=True)
print(df)

selected_rows = df.query('Industry_Changed == \'Technology\' or ROI_Changed > 35')
print(selected_rows.to_string())
print(len(selected_rows))


sorted_df = df.sort_values(by='ROI_Changed')
print(sorted_df.to_string(index=False))

df1 = df.sort_values(by=['ROI_Changed', 'Maturity_Changed'])
print(df1.to_string(index=False))

grouped = df.groupby('Industry_Changed')['ROI_Changed'].sum()
print(grouped.to_string())
print("grouped: " , len(grouped))

df_cleaned = df.dropna()
print("Cleaned Data:\n" , df_cleaned)
df.fillna(0, inplace=True)
print("\nData after filling NaN with 0:\n" , df)

data = [2, 4, 6, 8]
array1 = pd.array(data)
print(array1)

int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)
print()

import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("Part 3 Seaborn")

df_vis = pd.read_csv('C:/Users/HP/Documents/GitHub/FULL-STACK-B-6-MON-to-TUE-7pm-to-9pm/FinalAssessment/AI-Adoption-in-Fortune-500-Companies/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv')
df_vis.columns = df_vis.columns.str.strip()

variables = ['AI_Maturity_Score', 'Revenue_USD']

for var in variables:
    plt.figure()
    sns.regplot(x=var, y='AI_ROI_Percent', data=df_vis).set(
        title=f'Regression plot of {var} and AI ROI Percent')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'regplot_{var}.png'), dpi=120, bbox_inches='tight')
    plt.close()

plt.figure()
correlations = df_vis[['Year', 'AI_ROI_Percent', 'AI_Maturity_Score', 'Revenue_USD']].corr()
print("correlations...\n" , correlations)
g = sns.heatmap(correlations, annot=True).set(title='Heat map of AI Adoption Data - Pearson Correlations')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_correlations.png'), dpi=120, bbox_inches='tight')
plt.close()

plt.figure()
sns.histplot(df_vis['AI_ROI_Percent'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of AI ROI Percent')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'hist_roi.png'), dpi=120, bbox_inches='tight')
plt.close()

plt.figure()
sns.boxplot(data=df_vis, x='Year', y='AI_Maturity_Score', palette='pastel')
plt.title('AI Maturity Score by Year')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'box_maturity_year.png'), dpi=120, bbox_inches='tight')
plt.close()

sns.pairplot(df_vis[['AI_ROI_Percent', 'AI_Maturity_Score', 'Revenue_USD']])
plt.savefig(os.path.join(OUTPUT_DIR, 'pairplot.png'), dpi=100, bbox_inches='tight')
plt.close()
print()

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
df_ml = pd.read_csv('C:/Users/HP/Documents/GitHub/FULL-STACK-B-6-MON-to-TUE-7pm-to-9pm/FinalAssessment/AI-Adoption-in-Fortune-500-Companies/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv')
df_ml.columns = df_ml.columns.str.strip()

le_industry = LabelEncoder()
le_empsize  = LabelEncoder()
le_usesai   = LabelEncoder()
le_comptype = LabelEncoder()

df_ml['Industry_enc']   = le_industry.fit_transform(df_ml['Industry'])
df_ml['EmpSize_enc']    = le_empsize.fit_transform(df_ml['Employee_Size'])
df_ml['Uses_AI_enc']    = le_usesai.fit_transform(df_ml['Uses_AI'])
df_ml['CompType_enc']   = le_comptype.fit_transform(df_ml['Company_Type'])

df_ml['ROI_High'] = (df_ml['AI_ROI_Percent'] > 25).astype(int)

feature_cols = ['Year', 'AI_Maturity_Score', 'Revenue_USD',
                'Industry_enc', 'EmpSize_enc', 'CompType_enc']

X = df_ml[feature_cols]
y_cls = df_ml['ROI_High']
y_reg = df_ml['AI_ROI_Percent']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("X: " , X)
print("y_cls (classification target): " , y_cls.values)
print("y_reg (regression target):     " , y_reg.values)
print(f"Train/Test split will be 80/20 with random_state=25")

SEED = 25
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X_scaled, y_cls, train_size=0.8, random_state=SEED)

Xr_train, Xr_test, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_reg, train_size=0.8, random_state=SEED)
print(f"Train size: {round(len(X_train) / len(X_scaled) * 100)}% \nTest size: {round(len(X_test) / len(X_scaled) * 100)}%")

print()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

print("Part 5 Classification")

logistic_regression = LogisticRegression(max_iter=1000, random_state=SEED)
svm                 = SVC(kernel='linear', random_state=SEED)
tree                = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=SEED)
logistic_regression.fit(X_train, y_train_cls)
svm.fit(X_train, y_train_cls)
tree.fit(X_train, y_train_cls)
log_reg_preds = logistic_regression.predict(X_test)
svm_preds     = svm.predict(X_test)
tree_preds    = tree.predict(X_test)

model_preds = {
    "Logistic Regression"   : log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree"         : tree_preds
}
for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test_cls, preds, target_names=['Not High ROI', 'High ROI'])}", sep="\n\n")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Classification - Confusion Matrices', fontsize=13, fontweight='bold')

for ax, (model_name, preds) in zip(axes, model_preds.items()):
    cnf_matrix = confusion_matrix(y_test_cls, preds)
    class_names = [0, 1]
    tick_marks  = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    ax.xaxis.set_label_position("top")
    ax.set_title(model_name, y=1.01)
    ax.set_ylabel('Actual label')
    ax.set_xlabel('Predicted label')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'classification_confusion_matrices.png'), dpi=130, bbox_inches='tight')
plt.close()

from sklearn import metrics
y_pred_proba = logistic_regression.predict_proba(X_test)[::, 1]
fpr, tpr, _  = metrics.roc_curve(y_test_cls, y_pred_proba)
auc          = metrics.roc_auc_score(y_test_cls, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label="Logistic Regression, auc=" + str(round(auc, 4)))
plt.legend(loc=4)
plt.title('ROC Curve - Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=130, bbox_inches='tight')
plt.close()
print()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Part 6 Regression")

y_simple = df_ml['AI_ROI_Percent'].values.reshape(-1, 1)
X_simple = df_ml['AI_Maturity_Score'].values.reshape(-1, 1)

print("y_simple: " , y_simple)
print("X_simple: " , X_simple)
print(df_ml['AI_Maturity_Score'].values)
print(df_ml['AI_Maturity_Score'].values.shape)
print(X_simple.shape)
print(X_simple)

SEED = 42
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=SEED)
print(f"Train size: {round(len(X_s_train) / len(X_simple) * 100)}% \nTest size: {round(len(X_s_test) / len(X_simple) * 100)}%")
regressor = LinearRegression()
regressor.fit(X_s_train, y_s_train)

print("regressor.intercept_: " , regressor.intercept_)
print("regressor.coef_:      " , regressor.coef_)

def calc(slope, intercept, maturity_score):
    return slope * maturity_score + intercept

score = calc(regressor.coef_, regressor.intercept_, 80)
print("Predicted ROI for Maturity Score 80: " , score)

score2 = regressor.predict([[80]])
print("regressor.predict([[80]]): " , score2)

y_pred_simple = regressor.predict(X_s_test)

df_preds = pd.DataFrame({'Actual': y_s_test.squeeze(), 'Predicted': y_pred_simple.squeeze()})
print("Actual vs Predicted:\n" , df_preds)

mae  = mean_absolute_error(y_s_test, y_pred_simple)
mse  = mean_squared_error(y_s_test, y_pred_simple)
rmse = np.sqrt(mse)
r2   = r2_score(y_s_test, y_pred_simple)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')


print("\nMultiple Linear Regression:")
print("X.shape: " , X.shape)

regressor_multi = LinearRegression()
regressor_multi.fit(Xr_train, y_train_reg)

print("regressor_multi.intercept_: " , regressor_multi.intercept_)
print("regressor_multi.coef_:      " , regressor_multi.coef_)
feature_names      = feature_cols
model_coefficients = regressor_multi.coef_

coefficients_df = pd.DataFrame(data=model_coefficients,
                                index=feature_names,
                                columns=['Coefficient value'])
print(coefficients_df)

y_pred_multi = regressor_multi.predict(Xr_test)
results = pd.DataFrame({'Actual': y_test_reg, 'Predicted': y_pred_multi})
print("Actual vs Predicted (Multiple Regression):\n" , results)

mae_m  = mean_absolute_error(y_test_reg, y_pred_multi)
mse_m  = mean_squared_error(y_test_reg, y_pred_multi)
rmse_m = np.sqrt(mse_m)

print(f'Mean absolute error: {mae_m:.2f}')
print(f'Mean squared error: {mse_m:.2f}')
print(f'Root mean squared error: {rmse_m:.2f}')

actual_minus_predicted   = sum((y_test_reg - y_pred_multi) ** 2)
actual_minus_actual_mean = sum((y_test_reg - y_test_reg.mean()) ** 2)
r2_manual = 1 - actual_minus_predicted / actual_minus_actual_mean
print('R²:', r2_manual)
print(" R2 also comes from score method of LinearRegression:\n" , regressor_multi.score(Xr_test, y_test_reg))

plt.figure()
sns.regplot(x=y_test_reg.values, y=y_pred_multi, scatter_kws={'alpha': 0.3}).set(
    title='Regression plot of Actual vs Predicted ROI')
plt.xlabel('Actual ROI %')
plt.ylabel('Predicted ROI %')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'regression_actual_vs_predicted.png'), dpi=130, bbox_inches='tight')
plt.close()
print()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

print("Part 7 Clustering")

X_blob, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)

plt.figure()
plt.grid(True)
plt.scatter(X_blob[:, 0], X_blob[:, 1], s=20)
plt.title('make_blobs - Sample Data Before Clustering')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'blobs_before_clustering.png'), dpi=120, bbox_inches='tight')
plt.close()
kmeans_blob = KMeans(n_clusters=4)
kmeans_blob.fit(X_blob)
y_kmeans = kmeans_blob.predict(X_blob)

plt.figure()
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans_blob.cluster_centers_
color   = np.array(['blue', 'hotpink', 'black', 'green'])
plt.scatter(centers[:, 0], centers[:, 1], c=color, s=100, alpha=0.9)
plt.title('make_blobs - After KMeans Clustering (K=4)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'blobs_after_clustering.png'), dpi=120, bbox_inches='tight')
plt.close()

X_clus        = df_ml[['AI_ROI_Percent', 'AI_Maturity_Score', 'Revenue_USD']].copy()
X_clus_scaled = StandardScaler().fit_transform(X_clus)
kmeans = KMeans(n_clusters=3, max_iter=100, random_state=SEED, n_init=10)
kmeans.fit(X_clus_scaled)

print("cluster_centers_.shape: " , kmeans.cluster_centers_.shape)
df_ml['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_clus_scaled, df_ml['Cluster'])
print(f"Silhouette Score: {sil_score:.4f}")

print("\nCluster Summary (mean values per cluster):")
print(df_ml.groupby('Cluster')[['AI_ROI_Percent', 'AI_Maturity_Score', 'Revenue_USD']].mean().round(2))

inertia_list = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km.fit(X_clus_scaled)
    inertia_list.append(km.inertia_)

plt.figure(figsize=(7.5, 3.5))
plt.plot(list(k_range), inertia_list, marker='o', color='steelblue')
plt.title('Elbow Method - Finding Best K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_elbow.png'), dpi=120, bbox_inches='tight')
plt.close()

plt.figure(figsize=(7.5, 3.5))
plt.scatter(df_ml['AI_Maturity_Score'], df_ml['AI_ROI_Percent'],
            c=df_ml['Cluster'], s=20, cmap='summer')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0],
            marker='x', c='r', s=50, alpha=0.9)
plt.title('KMeans Clustering - Maturity Score vs ROI')
plt.xlabel('AI Maturity Score')
plt.ylabel('AI ROI %')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_scatter.png'), dpi=120, bbox_inches='tight')
plt.close()

sns.pairplot(df_ml[['AI_ROI_Percent', 'AI_Maturity_Score', 'Revenue_USD', 'Cluster']],
             hue='Cluster')
plt.savefig(os.path.join(OUTPUT_DIR, 'clustering_pairplot.png'), dpi=100, bbox_inches='tight')
plt.close()
print()


from sklearn.ensemble import RandomForestClassifier

print("Part 8 Ensemble Learning")

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_classifier.fit(X_train, y_train_cls)
rf_preds = rf_classifier.predict(X_test)

print("Accuracy: " , accuracy_score(y_test_cls, rf_preds))
from sklearn.metrics import classification_report
target_names = ['Not High ROI', 'High ROI']
print(classification_report(y_test_cls, rf_preds, target_names=target_names))

cnf_matrix = confusion_matrix(y_test_cls, rf_preds)
print(" Model Evaluation using Confusion Matrix: " , cnf_matrix)

class_names = [0, 1]
fig, ax     = plt.subplots()
tick_marks  = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix - Random Forest', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_confusion_matrix.png'), dpi=130, bbox_inches='tight')
plt.close()

importances = rf_classifier.feature_importances_
feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False)

print("Feature Importances:")
print(feat_df.to_string(index=False))

plt.figure()
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title('Random Forest - Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_feature_importance.png'), dpi=130, bbox_inches='tight')
plt.close()

y_pred_proba_rf = rf_classifier.predict_proba(X_test)[::, 1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test_cls, y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_test_cls, y_pred_proba_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, label="Random Forest, auc=" + str(round(auc_rf, 4)))
plt.legend(loc=4)
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ensemble_roc_curve.png'), dpi=130, bbox_inches='tight')
plt.close()
print()
print("All diagrams are saved in outputs folder in my assignments")