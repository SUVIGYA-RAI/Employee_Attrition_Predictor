import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv("employee_attrition_.csv")
print(df.head())
print("Information of the DataSet: \n", df.info())
print("Illustration of Dataset: \n", df.describe())

df.dropna(subset=["Attrition"], inplace=True)
df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["YearsAtCompany"].fillna(df["YearsAtCompany"].median(), inplace=True)
df["JobSatisfaction"].fillna(df["JobSatisfaction"].mode()[0], inplace=True)
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
df["OverTime"].fillna(0, inplace=True)
df["Attrition"] = df["Attrition"].astype(int)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

sns.set(style="whitegrid", palette="Set2")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Age'], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Age Distribution")

sns.histplot(df['MonthlyIncome'], kde=True, ax=axes[1], color="salmon")
axes[1].set_title("Monthly Income Distribution")

sns.histplot(df['YearsAtCompany'], kde=True, ax=axes[2], color="limegreen")
axes[2].set_title("Years at Company Distribution")

plt.tight_layout()
plt.show()

sns.countplot(data=df, x="Attrition", palette="pastel")
plt.title("Attrition Count")
plt.show()

sns.countplot(data=df, x="OverTime", hue="Attrition", palette="coolwarm")
plt.title("Attrition by OverTime")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)  

y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

print("Decision Tree Report:")
print(classification_report(y_test, y_pred_tree))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_estimator(log_model, X_test_scaled, y_test, ax=axes[0], cmap="Blues")
axes[0].set_title("Logistic Regression")

ConfusionMatrixDisplay.from_estimator(tree_model, X_test, y_test, ax=axes[1], cmap="Oranges")
axes[1].set_title("Decision Tree")

plt.tight_layout()
plt.show()