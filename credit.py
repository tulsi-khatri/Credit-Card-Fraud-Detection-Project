import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("All libraries imported successfully ")


print("\nDataset is loading...")

df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully ")

print("\n🔍 Basic Information:")
print(df.info())

print("\n Shape of dataset:", df.shape)

print("\n Missing values check:")
print(df.isnull().sum())

print("\n Class Distribution (0 = Normal, 1 = Fraud):")
print(df['Class'].value_counts())

plt.figure(figsize=(6,4))
df['Class'].value_counts().plot(kind='bar')
plt.title("Normal vs Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()



print("\nSeparating features and target...")

X = df.drop('Class', axis=1)
y = df['Class']

print("Done ")

print("\nScaling data...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling complete ")


print("\nSplitting data into train & test...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


print("\n==============================")
print("Training Logistic Regression...")
print("==============================")

lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

print("Model trained successfully!!!!")

# Prediction
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("\n Logistic Regression Results:")

print("Accuracy:", accuracy_score(y_test, y_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\n==============================")
print("Training Isolation Forest...")
print("==============================")

iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.001,
    random_state=42
)

iso_model.fit(X_train)

print("Isolation Forest trained ")


y_pred_iso = iso_model.predict(X_test)


y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

# Evaluation
print("\n Isolation Forest Results:")

print("Accuracy:", accuracy_score(y_test, y_pred_iso))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_iso))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_iso))



print("\n==============================")
print(" FINAL CONCLUSION")
print("==============================")

print("""
1. Dataset me fraud cases bahut kam the (imbalanced data).
2. Logistic Regression simple hai but fraud detect karne me limited hai.
3. Isolation Forest specially anomaly detection ke liye better perform karta hai.
4. Future me XGBoost ya Deep Learning use kar sakte hain aur better results mil sakte hain.
""")

print("\n Project Completed Successfully ")