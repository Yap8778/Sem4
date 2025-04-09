import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read the dataset from CSV
file_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\congressional+voting+records\\dataset.csv"
df = pd.read_csv(file_path)

# Data Imputation (Mode)
df = df.replace("?", np.nan)
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encoding
# Convert target "republican"/"democrat" to numeric (1/0)
# Convert "y"/"n" votes to numeric (1/0)
df["Target"] = df["Target"].map({"republican": 1, "democrat": 0})

feature_cols = [c for c in df.columns if c != "Target"]
for col in feature_cols:
    df[col] = df[col].map({"y": 1, "n": 0})

# Save the imputed + encoded dataset
complete_dataset_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\congressional+voting+records\\Complete_dataset.csv"
df.to_csv(complete_dataset_path, index=False)
print(f"Imputed + encoded dataset saved to: {complete_dataset_path}")

# Split into features (X) and target (y)
X = df.drop(columns=["Target"])
y = df["Target"]

# Scale features to [0,1] for Chi-Square
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Chi-Square for Top 10 Features
chi2_selector = SelectKBest(score_func=chi2, k=10)
X_selected = chi2_selector.fit_transform(X_scaled, y)

# Save the Chi-Square scores to CSV (for all features, not just top 10)
chi2_selector.fit(X_scaled, y)
chi2_scores = pd.DataFrame({
    "Feature": X.columns,
    "Chi2 Score": chi2_selector.scores_
})
chi2_scores.sort_values(by="Chi2 Score", ascending=False, inplace=True)

chisquare_csv_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\congressional+voting+records\\chisquare_score.csv"
chi2_scores.to_csv(chisquare_csv_path, index=False)
print(f"\nChi-Square scores saved to: {chisquare_csv_path}")

# KNN Model - Using Selected (Top 10) Features
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

knn_sel = KNeighborsClassifier(n_neighbors=5)
knn_sel.fit(X_train_sel, y_train_sel)
y_pred_sel = knn_sel.predict(X_test_sel)
acc_sel = accuracy_score(y_test_sel, y_pred_sel)

# KNN Model - Using ALL Features
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
knn_all = KNeighborsClassifier(n_neighbors=5)
knn_all.fit(X_train_all, y_train_all)
y_pred_all = knn_all.predict(X_test_all)
acc_all = accuracy_score(y_test_all, y_pred_all)

# Print Results
print("\nComparison of KNN Models:")
print(f"1. Accuracy with Top-10 Chi-Square Features: {acc_sel:.4f}")
print(f"2. Accuracy with ALL Features:               {acc_all:.4f}")
