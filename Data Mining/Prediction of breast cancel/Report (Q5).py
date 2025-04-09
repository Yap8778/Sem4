import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
file_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\Complete_dataset.csv"
df = pd.read_csv(file_path)

# Store Patient ID
patient_ids = df["ID"]

df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})  # 1 = Malignant, 0 = Benign
# Select features and target variable
X = df.iloc[:, 2:32]  # All 30 features
y = df["Diagnosis"]    # Target: Malignancy classification

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Compute Mutual Information (MI) for feature selection
# Calculate MI scores
mi_scores = mutual_info_classif(X, y)

# Create MI DataFrame
mi_df = pd.DataFrame({
    "Feature": X.columns,
    "Mutual Information": mi_scores
})

# Plot MI Scores Bar Chart to visualize it
plt.figure(dpi=100, figsize=(10, 6))
mi_df.sort_values(by="Mutual Information", ascending=False, inplace=True)
plt.barh(mi_df["Feature"], mi_df["Mutual Information"], color="skyblue")
plt.xlabel("Mutual Information Score")
plt.ylabel("Features")
plt.title("Mutual Information Scores for Feature Selection")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# Select the top 10 most important features
selected_features = mi_df["Feature"][:10].tolist()

# Save MI results
mi_csv_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\MutualInformation_features.csv"
mi_df.to_csv(mi_csv_path, index=False)

print("\nMutual information calculation completed, results saved to:", mi_csv_path)
print("\nTop 10 selected features based on mutual information:", selected_features)

# Step 3: Train the KNN model using only selected 10 features
# Extract selected features
X_selected = df[selected_features].values

# Normalize selected features
X_scaled_selected = scaler.fit_transform(X_selected)

# Split data
X_train_sel, X_test_sel, y_train_sel, y_test_sel, id_train_sel, id_test_sel = train_test_split(
    X_scaled_selected, y, patient_ids, test_size=0.2, random_state=42
)

# Train the KNN model
knn_selected = KNeighborsClassifier(n_neighbors=5)
knn_selected.fit(X_train_sel, y_train_sel)

# Make predictions
y_pred_sel = knn_selected.predict(X_test_sel)

# Step 4: Save the prediction results to CSV
# Ensure index consistency
id_test_sel = id_test_sel.reset_index(drop=True)
y_test_sel = y_test_sel.reset_index(drop=True)

# Create DataFrame
results_df_selected = pd.DataFrame({
    "Patient ID": id_test_sel, 
    "Actual Diagnosis": y_test_sel, 
    "Predicted Diagnosis": y_pred_sel
})

# Save prediction results
prediction_csv_path_sel = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\Prediction_vs_Actual_MutualInformation.csv"
results_df_selected.to_csv(prediction_csv_path_sel, index=False)

# Calculate accuracy
accuracy_selected = accuracy_score(y_test_sel, y_pred_sel)

# Step 5: Print results
print("\n--- KNN Model Results With Mutual Information Feature Selection ---")
print(f"Accuracy using selected features: {accuracy_selected:.4f}")
print(f"\nPredict Result has been saved to: {prediction_csv_path_sel}")
