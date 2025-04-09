import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
file_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\Complete_dataset.csv"
df = pd.read_csv(file_path)

# Store Patient ID
patient_ids = df["ID"]

# Convert Diagnosis ('M' = Malignant, 'B' = Benign) to binary values
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0}).astype(int)  # 1 = Malignant, 0 = Benign

# Step 2: Feature Selection using t-test
# Separate by diagnosis
df_group_0 = df[df["Diagnosis"] == 0]  # Benign (0)
df_group_1 = df[df["Diagnosis"] == 1]  # Malignant (1)

# Get feature columns (excluding ID and Diagnosis)
feature_columns = df.columns[2:32]

# Compute p-values using t-test
p_values = []
for feature in feature_columns:
    group_0_values = df_group_0[feature]
    group_1_values = df_group_1[feature]
    
    # Calculate t-test p-value
    _, p_value = stats.ttest_ind(group_0_values, group_1_values, equal_var=False)
    p_values.append((feature, p_value))

# Create DataFrame for p-values
p_value_df = pd.DataFrame(p_values, columns=["Feature", "p_value"])

# Sort by p-value (ascending) and select top 10 features
p_value_df.sort_values(by=["p_value"], ascending=True, inplace=True)
selected_features = p_value_df["Feature"].head(10).tolist()

# Save p-value results to CSV
p_value_csv_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\Feature_p_values.csv"
p_value_df.to_csv(p_value_csv_path, index=False)

print("\nP-value computation completed, results saved to:", p_value_csv_path)
print("\nTop 10 selected features based on p-value:", selected_features)

# Step 3: Train the KNN model using only selected features
X_selected = df[selected_features]  # Use only selected features
y = df["Diagnosis"]

# Normalize selected features using StandardScaler
scaler = StandardScaler()
X_scaled_selected = scaler.fit_transform(X_selected)

# Split dataset (80% training, 20% testing)
X_train_sel, X_test_sel, y_train_sel, y_test_sel, id_train_sel, id_test_sel = train_test_split(
    X_scaled_selected, y, patient_ids, test_size=0.2, random_state=42
)

# Train KNN model on selected features
knn_selected = KNeighborsClassifier(n_neighbors=5)
knn_selected.fit(X_train_sel, y_train_sel)

# Make predictions using selected features
y_pred_sel = knn_selected.predict(X_test_sel)

# Save Actual vs Predicted Results to CSV (Selected Features)
results_df_selected = pd.DataFrame({
    "Patient ID": id_test_sel.values, 
    "Actual Diagnosis": y_test_sel.values, 
    "Predicted Diagnosis": y_pred_sel
})

# Define file path for saving predictions (Selected Features)
prediction_csv_path_sel = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\Prediction_vs_Actual_SelectedFeatures.csv"
results_df_selected.to_csv(prediction_csv_path_sel, index=False)

# Compute accuracy for selected features
accuracy_selected = accuracy_score(y_test_sel, y_pred_sel)

# Step 4: Print Summary
print("\n--- KNN Model Results With T-Test Feature Selection ---")
print(f"Accuracy using selected features (Top 10): {accuracy_selected:.4f}")
print(f"\nPredictions (Selected Features) saved to: {prediction_csv_path_sel}")
