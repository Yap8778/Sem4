import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Step 2: Select all features (excluding ID) and target variable
X = df.iloc[:, 2:32]  # Use all 30 features (without ID and Diagnosis)
y = df["Diagnosis"]    # Target variable (1 = Malignant, 0 = Benign)

# Step 3: Normalize all features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale all features

# Step 4: Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X_scaled, y, patient_ids, test_size=0.2, random_state=42
)

# Step 5: Train KNN model using all features
knn_all = KNeighborsClassifier(n_neighbors=5)
knn_all.fit(X_train, y_train)

# Step 6: Make predictions using all features
y_pred_all = knn_all.predict(X_test)

# Step 7: Save the prediction results to CSV (All Features)
results_df_all = pd.DataFrame({
    "Patient ID": id_test.values,
    "Actual Diagnosis": y_test.values,
    "Predicted Diagnosis": y_pred_all
})

# Define file path for saving predictions (All Features)
prediction_csv_path_all = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\Prediction_vs_Actual_AllFeatures.csv"
results_df_all.to_csv(prediction_csv_path_all, index=False)

# Step 8: Compute accuracy for all features
accuracy_all = accuracy_score(y_test, y_pred_all)

# Step 9: Print results
print("\n--- KNN Model Results Using All Features ---")
print(f"Accuracy using all features: {accuracy_all:.4f}")
print(f"\nPredictions (All Features) saved to: {prediction_csv_path_all}")
