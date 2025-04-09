import pandas as pd
import os  # Import os to check file size
from ucimlrepo import fetch_ucirepo 

# Fetch dataset from UCI dictionary
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

# Extract data (features and values)
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 

# Define file path for the local dataset
file_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\wdbc.data"

# Extract full feature names from the dataset metadata
# Get actual feature names
feature_names = list(X.columns)  
# Add ID & Diagnosis
column_names = ["ID", "Diagnosis"] + feature_names  

# Read the dataset without headers (since wdbc.data has no column headers)
Complete_dataset = pd.read_csv(file_path, header=None, names=column_names)

# Save the dataset with full feature names
csv_file_path = r"C:\\Users\\junso\\Documents\\DH\\SEM 4\\Data Mining\\breast+cancer+wisconsin+diagnostic\\Complete_dataset.csv"
Complete_dataset.to_csv(csv_file_path, index=False)

# Get the file size in KB to decide need to use feather function to compress file or not
file_size = os.path.getsize(csv_file_path) / 1024  # Convert bytes to KB

# Display the first 5 rows of the complete dataset
print("-------------Show some examples--------------")
print("")
print(Complete_dataset.head())
print(f"\nDataset saved successfully as: {csv_file_path}")
print(f"\nFile size of normal version: {file_size:.2f} KB")
