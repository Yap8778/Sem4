import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# reading file as pandas data frame
file_path = "C:\\Users\\junso\\Downloads\\cirrhosis_cleaned.csv"
df_all = pd.read_csv(file_path)

# Keep ID as the identifier for comparison later
df_id = df_all.iloc[:, 0]  

# Select row and columns
# Features (Column)
df = df_all.iloc[:, 1:16]  
# Status, which is the prediction we want in result
df_label = df_all.iloc[:, 16]  

# Convert labels (row) to integers, to make sure the consistency 
y = np.ravel(df_label).astype(int)

# Retrieve the data in the feature column
X_raw = df.values
X_scaled = preprocessing.scale(X_raw)

# Get the amount of the feature 
num_of_features = X_scaled.shape[1]
p_values = np.ones(num_of_features)

# Compare first 126 samples (assumed deaf) vs remaining (assumed survived)
# T test and get p value
for i in range(num_of_features):
    feature_from_dead = X_scaled[0:126, i]
    feature_from_survived = X_scaled[126:, i]
    # Using t-test to calculate the p-value
    t_stat, p_value = stats.ttest_ind(feature_from_survived, feature_from_dead)
    # Store the p-value of each feature
    p_values[i] = p_value

# Create a DataFrame for p-values
p_value_df = pd.DataFrame({
    # Show the index number
    'feat_no': np.arange(num_of_features),
    # Use actual feature names
    'feature_name': df.columns,  
    # Record the p-value that calculated by t-test
    'p_value': p_values
})

# Sort the p value decending, it will show from high to low p-value
p_value_df.sort_values(by=['p_value'], ascending=False, inplace=True)  
p_value_df.reset_index(drop=True, inplace=True)

# Record the amount of remaining features
feat = num_of_features
# Store the index want to be deleted
cols_num_to_delete = []
X_current = X_scaled.copy()

# Record the highest model accuracy
best_accuracy = 0.0
# Store the best combination of features
best_features = []

# Keep delete one feature, until find the best feature combination
for i in range(num_of_features):
    # Split the data to test set (50%) and train set(50%)
    # Stratify is to make sure ratio remains unchanged to prevent abnormal distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_current, y, test_size=0.5, random_state=42, stratify=y
    )
    
    # 1st Random Forest: Find the feature combination with the highest accuracy
    # n_estimators = More decision trees can enhance model stability and accuracy, but enhance computation
    # max_depth = If max_depth is too large, the tree will over-learn the training data (overfitting)
    # If too small, the tree may not learn enough information (underfitting)
    # max_features = Randomly select square root of total number of features for judgment
    # class weight = Adjust category weights to fit the model
    # random_state = Ensure everytimes runs can create the same random forest model
    rf_final = RandomForestClassifier(n_estimators=500,max_depth=20,max_features="sqrt",class_weight="balanced",
    random_state=42)
    rf_final.fit(X_train, y_train)
    
    # Get predictions
    prediction = rf_final.predict(X_test)
    
    # Measure accuracy
    acc = accuracy_score(y_test, prediction)
    print(f"Iteration={i}, number_of_features={feat}, accuracy={acc:.4f}")
    
    # Track the best features with highest accuracy
    if acc > best_accuracy:
        best_accuracy = acc
        best_features = sorted(set(range(num_of_features)) - set(cols_num_to_delete))

    # Stop if there's only 1 feature left
    if feat == 1:
        break

    # Remove the feature with the highest p-value
    # Find the index of highest p-value
    worst_feature_col = p_value_df.loc[i, 'feat_no']
    cols_num_to_delete.append(worst_feature_col)
    # Delete that features from the dataset
    X_current = np.delete(X_scaled, cols_num_to_delete, axis=1)
    
    # Record the number of features remaining in the dataset
    feat = feat - 1

#  Selected the best features combination to become the final dataset used for training
X_filtered = X_scaled[:, best_features]

# Trainning class 70%, testing class 30%
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X_filtered, y, df_id, test_size=0.3, random_state=42, stratify=y
)

# 2nd Random Forest: Calculate the accuracy on the final test set
# Train final Random Forest model
rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X_train, y_train)

# Predict on the test set
# rf_final = The trained random forest model
y_pred_rf = rf_final.predict(X_test)

# Find the accuracy 
final_acc_rf = accuracy_score(y_test, y_pred_rf)

# Show actual vs predicted values along with ID
actual_vs_predicted_selectedFeatures = pd.DataFrame({"ID": id_test, "Actual": y_test, "Predicted": y_pred_rf})

# Acccuracy 
print(f"\nFinal test accuracy with feature selection: {final_acc_rf:.4f}")
 
# Train/test split (70/30) for all features
X_train_full, X_test_full, y_train_full, y_test_full, id_train_full, id_test_full = train_test_split(
    X_scaled, y, df_id, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest with all features
rf_final_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final_full.fit(X_train_full, y_train_full)

# Predict on the test set
y_pred_rf_full = rf_final_full.predict(X_test_full)
 
# Final accuracy without feature selection
final_acc_rf_full = accuracy_score(y_test_full, y_pred_rf_full)
 
# Show actual vs predicted values along with ID
actual_vs_predicted_allFeatures = pd.DataFrame({"ID": id_test_full, "Actual": y_test_full, "Predicted": y_pred_rf_full})
 
# Accuracy for full feature model
print(f"\nFinal test accuracy without feature selection: {final_acc_rf_full:.4f}")