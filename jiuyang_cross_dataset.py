#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load the dataset
file_path = 'a.csv'
data = pd.read_csv(file_path)

# Filter only the data with labels "ALS" and "HC"
filtered_data = data[data['label'].isin(['PD', 'HC'])]

# Separate training and testing data based on "Dataset" column
test_data = filtered_data[filtered_data['Dataset'] == 'PD_dataset_1']
train_data = filtered_data[filtered_data['Dataset'] == 'Italian']

# Separate features and labels for training and testing
X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Encode labels (ALS -> 1, HC -> 0)
y_train = y_train.map({'PD': 1, 'HC': 0})
y_test = y_test.map({'PD': 1, 'HC': 0})

# Drop irrelevant columns if they exist
irrelevant_columns = ['subjectID', 'file_path', 'voiced_file_path', 'Age', 'Sex', 'Severity', 'Phoneme', 'Dataset']
X_train = X_train.drop(columns=[col for col in irrelevant_columns if col in X_train.columns])
X_test = X_test.drop(columns=[col for col in irrelevant_columns if col in X_test.columns])

# One-hot encode categorical columns if they still exist
categorical_columns = ['subjectID', 'file_path', 'voiced_file_path', 'Age', 'Sex', 'Severity', 'Phoneme', 'Dataset']
categorical_columns_train = [col for col in categorical_columns if col in X_train.columns]
categorical_columns_test = [col for col in categorical_columns if col in X_test.columns]
X_train = pd.get_dummies(X_train, columns=categorical_columns_train, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_columns_test, drop_first=True)

# Align columns of X_train and X_test to ensure they have the same features
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Replace infinite values with NaN and then impute missing values
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Support Vector Machine Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Print classification reports and accuracy scores for both models
print("SVM Classifier Report:")
print(classification_report(y_test, y_pred_svm))
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")

print("\nRandom Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")


# In[ ]:




