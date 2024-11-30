#!/usr/bin/env python
# coding: utf-8

# In[10]:


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

# Filter only the data with labels "PD" and "HC"
filtered_data = data[data['label'].isin(['PD', 'HC'])]

# Separate features and labels
X = filtered_data.drop(columns=['label'])
y = filtered_data['label']

# Encode labels (PD -> 1, HC -> 0)
y = y.map({'PD': 1, 'HC': 0})

# Drop irrelevant columns if they exist
irrelevant_columns = ['subjectID', 'file_path', 'voiced_file_path', 'Age', 'Sex', 'Severity', 'Phoneme', 'Dataset']
X = X.drop(columns=[col for col in irrelevant_columns if col in X.columns])

# One-hot encode categorical columns if they still exist
categorical_columns = ['Sex', 'Phoneme', 'Dataset']
categorical_columns = [col for col in categorical_columns if col in X.columns]
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Replace infinite values with NaN and then impute missing values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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




