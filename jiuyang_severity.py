#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = 'final_metadata_acoustic_features.csv'
df = pd.read_csv(file_path)

# Filter rows where label is 'ALS' and drop rows with missing Severity values
df = df[(df['label'] == 'ALS') & (~df['Severity'].isna())]

# Drop irrelevant features
irrelevant_columns = ['subjectID', 'file_path', 'voiced_file_path', 'Age', 'Sex', 'Phoneme', 'Dataset']  # Example irrelevant columns
df = df.drop(columns=irrelevant_columns, errors='ignore')

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Define features and target
target = 'Severity'
features = df.drop(columns=[target])
X = features
y = df[target]

# Handle missing values - dropping or imputing
X = X.dropna(axis=1, how='any')  # Drop features with missing values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (optional) - select top k features based on f_regression
k = 10  # Number of features to keep
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X_scaled, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train model - Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse}')


# In[ ]:




