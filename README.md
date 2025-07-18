# canine_disease
A machine learning-based project aimed at identifying and classifying common canine diseases using clinical data, symptoms, and image analysis. This repository contains datasets, preprocessing scripts, model training notebooks, and evaluation metrics to aid in early diagnosis and improve veterinary healthcare for dogs.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
# Replace 'canine_data.csv' with your actual file path
data = pd.read_csv('canine_data.csv')

# Assuming the last column is the target (disease label)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_test_scaled)

print("=== SVM Results ===")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Note: Random Forest doesn't need scaling necessarily
rf_preds = rf_model.predict(X_test)

print("=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# Example: Predict a new sample (replace with actual data)
# new_sample = np.array([[feature1, feature2, ..., featureN]])
# new_sample_scaled = scaler.transform(new_sample)
# prediction_svm = svm_model.predict(new_sample_scaled)
# prediction_rf = rf_model.predict(new_sample)
# print("SVM Prediction:", prediction_svm)
# print("Random Forest Prediction:", prediction_rf)
