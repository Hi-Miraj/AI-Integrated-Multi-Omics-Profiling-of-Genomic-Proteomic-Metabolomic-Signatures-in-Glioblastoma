#!/usr/bin/env python3
"""
Implement predictive models for glioblastoma detection using proteomic data.
This script trains Random Forest, SVM, and Neural Network models on the preprocessed proteomic data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# Create output directories
os.makedirs('/home/ubuntu/glioblastoma_research/models/proteomic', exist_ok=True)
os.makedirs('/home/ubuntu/glioblastoma_research/results/proteomic', exist_ok=True)

print("Loading preprocessed proteomic data...")
# Load the preprocessed data
expression_df = pd.read_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_scaled_expression.csv', index_col=0)
samples_info = pd.read_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_sample_info.csv')

# Prepare data for modeling
X = expression_df.T  # Transpose to get samples as rows
y = samples_info['subtype']  # Use the subtype column for classification

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
print(f"Class distribution: {pd.Series(y).value_counts()}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("Training and evaluating Random Forest model...")
# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Random Forest - Accuracy: {accuracy_rf:.4f}, Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, F1: {f1_rf:.4f}")

# Feature importance for Random Forest
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
top_features = feature_importances.sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 20 Important Features - Random Forest (Proteomic Data)')
plt.tight_layout()
plt.savefig('/home/ubuntu/glioblastoma_research/results/proteomic/rf_feature_importance.png', dpi=300)

print("Training and evaluating SVM model...")
# SVM model
svm_model = SVC(probability=True, class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)

# Evaluate SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

print(f"SVM - Accuracy: {accuracy_svm:.4f}, Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}, F1: {f1_svm:.4f}")

print("Training and evaluating Neural Network model...")
# Neural Network model
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
y_prob_nn = nn_model.predict_proba(X_test)

# Evaluate Neural Network model
accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn, average='weighted')
recall_nn = recall_score(y_test, y_pred_nn, average='weighted')
f1_nn = f1_score(y_test, y_pred_nn, average='weighted')

print(f"Neural Network - Accuracy: {accuracy_nn:.4f}, Precision: {precision_nn:.4f}, Recall: {recall_nn:.4f}, F1: {f1_nn:.4f}")

# Perform stratified cross-validation for all models
print("Performing stratified cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest CV
rf_cv_scores = cross_val_score(rf_model, X, y_encoded, cv=cv, scoring='f1_weighted')
print(f"Random Forest CV F1 scores: {rf_cv_scores}")
print(f"Random Forest CV F1 mean: {rf_cv_scores.mean():.4f}, std: {rf_cv_scores.std():.4f}")

# SVM CV
svm_cv_scores = cross_val_score(svm_model, X, y_encoded, cv=cv, scoring='f1_weighted')
print(f"SVM CV F1 scores: {svm_cv_scores}")
print(f"SVM CV F1 mean: {svm_cv_scores.mean():.4f}, std: {svm_cv_scores.std():.4f}")

# Neural Network CV
nn_cv_scores = cross_val_score(nn_model, X, y_encoded, cv=cv, scoring='f1_weighted')
print(f"Neural Network CV F1 scores: {nn_cv_scores}")
print(f"Neural Network CV F1 mean: {nn_cv_scores.mean():.4f}, std: {nn_cv_scores.std():.4f}")

# Create confusion matrices
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Random Forest Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 3, 2)
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 3, 3)
cm_nn = confusion_matrix(y_test, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Neural Network Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('/home/ubuntu/glioblastoma_research/results/proteomic/confusion_matrices.png', dpi=300)

# Create ROC curves for multi-class
plt.figure(figsize=(10, 8))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# For each model
models = {
    'Random Forest': y_prob_rf,
    'SVM': y_prob_svm,
    'Neural Network': y_prob_nn
}

colors = ['blue', 'red', 'green']
for i, (model_name, y_prob) in enumerate(models.items()):
    # Compute micro-average ROC curve and ROC area
    fpr[model_name], tpr[model_name], _ = roc_curve(
        np.eye(len(class_names))[y_test].ravel(), 
        y_prob.ravel()
    )
    roc_auc[model_name] = auc(fpr[model_name], tpr[model_name])
    
    plt.plot(
        fpr[model_name], 
        tpr[model_name], 
        color=colors[i],
        lw=2, 
        label=f'{model_name} (AUC = {roc_auc[model_name]:.2f})'
    )

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Proteomic Data Models')
plt.legend(loc="lower right")
plt.savefig('/home/ubuntu/glioblastoma_research/results/proteomic/roc_curves.png', dpi=300)

# Save model performance metrics
results = pd.DataFrame({
    'Model': ['Random Forest', 'SVM', 'Neural Network'],
    'Accuracy': [accuracy_rf, accuracy_svm, accuracy_nn],
    'Precision': [precision_rf, precision_svm, precision_nn],
    'Recall': [recall_rf, recall_svm, recall_nn],
    'F1 Score': [f1_rf, f1_svm, f1_nn],
    'CV F1 Mean': [rf_cv_scores.mean(), svm_cv_scores.mean(), nn_cv_scores.mean()],
    'CV F1 Std': [rf_cv_scores.std(), svm_cv_scores.std(), nn_cv_scores.std()]
})

results.to_csv('/home/ubuntu/glioblastoma_research/results/proteomic/model_performance.csv', index=False)
print("Results saved to CSV file.")

# Save models
import pickle
with open('/home/ubuntu/glioblastoma_research/models/proteomic/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('/home/ubuntu/glioblastoma_research/models/proteomic/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
with open('/home/ubuntu/glioblastoma_research/models/proteomic/nn_model.pkl', 'wb') as f:
    pickle.dump(nn_model, f)

print("Models saved successfully.")
print("Proteomic data modeling completed.")
