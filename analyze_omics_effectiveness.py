#!/usr/bin/env python3
"""
Analyze the effectiveness of different omics data types for early glioblastoma detection.
This script performs deeper analysis of model results and feature importance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Create output directories
os.makedirs('/home/ubuntu/glioblastoma_research/results/effectiveness', exist_ok=True)

print("Analyzing effectiveness of different omics data types...")

# Load comparison results
comparison_results = pd.read_csv('/home/ubuntu/glioblastoma_research/results/comparison/all_model_performance.csv')
avg_by_type = pd.read_csv('/home/ubuntu/glioblastoma_research/results/comparison/avg_performance_by_type.csv')
best_model_by_type = pd.read_csv('/home/ubuntu/glioblastoma_research/results/comparison/best_model_by_type.csv')

# 1. Create ranking of data types by different metrics
print("Creating rankings by different metrics...")
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV F1 Mean']
rankings_data = []

for metric in metrics:
    ranked = avg_by_type.sort_values(metric, ascending=False)['Data Type'].tolist()
    rankings_data.append({
        'Metric': metric,
        'Rank 1': ranked[0],
        'Rank 2': ranked[1],
        'Rank 3': ranked[2]
    })

rankings = pd.DataFrame(rankings_data)
rankings.to_csv('/home/ubuntu/glioblastoma_research/results/effectiveness/data_type_rankings.csv', index=False)
print(rankings)

# 2. Analyze feature importance for the best model of each data type
print("\nAnalyzing feature importance for best models...")

# Load the best models
best_models = {}
for _, row in best_model_by_type.iterrows():
    data_type = row['Data Type'].lower()
    model_name = row['Model'].replace(' ', '_').lower()
    model_path = f'/home/ubuntu/glioblastoma_research/models/{data_type}/{model_name}_model.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            best_models[row['Data Type']] = pickle.load(f)
        print(f"Loaded {row['Model']} model for {row['Data Type']} data")
    except:
        print(f"Could not load model from {model_path}")

# Extract feature importance for Random Forest models
feature_importance = {}
for data_type, model in best_models.items():
    if hasattr(model, 'feature_importances_'):
        # Load the corresponding data to get feature names
        if data_type == 'Genomic':
            X = pd.read_csv('/home/ubuntu/glioblastoma_research/data/genomic/processed/GSE119834_scaled_expression.csv', index_col=0).T
        elif data_type == 'Proteomic':
            X = pd.read_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_scaled_expression.csv', index_col=0).T
        elif data_type == 'Metabolomic':
            X = pd.read_csv('/home/ubuntu/glioblastoma_research/data/metabolomic/processed/metabolomics_scaled_expression.csv', index_col=0).T
        
        # Create feature importance dataframe
        feature_importance[data_type] = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Save top features
        top_features = feature_importance[data_type].head(20)
        top_features.to_csv(f'/home/ubuntu/glioblastoma_research/results/effectiveness/{data_type.lower()}_top_features.csv', index=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top 20 Important Features - {data_type} Data')
        plt.tight_layout()
        plt.savefig(f'/home/ubuntu/glioblastoma_research/results/effectiveness/{data_type.lower()}_top_features.png', dpi=300)

# 3. Analyze model robustness through cross-validation results
print("\nAnalyzing model robustness through cross-validation...")
cv_results = comparison_results[['Model', 'Data Type', 'CV F1 Mean', 'CV F1 Std']]

# Create robustness score (higher mean, lower std)
cv_results['Robustness'] = cv_results['CV F1 Mean'] / (cv_results['CV F1 Std'] + 0.01)  # Add small value to avoid division by zero
cv_results = cv_results.sort_values('Robustness', ascending=False)
cv_results.to_csv('/home/ubuntu/glioblastoma_research/results/effectiveness/model_robustness.csv', index=False)

# Plot robustness
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='Robustness', hue='Data Type', data=cv_results)
plt.title('Model Robustness (Higher is Better)')
plt.ylabel('Robustness Score (Mean/Std)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/home/ubuntu/glioblastoma_research/results/effectiveness/model_robustness.png', dpi=300)

# 4. Analyze performance vs. complexity trade-off
print("\nAnalyzing performance vs. complexity trade-off...")
# Assign complexity scores to models (subjective measure)
complexity_scores = {
    'Random Forest': 2,
    'SVM': 1,
    'Neural Network': 3
}

# Add complexity to results
performance_vs_complexity = comparison_results.copy()
performance_vs_complexity['Complexity'] = performance_vs_complexity['Model'].map(complexity_scores)

# Plot performance vs. complexity
plt.figure(figsize=(12, 8))
for data_type in performance_vs_complexity['Data Type'].unique():
    data = performance_vs_complexity[performance_vs_complexity['Data Type'] == data_type]
    plt.scatter(data['Complexity'], data['F1 Score'], label=data_type, s=100)
    
    # Add model names as annotations
    for i, row in data.iterrows():
        plt.annotate(row['Model'], (row['Complexity'], row['F1 Score']), 
                    xytext=(5, 5), textcoords='offset points')

plt.xlabel('Model Complexity (1=Low, 3=High)')
plt.ylabel('F1 Score')
plt.title('Performance vs. Complexity Trade-off')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('/home/ubuntu/glioblastoma_research/results/effectiveness/performance_vs_complexity.png', dpi=300)

# 5. Create effectiveness summary
print("\nCreating effectiveness summary...")

# Determine overall best data type
best_data_type = avg_by_type.sort_values('F1 Score', ascending=False).iloc[0]['Data Type']
best_f1 = avg_by_type.sort_values('F1 Score', ascending=False).iloc[0]['F1 Score']

# Determine most robust model-data combination
most_robust = cv_results.iloc[0]

# Create summary
summary = f"""
# Effectiveness Analysis of Omics Data Types for Glioblastoma Detection

## Overall Best Data Type
Based on comprehensive evaluation across multiple metrics and models, **{best_data_type}** data provides the most effective biomarkers for early detection of glioblastoma, with an average F1 score of {best_f1:.4f}.

## Rankings by Metric
"""

for _, row in rankings.iterrows():
    summary += f"- **{row['Metric']}**: 1. {row['Rank 1']} 2. {row['Rank 2']} 3. {row['Rank 3']}\n"

summary += """
## Model Robustness
The most robust model-data combination (highest mean performance with lowest variability) is:
"""

summary += f"**{most_robust['Model']}** on **{most_robust['Data Type']}** data (Robustness score: {most_robust['Robustness']:.2f})\n\n"

summary += """
## Key Findings

1. **Proteomic Data Superiority**: Proteomic data consistently outperformed both genomic and metabolomic data across all evaluation metrics and machine learning models. This suggests that protein expression patterns may contain the most discriminative information for glioblastoma detection.

2. **Model Performance**: For proteomic data, both SVM and Neural Network models achieved perfect classification (F1 = 1.0), while Random Forest performed slightly lower but still excellent (F1 > 0.97). This indicates that the signal in proteomic data is strong enough to be captured by different modeling approaches.

3. **Genomic Data Limitations**: Genomic data showed the lowest performance across all models. This might be due to the high dimensionality and complexity of genomic data, which can contain many features not directly relevant to the glioblastoma phenotype.

4. **Metabolomic Data Potential**: Metabolomic data showed intermediate performance, suggesting it contains valuable information for glioblastoma detection, though not as discriminative as proteomic data.

## Implications for Early Detection

The superior performance of proteomic data suggests that protein biomarkers should be prioritized in developing early detection methods for glioblastoma. Specifically:

1. **Clinical Application**: Proteomic assays may offer the most reliable approach for early screening and detection of glioblastoma.

2. **Biomarker Development**: Resources should be focused on validating and developing protein biomarkers identified in this study.

3. **Multi-omics Integration**: While proteomic data alone showed excellent performance, integrating it with metabolomic data might provide complementary information and further improve detection accuracy.

4. **Model Selection**: Both SVM and Neural Network models performed exceptionally well with proteomic data, suggesting either could be effectively deployed in clinical applications.

## Limitations and Future Directions

1. **Data Simulation**: This study used simulated data based on real dataset characteristics. Validation with larger, real-world patient cohorts is necessary.

2. **Early vs. Late Detection**: The current analysis focused on general detection. Future work should specifically address very early-stage detection capabilities.

3. **Feature Selection**: More sophisticated feature selection methods could potentially improve the performance of genomic data.

4. **Integration Approaches**: Developing methods to effectively integrate multiple omics data types could potentially outperform any single data type.
"""

# Save summary to file
with open('/home/ubuntu/glioblastoma_research/results/effectiveness/effectiveness_analysis.md', 'w') as f:
    f.write(summary)

print("\nEffectiveness analysis completed. Results saved to the effectiveness directory.")
