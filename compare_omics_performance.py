#!/usr/bin/env python3
"""
Compare model performance across genomic, proteomic, and metabolomic data types
for early detection of glioblastoma.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directories
os.makedirs('/home/ubuntu/glioblastoma_research/results/comparison', exist_ok=True)

print("Loading model performance results for all omics data types...")

# Load performance metrics for each data type
genomic_results = pd.read_csv('/home/ubuntu/glioblastoma_research/results/genomic/model_performance.csv')
proteomic_results = pd.read_csv('/home/ubuntu/glioblastoma_research/results/proteomic/model_performance.csv')
metabolomic_results = pd.read_csv('/home/ubuntu/glioblastoma_research/results/metabolomic/model_performance.csv')

# Add data type column to each dataframe
genomic_results['Data Type'] = 'Genomic'
proteomic_results['Data Type'] = 'Proteomic'
metabolomic_results['Data Type'] = 'Metabolomic'

# Combine all results
all_results = pd.concat([genomic_results, proteomic_results, metabolomic_results], ignore_index=True)

print("Combined results:")
print(all_results)

# Save combined results
all_results.to_csv('/home/ubuntu/glioblastoma_research/results/comparison/all_model_performance.csv', index=False)

# Create comparison visualizations
print("Creating comparison visualizations...")

# 1. Bar chart comparing F1 scores across data types and models
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='F1 Score', hue='Data Type', data=all_results)
plt.title('F1 Score Comparison Across Omics Data Types')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/home/ubuntu/glioblastoma_research/results/comparison/f1_score_comparison.png', dpi=300, bbox_inches='tight')

# 2. Bar chart comparing accuracy across data types and models
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='Accuracy', hue='Data Type', data=all_results)
plt.title('Accuracy Comparison Across Omics Data Types')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/home/ubuntu/glioblastoma_research/results/comparison/accuracy_comparison.png', dpi=300, bbox_inches='tight')

# 3. Heatmap of all metrics across data types and models
# Reshape data for heatmap
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV F1 Mean']
heatmap_data = []

for metric in metrics:
    for model in all_results['Model'].unique():
        row = {'Metric': metric, 'Model': model}
        for data_type in all_results['Data Type'].unique():
            value = all_results[(all_results['Model'] == model) & (all_results['Data Type'] == data_type)][metric].values[0]
            row[data_type] = value
        heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data)
heatmap_df.to_csv('/home/ubuntu/glioblastoma_research/results/comparison/metrics_by_model_and_type.csv', index=False)

# Create pivot table for heatmap
heatmap_pivot = pd.pivot_table(
    all_results, 
    values='F1 Score', 
    index=['Data Type'], 
    columns=['Model']
)

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_pivot, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.4f')
plt.title('F1 Score Heatmap by Data Type and Model')
plt.tight_layout()
plt.savefig('/home/ubuntu/glioblastoma_research/results/comparison/f1_heatmap.png', dpi=300, bbox_inches='tight')

# 4. Radar chart for comparing all metrics across data types
# Prepare data for radar chart
def create_radar_chart(data_type):
    # Filter data for the specific data type
    data = all_results[all_results['Data Type'] == data_type]
    
    # Get metrics for each model
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV F1 Mean']
    models = data['Model'].unique()
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    for model in models:
        model_data = data[data['Model'] == model][metrics].values[0].tolist()
        model_data += model_data[:1]  # Close the loop
        ax.plot(angles, model_data, linewidth=2, label=model)
        ax.fill(angles, model_data, alpha=0.1)
    
    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f'{data_type} Data - Model Performance Metrics')
    ax.set_ylim(0, 1.05)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/glioblastoma_research/results/comparison/{data_type.lower()}_radar.png', dpi=300, bbox_inches='tight')

# Create radar charts for each data type
for data_type in all_results['Data Type'].unique():
    create_radar_chart(data_type)

# 5. Calculate average performance across models for each data type
avg_by_type = all_results.groupby('Data Type')[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV F1 Mean']].mean().reset_index()
print("\nAverage performance by data type:")
print(avg_by_type)
avg_by_type.to_csv('/home/ubuntu/glioblastoma_research/results/comparison/avg_performance_by_type.csv', index=False)

# Create bar chart of average performance by data type
plt.figure(figsize=(12, 8))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV F1 Mean']
avg_by_type_melted = pd.melt(avg_by_type, id_vars=['Data Type'], value_vars=metrics_to_plot, var_name='Metric', value_name='Value')

sns.barplot(x='Data Type', y='Value', hue='Metric', data=avg_by_type_melted)
plt.title('Average Performance Metrics by Omics Data Type')
plt.xlabel('Data Type')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/home/ubuntu/glioblastoma_research/results/comparison/avg_performance_by_type.png', dpi=300, bbox_inches='tight')

# 6. Determine the best performing data type based on F1 score
best_data_type = avg_by_type.sort_values('F1 Score', ascending=False).iloc[0]['Data Type']
best_f1 = avg_by_type.sort_values('F1 Score', ascending=False).iloc[0]['F1 Score']
print(f"\nBest performing data type based on average F1 score: {best_data_type} (F1 = {best_f1:.4f})")

# 7. Determine the best model for each data type
best_model_by_type = all_results.loc[all_results.groupby('Data Type')['F1 Score'].idxmax()][['Data Type', 'Model', 'F1 Score']]
print("\nBest model for each data type:")
print(best_model_by_type)
best_model_by_type.to_csv('/home/ubuntu/glioblastoma_research/results/comparison/best_model_by_type.csv', index=False)

# Create summary of findings
summary = f"""
# Summary of Comparative Analysis

## Best Performing Omics Data Type
Based on the average F1 score across all models, the **{best_data_type}** data type provides the most effective biomarkers for early detection of glioblastoma, with an average F1 score of {best_f1:.4f}.

## Best Models by Data Type
"""

for _, row in best_model_by_type.iterrows():
    summary += f"- **{row['Data Type']}**: {row['Model']} (F1 = {row['F1 Score']:.4f})\n"

summary += """
## Overall Performance Comparison
"""

for _, row in avg_by_type.iterrows():
    summary += f"- **{row['Data Type']}**: Accuracy = {row['Accuracy']:.4f}, F1 Score = {row['F1 Score']:.4f}, CV F1 Mean = {row['CV F1 Mean']:.4f}\n"

summary += """
## Conclusion
"""

# Add conclusion based on results
data_types_ranked = avg_by_type.sort_values('F1 Score', ascending=False)['Data Type'].tolist()
summary += f"The comparative analysis shows that {data_types_ranked[0]} data provides the most reliable biomarkers for early detection of glioblastoma, followed by {data_types_ranked[1]} and {data_types_ranked[2]} data. "

if best_data_type == 'Proteomic':
    summary += "Proteomic data likely captures the functional state of tumor cells more effectively, providing more reliable biomarkers for early detection."
elif best_data_type == 'Genomic':
    summary += "Genomic data likely captures the underlying genetic alterations driving glioblastoma development, providing more reliable biomarkers for early detection."
elif best_data_type == 'Metabolomic':
    summary += "Metabolomic data likely captures the altered metabolic state of tumor cells, providing more reliable biomarkers for early detection."

# Save summary to file
with open('/home/ubuntu/glioblastoma_research/results/comparison/summary_of_findings.md', 'w') as f:
    f.write(summary)

print("\nComparison analysis completed. Results saved to the comparison directory.")
