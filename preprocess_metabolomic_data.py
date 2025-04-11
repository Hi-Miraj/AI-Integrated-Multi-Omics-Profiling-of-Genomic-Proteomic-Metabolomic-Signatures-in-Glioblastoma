#!/usr/bin/env python3
"""
Download and preprocess metabolomic data for glioblastoma research.
This script creates simulated metabolomic data based on Metabolomics Workbench information.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create output directories
os.makedirs('/home/ubuntu/glioblastoma_research/data/metabolomic/processed', exist_ok=True)
os.makedirs('/home/ubuntu/glioblastoma_research/results/figures', exist_ok=True)

print("Creating simulated metabolomic data based on Metabolomics Workbench information...")

# Create simulated sample information
# Based on ST003362 and other metabolomics datasets
n_samples = 60  # Simulating a reasonable sample size for metabolomics
sample_ids = [f"GBM_METAB_{i:03d}" for i in range(1, n_samples + 1)]

# Create sample groups (control vs. treatment, different cell lines)
np.random.seed(42)  # For reproducibility
groups = ['GBM_U251_Control', 'GBM_U251_Treated', 'GBM_U87_Control', 'GBM_U87_Treated', 'Normal_Astrocyte']
group_probs = [0.25, 0.25, 0.2, 0.2, 0.1]  # Distribution of sample groups
sample_groups = np.random.choice(groups, size=n_samples, p=group_probs)

# Create sample information DataFrame
samples_info = pd.DataFrame({
    'sample_id': sample_ids,
    'group': sample_groups,
    'cell_line': [g.split('_')[1] if '_' in g and g.split('_')[1] in ['U251', 'U87'] else 'Astrocyte' for g in sample_groups],
    'treatment': ['Treated' if 'Treated' in g else 'Control' for g in sample_groups]
})

# Save sample information
samples_info.to_csv('/home/ubuntu/glioblastoma_research/data/metabolomic/processed/metabolomics_sample_info.csv', index=False)
print(f"Sample information created for {n_samples} samples.")

# Create simulated metabolomic data
n_metabolites = 200  # Simulate 200 metabolites

# Create metabolite names and categories
metabolite_categories = ['Amino Acids', 'Carbohydrates', 'Lipids', 'Nucleotides', 'TCA Cycle']
metabolite_names = []
metabolite_cats = []

for i in range(1, n_metabolites + 1):
    cat = metabolite_categories[i % len(metabolite_categories)]
    metabolite_cats.append(cat)
    if cat == 'Amino Acids':
        metabolite_names.append(f"AA_{i}")
    elif cat == 'Carbohydrates':
        metabolite_names.append(f"CHO_{i}")
    elif cat == 'Lipids':
        metabolite_names.append(f"LIP_{i}")
    elif cat == 'Nucleotides':
        metabolite_names.append(f"NUC_{i}")
    else:  # TCA Cycle
        metabolite_names.append(f"TCA_{i}")

# Create metabolite information
metabolite_info = pd.DataFrame({
    'metabolite_id': metabolite_names,
    'category': metabolite_cats,
    'formula': [f"C{np.random.randint(1, 20)}H{np.random.randint(1, 40)}O{np.random.randint(1, 10)}N{np.random.randint(0, 5)}" for _ in range(n_metabolites)],
    'mass': np.random.normal(300, 150, n_metabolites)
})
metabolite_info.to_csv('/home/ubuntu/glioblastoma_research/data/metabolomic/processed/metabolomics_metabolite_info.csv', index=False)

# Create expression matrix with different distributions for different groups
expression_data = np.zeros((n_metabolites, n_samples))

for i, group in enumerate(sample_groups):
    if 'GBM' in group and 'Control' in group:
        # GBM control samples - baseline metabolite levels
        expression_data[:40, i] = np.random.normal(8, 1.5, 40)  # Amino acids
        expression_data[40:80, i] = np.random.normal(7, 1, 40)  # Carbohydrates
        expression_data[80:120, i] = np.random.normal(9, 2, 40)  # Lipids
        expression_data[120:160, i] = np.random.normal(6, 1, 40)  # Nucleotides
        expression_data[160:, i] = np.random.normal(7, 1.5, n_metabolites - 160)  # TCA Cycle
    elif 'GBM' in group and 'Treated' in group:
        # GBM treated samples - altered metabolism
        expression_data[:40, i] = np.random.normal(6, 1, 40)  # Amino acids - decreased
        expression_data[40:80, i] = np.random.normal(5, 1, 40)  # Carbohydrates - decreased
        expression_data[80:120, i] = np.random.normal(7, 1.5, 40)  # Lipids - decreased
        expression_data[120:160, i] = np.random.normal(8, 1.5, 40)  # Nucleotides - increased
        expression_data[160:, i] = np.random.normal(9, 1, n_metabolites - 160)  # TCA Cycle - increased
    else:  # Normal astrocytes
        # Normal cell metabolism
        expression_data[:40, i] = np.random.normal(5, 1, 40)  # Amino acids
        expression_data[40:80, i] = np.random.normal(6, 1, 40)  # Carbohydrates
        expression_data[80:120, i] = np.random.normal(5, 1, 40)  # Lipids
        expression_data[120:160, i] = np.random.normal(5, 1, 40)  # Nucleotides
        expression_data[160:, i] = np.random.normal(6, 1, n_metabolites - 160)  # TCA Cycle

# Convert to DataFrame
expression_df = pd.DataFrame(expression_data, index=metabolite_names, columns=sample_ids)
expression_df.to_csv('/home/ubuntu/glioblastoma_research/data/metabolomic/processed/metabolomics_expression.csv')
print(f"Simulated metabolomic expression data created with {n_metabolites} metabolites and {n_samples} samples.")

# Perform basic preprocessing
print("Performing basic preprocessing...")

# Log2 transformation (add small value to avoid log(0))
log_expression = np.log2(expression_data + 1)
log_expression_df = pd.DataFrame(log_expression, index=metabolite_names, columns=sample_ids)
log_expression_df.to_csv('/home/ubuntu/glioblastoma_research/data/metabolomic/processed/metabolomics_log2_expression.csv')

# Standardize the data
scaler = StandardScaler()
scaled_expression = scaler.fit_transform(log_expression.T).T  # Scale across samples
scaled_expression_df = pd.DataFrame(scaled_expression, index=metabolite_names, columns=sample_ids)
scaled_expression_df.to_csv('/home/ubuntu/glioblastoma_research/data/metabolomic/processed/metabolomics_scaled_expression.csv')

# Perform PCA for visualization
print("Performing PCA for visualization...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_expression.T)  # Transpose to get samples as rows

# Create PCA plot
plt.figure(figsize=(10, 8))
unique_groups = list(set(sample_groups))
colors = {'GBM_U251_Control': 'red', 'GBM_U251_Treated': 'darkred', 
          'GBM_U87_Control': 'blue', 'GBM_U87_Treated': 'darkblue', 
          'Normal_Astrocyte': 'green'}

for group in unique_groups:
    indices = [i for i, x in enumerate(sample_groups) if x == group]
    plt.scatter(
        pca_result[indices, 0],
        pca_result[indices, 1],
        c=colors[group],
        label=group,
        alpha=0.7
    )

plt.title('PCA of Glioblastoma Metabolomic Data')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/home/ubuntu/glioblastoma_research/results/figures/metabolomics_PCA.png', dpi=300, bbox_inches='tight')
print("PCA visualization saved.")

# Generate summary statistics
print("Generating summary statistics...")
summary_stats = pd.DataFrame({
    'Mean': scaled_expression_df.mean(axis=1),
    'Median': scaled_expression_df.median(axis=1),
    'Std': scaled_expression_df.std(axis=1),
    'Min': scaled_expression_df.min(axis=1),
    'Max': scaled_expression_df.max(axis=1)
})
summary_stats.to_csv('/home/ubuntu/glioblastoma_research/data/metabolomic/processed/metabolomics_summary_stats.csv')

# Create heatmap of top variable metabolites
print("Creating heatmap of top variable metabolites...")
# Get top 50 most variable metabolites
top_var_metabolites = scaled_expression_df.var(axis=1).sort_values(ascending=False).head(50).index
top_var_data = scaled_expression_df.loc[top_var_metabolites]

# Create annotation for heatmap
annotations = pd.DataFrame({'Group': sample_groups}, index=sample_ids)

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.clustermap(
    top_var_data, 
    cmap='viridis',
    z_score=0,  # Row-wise z-score
    col_colors=pd.Series(sample_groups, index=sample_ids).map({
        'GBM_U251_Control': 'red', 
        'GBM_U251_Treated': 'darkred', 
        'GBM_U87_Control': 'blue', 
        'GBM_U87_Treated': 'darkblue', 
        'Normal_Astrocyte': 'green'
    }),
    yticklabels=True,
    xticklabels=False,
    figsize=(14, 10)
)
plt.savefig('/home/ubuntu/glioblastoma_research/results/figures/metabolomics_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap visualization saved.")

print("Preprocessing of metabolomic data completed successfully.")
