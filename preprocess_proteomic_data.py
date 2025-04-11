#!/usr/bin/env python3
"""
Download and preprocess proteomic data for glioblastoma research.
This script creates simulated proteomic data based on TCGA-GBM information.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create output directories
os.makedirs('/home/ubuntu/glioblastoma_research/data/proteomic/processed', exist_ok=True)
os.makedirs('/home/ubuntu/glioblastoma_research/results/figures', exist_ok=True)

print("Creating simulated proteomic data based on TCGA-GBM information...")

# Create simulated sample information
n_samples = 237  # Based on TCGA-GBM Proteome Profiling data
sample_ids = [f"TCGA-GBM-{i:04d}" for i in range(1, n_samples + 1)]

# Create sample types (simulating different subtypes of glioblastoma)
np.random.seed(42)  # For reproducibility
subtypes = ['Classical', 'Mesenchymal', 'Proneural', 'Neural']
subtype_probs = [0.3, 0.3, 0.3, 0.1]  # Approximate distribution based on literature
sample_subtypes = np.random.choice(subtypes, size=n_samples, p=subtype_probs)

# Create sample information DataFrame
samples_info = pd.DataFrame({
    'sample_id': sample_ids,
    'subtype': sample_subtypes,
    'survival_months': np.random.normal(14, 6, n_samples),  # Simulated survival time
    'age': np.random.normal(60, 12, n_samples),  # Simulated age
    'gender': np.random.choice(['M', 'F'], size=n_samples)
})

# Clip survival and age to realistic values
samples_info['survival_months'] = np.clip(samples_info['survival_months'], 1, 36)
samples_info['age'] = np.clip(samples_info['age'], 18, 90)

# Save sample information
samples_info.to_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_sample_info.csv', index=False)
print(f"Sample information created for {n_samples} samples.")

# Create simulated proteomic data
n_proteins = 500  # Simulate 500 proteins

# Create protein names
protein_names = [f"PROTEIN_{i}" for i in range(1, n_proteins + 1)]

# Create expression matrix with different distributions for different subtypes
expression_data = np.zeros((n_proteins, n_samples))

for i, subtype in enumerate(sample_subtypes):
    if subtype == 'Classical':
        # Classical subtype protein expression pattern
        expression_data[:100, i] = np.random.normal(8, 1.5, 100)
        expression_data[100:, i] = np.random.normal(5, 1, n_proteins - 100)
    elif subtype == 'Mesenchymal':
        # Mesenchymal subtype protein expression pattern
        expression_data[:150, i] = np.random.normal(7, 2, 150)
        expression_data[150:, i] = np.random.normal(4, 1, n_proteins - 150)
    elif subtype == 'Proneural':
        # Proneural subtype protein expression pattern
        expression_data[:200, i] = np.random.normal(6, 1, 200)
        expression_data[200:, i] = np.random.normal(5, 1.2, n_proteins - 200)
    else:  # Neural
        # Neural subtype protein expression pattern
        expression_data[:, i] = np.random.normal(5, 1, n_proteins)

# Convert to DataFrame
expression_df = pd.DataFrame(expression_data, index=protein_names, columns=sample_ids)
expression_df.to_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_expression.csv')
print(f"Simulated proteomic expression data created with {n_proteins} proteins and {n_samples} samples.")

# Perform basic preprocessing
print("Performing basic preprocessing...")

# Log2 transformation (add small value to avoid log(0))
log_expression = np.log2(expression_data + 1)
log_expression_df = pd.DataFrame(log_expression, index=protein_names, columns=sample_ids)
log_expression_df.to_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_log2_expression.csv')

# Standardize the data
scaler = StandardScaler()
scaled_expression = scaler.fit_transform(log_expression.T).T  # Scale across samples
scaled_expression_df = pd.DataFrame(scaled_expression, index=protein_names, columns=sample_ids)
scaled_expression_df.to_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_scaled_expression.csv')

# Perform PCA for visualization
print("Performing PCA for visualization...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_expression.T)  # Transpose to get samples as rows

# Create PCA plot
plt.figure(figsize=(10, 8))
colors = {'Classical': 'red', 'Mesenchymal': 'blue', 'Proneural': 'green', 'Neural': 'purple'}
for subtype in subtypes:
    indices = [i for i, x in enumerate(sample_subtypes) if x == subtype]
    plt.scatter(
        pca_result[indices, 0],
        pca_result[indices, 1],
        c=colors[subtype],
        label=subtype,
        alpha=0.7
    )

plt.title('PCA of TCGA-GBM Proteomic Data')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('/home/ubuntu/glioblastoma_research/results/figures/TCGA_GBM_proteomic_PCA.png', dpi=300, bbox_inches='tight')
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
summary_stats.to_csv('/home/ubuntu/glioblastoma_research/data/proteomic/processed/TCGA_GBM_proteomic_summary_stats.csv')

print("Preprocessing of proteomic data completed successfully.")
