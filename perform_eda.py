import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
import os

# CONFIG
RAW_DIR = "data_raw"
PROC_DIR = "data_processed"

# 1. Load Data
print("Loading data for EDA...")
# Load raw terms for frequency analysis
df_terms = pd.read_csv(f"{RAW_DIR}/train_terms.tsv", sep='\t')

# Load processed targets for matrix analysis
Y_sparse = load_npz(f"{PROC_DIR}/Y_sparse.npz")
term_ids = np.load(f"{PROC_DIR}/term_ids.npy", allow_pickle=True)

# 2. PLOT 1: The Class Imbalance (Top 20 Functions)
print("Generating Class Distribution Plot...")
plt.figure(figsize=(12, 6))
term_counts = df_terms['term'].value_counts().head(20)
sns.barplot(x=term_counts.values, y=term_counts.index, palette="viridis")
plt.title("Top 20 Most Frequent Gene Ontology Terms")
plt.xlabel("Number of Proteins")
plt.ylabel("GO Term ID")
plt.tight_layout()
plt.savefig("eda_top_20_terms.png") # Saves image for your report
print("Saved 'eda_top_20_terms.png'")
plt.show()

# 3. PLOT 2: Sequence Length Distribution
print("Generating Sequence Length Histogram...")
lengths = []
with open(f"{RAW_DIR}/train_sequences.fasta", 'r') as f:
    current_len = 0
    for line in f:
        if line.startswith(">"):
            if current_len > 0: lengths.append(current_len)
            current_len = 0
        else:
            current_len += len(line.strip())
    lengths.append(current_len) # Add last one

plt.figure(figsize=(10, 5))
sns.histplot(lengths, bins=50, log_scale=(True, False), color="teal")
plt.title("Distribution of Protein Sequence Lengths")
plt.xlabel("Length (Log Scale)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_sequence_lengths.png")
print("Saved 'eda_sequence_lengths.png'")
plt.show()

# 4. PLOT 3: Label Co-Occurrence (Heatmap)
# We check if the Top 20 terms appear together
print("Generating Correlation Heatmap (Top 20)...")
# Get dense matrix of just top 20 columns
Y_top20 = Y_sparse[:, :20].toarray()
corr_matrix = np.corrcoef(Y_top20, rowvar=False)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, xticklabels=term_ids[:20], yticklabels=term_ids[:20], cmap="coolwarm", annot=False)
plt.title("Correlation Between Top 20 Functions")
plt.tight_layout()
plt.savefig("eda_correlation.png")
print("Saved 'eda_correlation.png'")
plt.show()