import numpy as np
import pandas as pd
import joblib
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import load_npz

# CONFIG
PROC_DIR = "data_processed"
RAW_DIR = "data_raw"

print("=== LOADING DATA FOR EVALUATION ===")
# 1. Load X (Features)
chunk_files = glob.glob(f"{PROC_DIR}/embeddings_chunk_*.npy")
chunk_files.sort(key=lambda f: int(re.search(r'chunk_(\d+)', f).group(1)))
X = np.concatenate([np.load(f) for f in chunk_files])

# 2. Load Y (Targets) - Align again quickly
fasta_ids = []
with open(f"{RAW_DIR}/train_sequences.fasta", 'r') as f:
    for line in f:
        if line.startswith(">"):
            try:
                fasta_ids.append(line.strip().split('|')[1])
            except:
                fasta_ids.append(line.strip().split()[0][1:])

Y_sparse = load_npz(f"{PROC_DIR}/Y_sparse.npz")
y_map = {pid: i for i, pid in enumerate(np.load(f"{PROC_DIR}/protein_ids.npy", allow_pickle=True))}
valid_indices = [i for i, pid in enumerate(fasta_ids) if pid in y_map]

X_final = X[valid_indices]
Y_final = np.array([Y_sparse[y_map[fasta_ids[i]]].toarray()[0] for i in valid_indices])

# 3. Split (Must use same seed as training to get same Validation Set)
_, X_val, _, y_val = train_test_split(X_final, Y_final, test_size=0.2, random_state=42)

# 4. Load Model
print("Loading Model...")
clf = joblib.load(f"{PROC_DIR}/final_model_500.pkl")

print("=== OPTIMIZING THRESHOLD ===")
# Get raw probabilities instead of hard predictions
y_proba = clf.predict_proba(X_val)
# Random Forest returns a list of arrays (one for each class). We need to stack them.
# This line converts the weird sklearn output format to a nice matrix
y_proba_matrix = np.array([p[:, 1] for p in y_proba]).T 

# Test thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
for thresh in thresholds:
    y_pred_new = (y_proba_matrix >= thresh).astype(int)
    score = f1_score(y_val, y_pred_new, average='micro')
    print(f"Threshold {thresh:.1f} -> Micro F1: {score:.4f}")

print("\n=== SAMPLE PREDICTIONS ===")
# Let's see what it actually predicts for the first protein in validation
term_ids = np.load(f"{PROC_DIR}/term_ids.npy", allow_pickle=True)
print(f"Protein 0 Real Functions: {term_ids[y_val[0] == 1]}")
print(f"Protein 0 Pred Functions (Thresh 0.2): {term_ids[y_proba_matrix[0] >= 0.2]}")