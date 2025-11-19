import numpy as np
import pandas as pd
import glob
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.sparse import load_npz
import joblib
import gc

# CONFIG
RAW_DIR = "data_raw"
PROC_DIR = "data_processed"

print("=== STEP 1: LOADING FEATURES (X) ===")
chunk_files = glob.glob(f"{PROC_DIR}/embeddings_chunk_*.npy")
# Robust sorting to handle 'chunk_0', 'chunk_5000'
chunk_files.sort(key=lambda f: int(re.search(r'chunk_(\d+)', f).group(1)))

print(f"Found {len(chunk_files)} chunk files. Loading...")
X_list = [np.load(f) for f in chunk_files]
X = np.concatenate(X_list)
print(f"X Shape: {X.shape}")

# Clean up list to free RAM immediately
del X_list
gc.collect()

print("\n=== STEP 2: ALIGNMENT ===")
print("Reading FASTA IDs...")
fasta_ids = []
with open(f"{RAW_DIR}/train_sequences.fasta", 'r') as f:
    for line in f:
        if line.startswith(">"):
            try:
                # Extract ID from >sp|ID|Name format
                fasta_ids.append(line.strip().split('|')[1])
            except IndexError:
                fasta_ids.append(line.strip().split()[0][1:])

print("Loading Targets...")
Y_sparse = load_npz(f"{PROC_DIR}/Y_sparse.npz")
y_protein_ids = np.load(f"{PROC_DIR}/protein_ids.npy", allow_pickle=True)
y_map = {pid: i for i, pid in enumerate(y_protein_ids)}

print("Aligning X and Y...")
valid_indices_x = [] 
Y_aligned = []

for i, pid in enumerate(fasta_ids):
    if pid in y_map:
        valid_indices_x.append(i)
        # Convert single row to dense only when needed to save RAM
        Y_aligned.append(Y_sparse[y_map[pid]].toarray()[0])

X_final = X[valid_indices_x]
Y_final = np.array(Y_aligned)

print(f"Final Training Data: {X_final.shape}")

# Clean up raw X/Y to free RAM for training
del X, Y_sparse, fasta_ids, y_protein_ids, Y_aligned
gc.collect()

print("\n=== STEP 3: TRAINING (Aggressive Middle Ground) ===")
X_train, X_val, y_train, y_val = train_test_split(X_final, Y_final, test_size=0.2, random_state=42)

print(f"Training on {X_train.shape[0]} samples with {Y_final.shape[1]} targets...")
print("Config: n_jobs=4, max_depth=12")

# --- OPTIMIZED RANDOM FOREST ---
# n_jobs=4: Uses 4 cores. Keeps memory overhead manageable (~8-10GB).
# max_depth=12: Restricts tree size to prevent memory explosion.
clf = RandomForestClassifier(
    n_estimators=50, 
    n_jobs=4,           
    max_depth=12,       
    random_state=42, 
    verbose=1
)

clf.fit(X_train, y_train)

print("\n=== STEP 4: EVALUATION ===")
y_pred = clf.predict(X_val)

micro_f1 = f1_score(y_val, y_pred, average='micro')
macro_f1 = f1_score(y_val, y_pred, average='macro')

print(f"\n" + "="*30)
print(f"RESULTS (Top 500 Terms)")
print(f"="*30)
print(f"Micro F1 Score: {micro_f1:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"="*30)

print("Saving model...")
joblib.dump(clf, f"{PROC_DIR}/final_model_500.pkl")
print("Done.")