import obonet
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import os

# CONFIG
RAW_DIR = 'data_raw'
PROC_DIR = 'data_processed'

print("=== PROCESSING TARGETS (Y) ===")

# 1. Load Ontology
print("Loading Gene Ontology Graph...")
graph = obonet.read_obo(os.path.join(RAW_DIR, 'go-basic.obo'))

# 2. Load Labels
print("Loading Labels...")
df_terms = pd.read_csv(os.path.join(RAW_DIR, 'train_terms.tsv'), sep='\t')

# 3. Filter Top 500 Terms (Aggressive Middle Ground)
# 1500 was too many for RAM. 100 is too simple. 500 is the sweet spot.
TARGET_COUNT = 500
print(f"Filtering for top {TARGET_COUNT} most common terms...")
top_terms = df_terms['term'].value_counts().head(TARGET_COUNT).index.tolist()
df_filtered = df_terms[df_terms['term'].isin(top_terms)]

# 4. Map IDs to Integers
proteins = df_filtered['EntryID'].unique()
protein_to_idx = {p: i for i, p in enumerate(proteins)}
term_to_idx = {t: i for i, t in enumerate(top_terms)}

# 5. Build Matrix
print("Building Target Matrix (Y)...")
# Use int8 to save memory
data = np.ones(len(df_filtered), dtype=np.int8) 
rows = [protein_to_idx[p] for p in df_filtered['EntryID']]
cols = [term_to_idx[t] for t in df_filtered['term']]

Y_matrix = csr_matrix((data, (rows, cols)), shape=(len(proteins), len(top_terms)))

# 6. Save
print("Saving to data_processed/...")
if not os.path.exists(PROC_DIR):
    os.makedirs(PROC_DIR)

np.save(os.path.join(PROC_DIR, 'protein_ids.npy'), proteins)
np.save(os.path.join(PROC_DIR, 'term_ids.npy'), top_terms)
save_npz(os.path.join(PROC_DIR, 'Y_sparse.npz'), Y_matrix)

print("SUCCESS: Targets ready.")