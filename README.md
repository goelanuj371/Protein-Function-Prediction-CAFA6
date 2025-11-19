 CAFA 6 Protein Function Prediction  
A Hybrid Cloud-Local Bioinformatics Pipeline for High-Dimensional Multi-Label Classification

## 🔬 Project Overview
This project implements an end-to-end machine learning pipeline to predict the biological function (Gene Ontology terms) of proteins based solely on their amino acid sequences. Developed for the CAFA 6 Challenge, the system addresses the computational complexity of high-dimensional biological data by architecting a hybrid cloud-local workflow.

The core innovation lies in decoupling heavy compute tasks from memory-intensive tasks: utilizing Google Colab T4 GPUs for deep feature extraction via Transfer Learning, and optimizing a local CPU environment for training on a directed acyclic graph (DAG) of 500 hierarchical targets.

### Key Achievements
- **Micro-F1 Score:** 0.33 (achieved via probability threshold optimization, effectively doubling the baseline performance of 0.15).  
- **Processing Speed:** 6× speedup in embedding generation (5 hours → 48 minutes) using Mixed Precision (FP16) and optimized batched inference on T4 GPUs.  
- **Scalability:** Successfully processed 142,000+ sequences with 500 hierarchical targets on consumer hardware (16GB RAM) using sparse matrix operations.

## 🏗 Architecture & Methodology
The pipeline follows a modular design pattern for reproducibility and resource efficiency.

### 1. Feature Engineering (Cloud - Google Colab T4)
Instead of manual physiochemical features, the project uses Transfer Learning with the ESM-2 Transformer.

- **Model:** `facebook/esm2_t30_150M_UR50D`  
- **Inference Strategy:**  
  - Frozen weights in `eval()` mode to extract embeddings.  
  - FP16 weights to reduce VRAM usage and enable batch size of 128.  
  - Embeddings stored as `.npy` chunks of 5,000 sequences each.

### 2. Data Pipeline & Alignment (Local - CPU)
- Parsed Gene Ontology DAG using `obonet` and `go-basic.obo`.  
- Reduced ~30,000 GO terms to the top 500.  
- Sparse encoding using `scipy.sparse.csr_matrix` (98% memory reduction).  
- Ensured correct alignment between sequence order and label order.

### 3. Modeling (Local - CPU)
- Multi-Label Random Forest Classifier.  
- Hardware-optimized hyperparameters:  
  - `n_estimators=50`  
  - `n_jobs=4`  
  - `max_depth=12`

## 📊 Results & Analysis

### Performance Metrics

| Metric     | Value   | Insight |
|------------|---------|---------|
| **Micro-F1** | **0.3286** | Strong performance on common functions. |
| **Macro-F1** | **0.0037** | Expected difficulty with rare classes. |

### Optimization Strategy
Initial F1 was ~0.15. Optimal decision threshold discovered: **0.20**.

### Exploratory Data Analysis (EDA)
- Class imbalance validated metric choice.  
- Length distribution justified <1024 residue truncation.  
- Correlation heatmaps showed hierarchical clustering → good fit for non-linear models.

## 📂 Repository Structure

```
├── data_processed/
├── notebooks/
├── 1_feature_engineering.ipynb
├── process_ontology.py
├── train_model.py
├── evaluate_model.py
├── perform_eda.py
└── README.md
```

## 🚀 Usage

### Prerequisites
- Python 3.8+
- 16GB RAM
- Google Colab access

### Installation

```
pip install obonet networkx pandas scikit-learn scipy numpy matplotlib seaborn transformers torch
```

### Execution Steps

#### 1. Generate Embeddings
Run `1_feature_engineering.ipynb` in Colab.

#### 2. Process Ontology
```
python process_ontology.py
```

#### 3. Train Model
```
python train_model.py
```

#### 4. Evaluate Results
```
python evaluate_model.py
```

## 📜 License
This work is licensed under the MIT License. © 2025 Anuj Goel.  
Created for research and educational use within the CAFA 6 Challenge.

