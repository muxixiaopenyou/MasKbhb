# MasKbhb: A Target-Centered Spatial Deep Learning Framework for Accurate Lysine $\beta$-Hydroxybutyrylation Site Prediction

> **Abstract**: Lysine $\beta$-hydroxybutyrylation (Kbhb) is an emerging yet increasingly important metabolism-associated post-translational modification (PTM) implicated in epigenetic regulation under ketogenic conditions. Existing computational approaches for PTM site prediction rely predominantly on local sequence context, although PTM selectivity is inherently influenced by the folded three-dimensional protein environment. Here, we propose PTM prediction as a target-centered structural neighborhood learning problem and present MasKbhb, a deep learning framework for Kbhb site prediction. MasKbhb represents full-length protein structures as residue-level spatial graphs and learns site-centric structural representations from folded structural neighborhoods through graph-based aggregation. MasKbhb substantially outperforms all available open-source Kbhb predictors on an independent test set, achieving an MCC of 0.6654 compared with 0.6233 for the strongest baseline. Residue attribution, graph edge importance analysis, and in silico mutagenesis further indicate that the model captures biologically meaningful structural interactions associated with Kbhb recognition. Robust generalization across human, mouse, and rice datasets demonstrates the transferability of the framework across diverse biological contexts. Collectively, this work supports the utility of structure-aware modeling for PTM site prediction and highlights the contribution of folded structural neighborhoods to Kbhb site recognition.

---

## The Structure of This Project

```
├── data/                          
│   ├── train.fasta                 
│   ├── test.fasta                  
│   ├── mouse_train.fasta          
│   ├── mouse_test.fasta            
│   ├── rice_train.fasta            
│   └── rice_test.fasta             
│
├── embedding/                      # Pre-computed graph data (download from Google Drive)
│   ├── train_embedding_graph.pt
│   ├── test_embedding_graph.pt
│   ├── mouse_train_embedding_graph.pt
│   ├── mouse_test_embedding_graph.pt
│   ├── rice_train_embedding_graph.pt
│   └── rice_test_embedding_graph.pt
│
├── pdb/                            # PDB structure files (download from Google Drive and extract into this directory)
│   ├── human_pdb.tar.gz
│   ├── human_seq_pdb.tar.gz
│   ├── mouse_pdb.tar.gz
│   └── rice_pdb.tar.gz
│
├── model/                          
│   ├── final_model.pth             
│   ├── best_params_*.json         
│   ├── all_trials_*.json           
│   ├── test_predictions.csv        
│   └── training_*.log              
│
├── mouse/
│   ├── mouse_model/                # Mouse model and optimization results
│   └── other_method_logs/          # Baseline comparison logs
│
├── rice/
│   ├── rice_model/                 # Rice model and optimization results
│   └── other_method_logs/          # Baseline comparison logs
│
├── hyper/                          # Hyperparameter optimization scripts
│   ├── Training Hyperparameter Optimization/
│   │   └── search.py               # 5-fold Group CV Bayesian optimization
│   └── Architecture Hyperparameter Search/
│       ├── windowsize/
│       ├── gat/
│       ├── transformer/
│       └── distance/
│
├── branch_ablation/                # Branch ablation experiments
│   ├── graph_branch/
│   └── onehot_branch/
│
├── structure_anlysis/              # Structure analysis experiments
│   ├── graphpooling/
│   ├── mlp/
│   └── seq/
│
├── other_logs/                     # Baseline comparison logs
│
├── generate_embedding.py           # Generate ESM-2 embeddings from FASTA sequences
├── plm_feature.py                  # Construct residue-level spatial graphs from PDB structures
├── model_final.py                  
├── model_mouse.py                  
├── model_rice.py                   
├── train_final.py                 
├── evaluation.py                  # Evaluate trained model on test set
├── utils.py                       
└── README.md
```

---

## Large Files Download

Owing to the large size of PDB files and pre-computed embeddings, they are provided via Google Drive:

### Embedding Files
Download from: https://drive.google.com/drive/u/1/folders/1mhBITqdlMHOtz8s22mxczMCFhVnJ9v07

Place the downloaded `.pt` graph files in the `embedding/` directory.

### PDB Files
Download from: https://drive.google.com/drive/u/1/folders/1D0BuOt-0kJu7zhC7VmGqEdPDzFZaKQfY

Place the downloaded `.tar.gz` files in the `pdb/` directory.

| File | Description | Destination |
|---|---|---|
| `*_embedding_graph.pt` | Pre-computed residue-level spatial graphs | `embedding/` |
| `human_pdb.tar.gz` | Human PDB structure files | `pdb/` |
| `human_seq_pdb.tar.gz` | Human sequence-based structure files | `pdb/` |
| `mouse_pdb.tar.gz` | Mouse PDB structure files | `pdb/` |
| `rice_pdb.tar.gz` | Rice PDB structure files | `pdb/` |


---

## Installation Guide

### 1. Create the Conda Environment

```bash
conda create -n maskbhb python=3.9 -y
conda activate maskbhb
```

---

### 2. Install Core Scientific Dependencies

```bash
conda install -c conda-forge \
    numpy=1.24.3 \
    scipy=1.13.1 \
    pandas=1.5.3 \
    matplotlib=3.7.1 \
    scikit-learn=1.6.1 \
    biopython=1.85 \
    -y
```

---

### 3. Install PyTorch with CUDA 11.8

```bash
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

---

### 4. Install PyTorch Geometric

```bash
pip install pyg-lib==0.4.0+pt22cu118 \
    torch-scatter==2.1.2+pt22cu118 \
    torch-sparse==0.6.18+pt22cu118 \
    torch-cluster==1.6.3+pt22cu118 \
    torch-geometric==2.6.1 \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

---

### 5. Install Transformer and Protein Language Model Dependencies

```bash
pip install \
    transformers==4.37.2 \
    tokenizers==0.15.2 \
    huggingface-hub==0.36.2 \
    safetensors==0.7.0 \
    fair-esm==2.0.0
```

---

### 6. Install Optimization Dependencies

```bash
pip install optuna==3.6.1
```

---

### 7. Environment Summary

| Package           | Version      |
| ----------------- | ------------ |
| Python            | 3.9          |
| PyTorch           | 2.2.2+cu118  |
| CUDA              | 11.8         |
| PyTorch Geometric | 2.6.1        |
| Transformers      | 4.37.2       |
| fair-esm          | 2.0.0        |
| Optuna            | 3.6.1        |
| NumPy             | 1.24.3       |
| SciPy             | 1.13.1       |
| scikit-learn      | 1.6.1        |
| biopython         | 1.85         |

---

## Usage

### Quick Start: Evaluate Pre-trained Model

To evaluate the pre-trained model on the test set, download the pre-computed `.pt` files from Google Drive, place them in the `embedding/` directory, then run:

```bash
python evaluation.py
```

This script:
- Loads the trained model from `model/final_model.pth`
- Evaluates on the test set
- Outputs metrics: MCC, SN, SP, ACC, BACC, PRECISION, F1, AUC, AUCPR
- Saves predictions to `model/test_predictions_{timestamp}.csv`
- Saves evaluation log to `model/evaluation_{timestamp}.log`

---

### Custom Data Processing *(optional)*

If you want to process your own data from raw sequences and PDB structures:

#### Step 1: Generate ESM-2 Embeddings

```bash
python generate_embedding.py
```

This script:
- Loads sequences from `data/train.fasta` and `data/test.fasta`
- Generates embeddings using ESM-2 (esm2_t33_650M_UR50D)
- Saves embeddings to `embedding/train_embedding.pt` and `embedding/test_embedding.pt`

#### Step 2: Construct Spatial Graphs

```bash
python plm_feature.py
```

This script:
- Parses PDB structures from the `pdb/` directory
- Constructs residue-level spatial graphs using pre-computed embeddings
- Saves graph data to `embedding/train_embedding_graph.pt` and `embedding/test_embedding_graph.pt`

---

### Model Training *(optional)*

If you want to retrain the model on your own dataset:

#### Step 3: Hyperparameter Optimization

```bash
python hyper/Training\ Hyperparameter\ Optimization/search.py
```

This script performs:
- **5-fold Group Cross-Validation**: Ensures samples from the same protein do not appear in both training and validation folds
- **Bayesian Optimization**: Uses Optuna TPE sampler to search for optimal hyperparameters
- **Early Stopping**: Monitors MCC with patience=5 to prevent overfitting

The optimization searches for:
- `learning_rate`: 1e-6 to 1e-2 (log scale)
- `weight_decay`: 1e-6 to 1e-2 (log scale)
- `batch_size`: [32, 64, 128, 256, 512]

Results are saved to:
- `best_params_*.json`: Best hyperparameters and recommended epochs
- `all_trials_*.json`: All trial results

#### Step 4: Final Training

```bash
python train_final.py
```

This script:
- Trains on the full training set
- Evaluates on the independent test set
- Saves the final model to `model/final_model.pth`

---

## Training Strategy

The model training follows a two-stage approach:

1. **Hyperparameter Optimization Stage**
   - 5-fold Group Cross-Validation
   - Bayesian optimization with Optuna TPE sampler
   - Each fold uses early stopping based on MCC
   - Output: Optimal hyperparameters (lr, weight_decay, batch_size) and recommended epochs

2. **Final Training Stage**
   - Train on the complete training set
   - Use optimal hyperparameters from Stage 1
   - Use median of best epochs across CV folds
   - Evaluate on the independent test set


