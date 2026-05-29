# MasKbhb: A Target-Centered Spatial Deep Learning Framework for Accurate Lysine $\beta$-Hydroxybutyrylation Site Prediction

> **Abstract**: Lysine $\beta$-hydroxybutyrylation (Kbhb) is an emerging yet increasingly important metabolism-associated post-translational modification (PTM) implicated in epigenetic remodeling and transcriptional regulation under ketogenic conditions. However, existing computational approaches for PTM site prediction have relied predominantly on local sequence context, although PTM selectivity is inherently governed by the folded three-dimensional protein environment. Here, we propose PTM prediction as a target-centered structural neighborhood learning problem and present MasKbhb, a deep learning framework for Kbhb site prediction. MasKbhb represents full-length protein structures as residue-level spatial graphs and learns target-centered structural representations from folded structural neighborhoods through graph-based aggregation. MasKbhb substantially outperforms all available open-source Kbhb predictors on an independent test set, achieving an MCC of 0.6654 compared with 0.6233 for the strongest baseline. Residue attribution, graph edge importance analysis, and in silico mutagenesis further indicate that the model captures biologically meaningful structural interactions associated with Kbhb recognition. Robust generalization across human, mouse, and rice datasets demonstrates the transferability of the framework across diverse biological contexts. Collectively, this work establishes a structure-aware paradigm for PTM site prediction and highlights the importance of folded structural neighborhoods in governing modification selectivity.

---

## The Structure of This Project

```
├── data/                           
│   # Human 
│   ├── train.fasta                 
│   ├── test.fasta                       
│   # Mouse 
│   ├── mouse_train.fasta           
│   ├── mouse_test.fasta            
│   # Rice 
│   ├── rice_train.fasta            
│   ├── rice_test.fasta             
│   # Pre-computed graph data (downloaded from Google Drive)
│   ├── train_embedding_graph.pt
│   ├── test_embedding_graph.pt
│   ├── mouse_train_embedding_graph.pt
│   ├── mouse_test_embedding_graph.pt
│   ├── rice_train_embedding_graph.pt
│   └── rice_test_embedding_graph.pt         
│
├── pdb/                            # PDB structure files (downloaded from Google Drive)
│   ├── human/                      
│   ├── human_seq/                  
│   ├── mouse/                      
│   └── rice/
│
├── models/                        
│   ├── fold_1.pth                  
│   ├── fold_2.pth
│   ├── fold_3.pth
│   ├── fold_4.pth
│   └── fold_5.pth                                        
│
├── mouse/                          
│   ├── mouse_models/              
│   └── other_method_logs/          
│
├── rice/                           
│   ├── rice_models/               
│   └── other_method_logs/         
│
├── branch_ablation/               
│   ├── graph_branch/              
│   └── onehot_branch/             
│
├── structure_analysis/              
│   ├── graphpooling/              
│   ├── mlp/                        
│   └── seq/                       
│
├── logs/                           # Training and baseline method comparison logs
│
├── generate_embedding.py           # Generate ESM-2 embeddings from FASTA sequences
├── plm_feature.py                  # Construct residue-level spatial graphs from PDB structures
├── model.py                        
├── train.py                        
├── utils.py                        
└── README.md                       
```
Owing to the large size of PDB files and pre-computed embeddings, they are provided via Google Drive:
 
```
https://drive.google.com/drive/folders/1YMhOaNpzMl70y_r2qaH3xUzO5sdGo0fZ?usp=share_link
```
 
The Google Drive folder contains:
 
| File / Folder | Description | Destination |
|---|---|---|
| `embedding/*.pt` graph files | Pre-computed residue-level spatial graphs | `data/` |
| `pdb/human_pdb.tar.gz` | Human PDB structure files | `pdb/` |
| `pdb/human_seq.tar.gz` | Human sequence-based ESMFold-predicted structure files | `pdb/` |
| `pdb/mouse_pdb.tar.gz` | Mouse PDB structure files | `pdb/` |
| `pdb/rice_pdb.tar.gz` | Rice PDB structure files | `pdb/` |
 
> **Note**: The pre-computed `.pt` graph files are sufficient to reproduce all training and evaluation results.
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

### 6. Environment Summary
 
| Package           | Version      |
| ----------------- | ------------ |
| Python            | 3.9          |
| PyTorch           | 2.2.2+cu118  |
| CUDA              | 11.8         |
| PyTorch Geometric | 2.6.1        |
| Transformers      | 4.37.2       |
| fair-esm          | 2.0.0        |
| NumPy             | 1.24.3       |
| SciPy             | 1.13.1       |
| scikit-learn      | 1.6.1        |
| biopython         | 1.85         |


---

## Usage
 
To reproduce the results directly, download the pre-computed `.pt` files from Google Drive, place them in the `data/` directory, and proceed to **Step 3**. Steps 1 and 2 are only needed if you want to regenerate the embeddings and graphs from raw sequences and PDB structures.

### Step 1: Generate ESM-2 Embeddings *(optional)*

```bash
python generate_embedding.py
```

This script:
- Loads sequences from `data/train.fasta` and `data/test.fasta`
- Generates embeddings using ESM-2 (esm2_t33_650M_UR50D)
- Saves embeddings to `data/train_embedding.pt` and `data/test_embedding.pt`

### Step 2: Construct Spatial Graphs *(optional)*

```bash
python plm_feature.py
```

This script:
- Parses PDB structures from the `pdb/` directory
- Constructs residue-level spatial graphs using pre-computed embeddings
- Saves graph data to `data/train_embedding_graph.pt` and `data/test_embedding_graph.pt`

### Step 3: Train the Model

```bash
python train.py
```

This script:
- Performs 5-fold stratified cross-validation on the training set
- Saves the best model from each fold to `models/fold_*.pth`
- Evaluates the ensemble on the independent test set
- Logs training progress and final metrics
