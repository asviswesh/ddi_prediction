This repository contains experiments on **drug–drug interaction link prediction** using
graph neural networks (GNNs) and semantic node embeddings. They are evaluated using the **OGB-ddi** benchmark.

We compare:

- A **structural GNN** baseline
- A **GNN augmented with semantic (SciBERT-based) node embeddings**
- A **semantic-only MLP with contrastive training** on precomputed embeddings

All code is organized as Colab-friendly notebooks.

# Repository Contents

### 1. `5M_STRUCTURE_Final_Link_Prediction_without_SciBert.ipynb`

Structural GNN baseline for OGB-ddi link prediction.

### 2. `5M_STRUCTURE_Final_Link_Prediction_with_SciBert.ipynb`

GNN, whose nodes are enhanced with additional semantic features.

### 3. `SEMANTIC_mlp_contrastive_precomp_Final_Link_Prediction_with_SciBert.ipynb`

Semantic-only model using precomputed text embeddings.

## Environment & Dependencies

The notebooks are designed to run in **Google Colab** with a GPU runtime.

Key libraries (installed in the notebooks):

- `torch` (PyTorch 2.x with CUDA)
- `torch_geometric` (PyG)
- `ogb` (for the OGB-ddi dataset and evaluation)
- `numpy`
- SciBERT embeddings

The notebooks already contain the appropriate `pip install` commands at the top,
including specific Torch / CUDA / PyG versions known to work in Colab.

## Data

We work with the **OGB-ddi** dataset from the
[Open Graph Benchmark](https://ogb.stanford.edu/):

- Automatically downloaded via `ogb` in the structural notebooks.
- Edges are split into train/validation/test as provided by OGB.
- Negative edges are sampled with care to avoid contamination from the
  validation and test sets.

## How to Run

### 1. Open in Google Colab

1. Upload the notebooks to your GitHub repo (or directly to Google Drive).
2. Open each notebook in **Google Colab**.
3. Set runtime to **GPU**:  
   `Runtime → Change runtime type → Hardware accelerator → GPU`.

### 2. Configure paths (Google Drive)

Each notebook assumes a base directory on your Drive, e.g.:
BASE_DIR = "/content/drive/MyDrive/CS145/neurips/FINAL-CODE/"

### 3. For the Semantic pipeline 

If you are using the semantic/SciBERT variants:

Run the semantic contrastive notebook
SEMANTIC_mlp_contrastive_precomp_Final_Link_Prediction_with_SciBert.ipynb
to train the MLP and save the projected embeddings to a .pt file.

Ensure the structural+SciBERT notebook
5M_STRUCTURE_Final_Link_Prediction_with_SciBert.ipynb
points to that file (e.g., projected_embeddings_512.pt). 

### 4. Structural baselines

Run 5M_STRUCTURE_Final_Link_Prediction_without_SciBert.ipynb to train the
purely structural GNN baseline.

Run 5M_STRUCTURE_Final_Link_Prediction_with_SciBert.ipynb to train the
structural + semantic model. (Make sure the notebook points to the projected_embeddings_512.pt created in Step 3)

### Results & Metrics

The main evaluation metric used in these experiments is:
Hits@20 for link prediction on OGB-ddi.

The notebooks log and plot:
- Training/validation loss over epochs
- Hits@20 on validation splits





  
