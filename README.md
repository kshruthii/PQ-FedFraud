# Federated Learning for Illicit Bitcoin Transaction Detection with Cryptographic Update Verification

This repository contains an organized Jupyter notebook for experiments using graph neural networks and federated learning on the Elliptic Bitcoin transaction dataset. It compares model-update verification modes including post-quantum ML-DSA signatures, ECDSA signatures, and a no-signature baseline.

> **Important:** This repository is an organized version of a research notebook, not a polished install-and-run Python package. It contains multiple experiment versions and some cells depend on variables defined in earlier cells. Run a single intended pipeline in order rather than executing every experiment section indiscriminately. The experimental code from original notebook cells 29 and 30 has been intentionally omitted, as requested.

## Repository contents

```text
crypto_federated_learning_github/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── crypto_federated_learning.ipynb
├── docs/
│   └── crypto_code_organized.md
└── results/
    └── .gitkeep
```

- `notebooks/crypto_federated_learning.ipynb` — notebook version of the organized code, with each original code block kept as its own cell.
- `docs/crypto_code_organized.md` — readable code reference organized by experiment.
- `requirements.txt` — Python dependencies to start from.
- `.gitignore` — excludes environments, datasets, checkpoints, and generated model files.

## What the project explores

- Loading and preparing Elliptic transaction features, class labels, and transaction edges.
- Representing transactions as a graph and training Graph Convolutional Network (GCN) models.
- Simulating federated clients and aggregating local model weights with Federated Averaging (FedAvg).
- Signing and verifying model updates using post-quantum ML-DSA or classical ECDSA, with a no-signature baseline.
- Evaluating detection performance with accuracy, precision, recall, F1 score, and ROC-AUC.
- Measuring signing/verification time, signature size, and training-round time.
- Running experiments across different client counts and random seeds in the newer experiment pipeline.

## Dataset

The code uses the **Elliptic Bitcoin transaction dataset**. The notebook originally refers to Kaggle paths such as:

```text
/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/
```

The dataset is **not included** in this GitHub repository. Obtain it from its authorized source, review its terms, and place the CSV files in a local `data/elliptic_bitcoin_dataset/` folder or update the notebook paths to match your environment.

Expected files:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

The current notebook uses hard-coded Kaggle paths in multiple experiment sections, so those paths must be updated before running locally. Do not commit the dataset CSVs to GitHub.

## Environment setup

Python 3.10 or 3.11 is a reasonable starting point, but compatibility depends on the PyTorch, PyTorch Geometric, and Open Quantum Safe (`liboqs`) versions available for your operating system.

Create and activate a virtual environment:

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Dependency notes

- **PyTorch Geometric:** installation may require a PyTorch-version-specific wheel. If `pip install -r requirements.txt` fails for `torch-geometric`, follow the official PyTorch Geometric installation instructions for your installed PyTorch version.
- **ML-DSA / `oqs`:** `liboqs-python` relies on the Open Quantum Safe library and may need additional platform-specific setup. Check the official project instructions if importing `oqs` fails or the selected ML-DSA algorithm is not enabled.
- A GPU is optional in the notebook; it chooses CUDA when available and otherwise uses the CPU. Training can be slow on CPU.

## Running the notebook

1. Install the dependencies in a compatible Python environment.
2. Download the Elliptic dataset separately.
3. Open `notebooks/crypto_federated_learning.ipynb` in Jupyter or VS Code.
4. Update the dataset paths for your environment.
5. Choose the intended experiment pipeline and execute its cells in order. The notebook has overlapping historical and newer experiment implementations; do not blindly run all cells from top to bottom.
6. Review the generated metrics and plots. Checkpoint and output paths in the source code were originally written for Kaggle (`/kaggle/working/...`) and may need to be changed for local execution.

## GitHub upload

From the repository folder:

```bash
git init
git add .
git commit -m "Add federated crypto transaction detection project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your actual GitHub repository URL. If the repository already exists locally, do not run `git init` again; add the files, commit, and push to the existing remote instead.

## Limitations and reproducibility

- The notebook is preserved as an organized research artifact; it has not been fully refactored into a single command-line application.
- Results depend on dataset version, dependency versions, random seeds, hardware, and the selected experiment section.
- The `NONE` mode is an experimental comparison baseline and does not authenticate model updates.
- The notebook's signing experiment measures signatures over a hash of model weights. This is a research prototype, not a production-ready secure federated-learning deployment.
- No trained model checkpoints or experiment outputs are included. They can be large and should be generated locally.
