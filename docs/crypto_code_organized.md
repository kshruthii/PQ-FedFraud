# Crypto notebook — code organized by purpose

This is an organized copy of the code cells from `crypto.ipynb`, not a rewritten or guaranteed runnable script. The notebook contains multiple overlapping versions, so don't run every section in sequence without choosing one version.

## Quick guide

- **Main newer pipeline:** sections 2, 4, then 5 (but check file paths and the order of definitions).

- **Older pipeline:** section 3. Keep it only if you need results from the earlier experiment.

- **Section 6 is experimental:** it changes how client graph data is built and contains only a partial training-loss snippet; it is not a complete replacement pipeline.

- Cells 3–4 only search for CSV files; cell 7/16 checks available post-quantum signature mechanisms; these are setup/debug cells, not core model code.


---
## 1. Setup and finding the dataset


### Original notebook cell 0

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')
```


### Original notebook cell 1

```python
!pip install torch_geometric -q
!pip install liboqs-python -q   # for ML-DSA-65 signing (Phase 3)
```


### Original notebook cell 3

```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        print(os.path.join(root, f))
```


### Original notebook cell 4

```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.endswith(".csv"):
            print(os.path.join(root, f))
```


### Original notebook cell 7

```python
import oqs
sigs = oqs.get_enabled_sig_mechanisms()
print([s for s in sigs if "ML-DSA" in s or "Dilithium" in s])
```


### Original notebook cell 16

```python
!pip install torch_geometric liboqs-python -q
import oqs
print(oqs.get_enabled_sig_mechanisms()[:3])
```


---
## 2. Load and prepare Elliptic data


### Original notebook cell 2

```python
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

feat_path = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
class_path = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edge_path = "/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"

features = pd.read_csv(feat_path, header=None)
classes = pd.read_csv(class_path)
edges = pd.read_csv(edge_path)

features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
classes.columns = ["txId", "class"]

merged = features.merge(classes, on="txId", how="left")
merged["class"] = merged["class"].map({"1": 1, "2": 0, "unknown": -1})

# node id -> row index mapping (needed for edge_index)
node_id_map = {txid: idx for idx, txid in enumerate(merged["txId"])}
edges["src"] = edges["txId1"].map(node_id_map)
edges["dst"] = edges["txId2"].map(node_id_map)
edges = edges.dropna()

x = torch.tensor(merged.iloc[:, 2:167].values, dtype=torch.float)
y = torch.tensor(merged["class"].values, dtype=torch.long)
time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)
edge_index = torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y, time_step=time_step)
labeled_mask = y != -1
print(data, "| labeled nodes:", labeled_mask.sum().item())
```


### Original notebook cell 5

```python
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

feat_path = "/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
class_path = "/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edge_path = "/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"

features = pd.read_csv(feat_path, header=None)
classes = pd.read_csv(class_path)
edges = pd.read_csv(edge_path)

features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
classes.columns = ["txId", "class"]

merged = features.merge(classes, on="txId", how="left")
merged["class"] = merged["class"].map({"1": 1, "2": 0, "unknown": -1})

# node id -> row index mapping (needed for edge_index)
node_id_map = {txid: idx for idx, txid in enumerate(merged["txId"])}
edges["src"] = edges["txId1"].map(node_id_map)
edges["dst"] = edges["txId2"].map(node_id_map)
edges = edges.dropna()

x = torch.tensor(merged.iloc[:, 2:167].values, dtype=torch.float)
y = torch.tensor(merged["class"].values, dtype=torch.long)
time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)
edge_index = torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y, time_step=time_step)
labeled_mask = y != -1
print(data, "| labeled nodes:", labeled_mask.sum().item())
```


### Original notebook cell 15

```python
#cell 0
import pandas as pd, numpy as np, torch, torch.nn.functional as F
import hashlib, time, os, json, copy

base = "/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/"
features = pd.read_csv(base + "elliptic_txs_features.csv", header=None)
classes = pd.read_csv(base + "elliptic_txs_classes.csv")
edges = pd.read_csv(base + "elliptic_txs_edgelist.csv")

features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
classes.columns = ["txId", "class"]
merged = features.merge(classes, on="txId", how="left")
merged["class"] = merged["class"].map({"1": 1, "2": 0, "unknown": -1})

node_id_map = {t: i for i, t in enumerate(merged["txId"])}
edges["src"] = edges["txId1"].map(node_id_map)
edges["dst"] = edges["txId2"].map(node_id_map)
edges = edges.dropna()

x = torch.tensor(merged.iloc[:, 2:167].values, dtype=torch.float)
y = torch.tensor(merged["class"].values, dtype=torch.long)
time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)
edge_index = torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long)
labeled_mask = y != -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(x.shape, edge_index.shape, labeled_mask.sum().item(), device)
```


---
## 3. Earlier federated-learning experiment (older version)


### Original notebook cell 6

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import os, json

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gcn_baseline.pt")
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "history.json")

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_dim=x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

x_dev = x.to(device)
edge_index_dev = edge_index.to(device)
y_dev = y.to(device)

train_mask = (labeled_mask & (time_step <= 34)).to(device)
test_mask = (labeled_mask & (time_step > 34)).to(device)

class_weights = torch.tensor([1.0, 4.0]).to(device)

TOTAL_EPOCHS = 200
start_epoch = 0
history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": [],
           "precision": [], "recall": [], "f1": [], "auc": []}

# ---- RESUME LOGIC: run this every time you restart the notebook ----
if os.path.exists(CKPT_PATH):
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
    print(f"Resumed from epoch {start_epoch}")
else:
    print("No checkpoint found — starting fresh from epoch 0")

# ---- TRAINING LOOP ----
for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(x_dev, edge_index_dev)
    loss = F.cross_entropy(out[train_mask], y_dev[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()

    # ---- Evaluation ----
    model.eval()
    with torch.no_grad():
        out = model(x_dev, edge_index_dev)
        preds = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)[:, 1]

        train_acc = accuracy_score(y_dev[train_mask].cpu(), preds[train_mask].cpu())
        test_y = y_dev[test_mask].cpu()
        test_preds = preds[test_mask].cpu()
        test_probs = probs[test_mask].cpu()

        test_acc = accuracy_score(test_y, test_preds)
        precision = precision_score(test_y, test_preds, pos_label=1, zero_division=0)
        recall = recall_score(test_y, test_preds, pos_label=1, zero_division=0)
        f1 = f1_score(test_y, test_preds, pos_label=1, zero_division=0)
        try:
            auc = roc_auc_score(test_y, test_probs)
        except ValueError:
            auc = float("nan")

    history["epoch"].append(epoch)
    history["train_loss"].append(loss.item())
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)
    history["auc"].append(auc)

    if epoch % 10 == 0 or epoch == TOTAL_EPOCHS - 1:
        print(f"Epoch {epoch:3d} | loss {loss.item():.4f} | train_acc {train_acc:.4f} | "
              f"test_acc {test_acc:.4f} | recall {recall:.4f} | f1 {f1:.4f} | auc {auc:.4f}")

    # ---- SAVE CHECKPOINT every 5 epochs (and always at the last epoch) ----
    if epoch % 5 == 0 or epoch == TOTAL_EPOCHS - 1:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, CKPT_PATH)
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f)

print("Training complete (or paused). Checkpoint + history saved to /kaggle/working/checkpoints/")
```


### Original notebook cell 8

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import hashlib, time, os, json, copy
import oqs

SIG_ALG = "ML-DSA-65"   # <-- update this after checking the printed list above

CKPT_DIR = "/kaggle/working/fl_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "fl_global.pt")
HISTORY_PATH = os.path.join(CKPT_DIR, "fl_history.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 1. Partition into simulated institutions ----------------
NUM_CLIENTS = 3
labeled_idx = torch.where(labeled_mask)[0]

# split labeled nodes into NUM_CLIENTS roughly equal chunks (by time_step order, mimics institutions over time)
sorted_idx = labeled_idx[torch.argsort(time_step[labeled_idx])]
client_splits = torch.chunk(sorted_idx, NUM_CLIENTS)

client_data = []
for i, idxs in enumerate(client_splits):
    sub_edge_index, _ = subgraph(idxs, edge_index, relabel_nodes=True, num_nodes=x.shape[0])
    client_data.append({
        "node_idx": idxs,
        "x": x[idxs].to(device),
        "y": y[idxs].to(device),
        "edge_index": sub_edge_index.to(device),
        "time_step": time_step[idxs],
    })
    print(f"Client {i}: {len(idxs)} nodes, time range {time_step[idxs].min().item()}-{time_step[idxs].max().item()}")

# global test set = last time steps, held out centrally for fair comparison across rounds
test_mask_global = (labeled_mask & (time_step > 34)).to(device)

# ---------------- 2. Model definition (same GCN as baseline) ----------------
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)

global_model = GCN(in_dim=x.shape[1]).to(device)
class_weights = torch.tensor([1.0, 4.0]).to(device)

# ---------------- 3. PQC key setup (one keypair per client, generated once) ----------------
client_keys = []
for i in range(NUM_CLIENTS):
    signer = oqs.Signature(SIG_ALG)
    pub = signer.generate_keypair()
    client_keys.append({"signer": signer, "public_key": pub})

def hash_model(state_dict):
    """Flatten model weights into bytes and hash with SHA3-256."""
    buf = b"".join(t.cpu().numpy().tobytes() for t in state_dict.values())
    return hashlib.sha3_256(buf).digest()

# ---------------- 4. Resume logic ----------------
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5
start_round = 0
history = {"round": [], "test_acc": [], "precision": [], "recall": [], "f1": [], "auc": [],
           "avg_sign_time_ms": [], "avg_verify_time_ms": [], "sig_size_bytes": [], "round_time_s": []}

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    global_model.load_state_dict(ckpt["model_state"])
    start_round = ckpt["round"] + 1
    with open(HISTORY_PATH, "r") as f:
        history = json.load(f)
    print(f"Resumed FL training from round {start_round}")
else:
    print("No FL checkpoint found — starting fresh")

# ---------------- 5. FedAvg training loop with PQC signing ----------------
for rnd in range(start_round, NUM_ROUNDS):
    round_start = time.time()
    local_states = []
    local_weights = []
    sign_times, verify_times, sig_sizes = [], [], []

    for i, cdata in enumerate(client_data):
        local_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=0.01, weight_decay=5e-4)

        local_model.train()
        for _ in range(LOCAL_EPOCHS):
            optimizer.zero_grad()
            out = local_model(cdata["x"], cdata["edge_index"])
            loss = F.cross_entropy(out, cdata["y"], weight=class_weights)
            loss.backward()
            optimizer.step()

        state = local_model.state_dict()

        # ---- PQC: hash + sign the update ----
        h = hash_model(state)
        t0 = time.perf_counter()
        signature = client_keys[i]["signer"].sign(h)
        t1 = time.perf_counter()
        sign_times.append((t1 - t0) * 1000)
        sig_sizes.append(len(signature))

        # ---- Server-side: verify signature before accepting update ----
        t0 = time.perf_counter()
        verifier = oqs.Signature(SIG_ALG)
        is_valid = verifier.verify(h, signature, client_keys[i]["public_key"])
        t1 = time.perf_counter()
        verify_times.append((t1 - t0) * 1000)

        if not is_valid:
            print(f"  Client {i}: SIGNATURE INVALID — update rejected")
            continue

        local_states.append(state)
        local_weights.append(len(cdata["node_idx"]))

    # ---- FedAvg aggregation ----
    total_weight = sum(local_weights)
    new_state = {}
    for key in local_states[0].keys():
        new_state[key] = sum(local_states[i][key] * (local_weights[i] / total_weight)
                              for i in range(len(local_states)))
    global_model.load_state_dict(new_state)

    # ---- Evaluate global model ----
    global_model.eval()
    with torch.no_grad():
        out = global_model(x.to(device), edge_index.to(device))
        preds = out.argmax(dim=1)
        probs = F.softmax(out, dim=1)[:, 1]
        test_y = y.to(device)[test_mask_global].cpu()
        test_preds = preds[test_mask_global].cpu()
        test_probs = probs[test_mask_global].cpu()

        test_acc = accuracy_score(test_y, test_preds)
        precision = precision_score(test_y, test_preds, pos_label=1, zero_division=0)
        recall = recall_score(test_y, test_preds, pos_label=1, zero_division=0)
        f1 = f1_score(test_y, test_preds, pos_label=1, zero_division=0)
        try:
            auc = roc_auc_score(test_y, test_probs)
        except ValueError:
            auc = float("nan")

    round_time = time.time() - round_start
    history["round"].append(rnd)
    history["test_acc"].append(test_acc)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)
    history["auc"].append(auc)
    history["avg_sign_time_ms"].append(sum(sign_times) / len(sign_times))
    history["avg_verify_time_ms"].append(sum(verify_times) / len(verify_times))
    history["sig_size_bytes"].append(sig_sizes[0])
    history["round_time_s"].append(round_time)

    print(f"Round {rnd:2d} | test_acc {test_acc:.4f} | recall {recall:.4f} | f1 {f1:.4f} | auc {auc:.4f} | "
          f"sign {history['avg_sign_time_ms'][-1]:.3f}ms | verify {history['avg_verify_time_ms'][-1]:.3f}ms | "
          f"sig_size {sig_sizes[0]}B | round_time {round_time:.2f}s")

    # ---- Checkpoint every round ----
    torch.save({"round": rnd, "model_state": global_model.state_dict()}, CKPT_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f)

print("FL+PQC training complete (or paused). Checkpoints in /kaggle/working/fl_checkpoints/")
```


### Original notebook cell 9

```python
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

def run_fl_experiment(crypto_mode, num_clients=3, num_rounds=20, local_epochs=5,
                       ckpt_dir="/kaggle/working/fl_checkpoints"):
    """crypto_mode: 'PQC', 'ECDSA', or 'NONE'"""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"fl_{crypto_mode}.pt")
    history_path = os.path.join(ckpt_dir, f"fl_{crypto_mode}_history.json")

    global_model = GCN(in_dim=x.shape[1]).to(device)
    history = {"round": [], "test_acc": [], "precision": [], "recall": [], "f1": [], "auc": [],
               "avg_sign_time_ms": [], "avg_verify_time_ms": [], "sig_size_bytes": [], "round_time_s": []}
    start_round = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        global_model.load_state_dict(ckpt["model_state"])
        start_round = ckpt["round"] + 1
        with open(history_path, "r") as f:
            history = json.load(f)
        print(f"[{crypto_mode}] Resumed from round {start_round}")
    else:
        print(f"[{crypto_mode}] Starting fresh")

    # ---- Key setup per mode ----
    if crypto_mode == "PQC":
        client_keys = []
        for i in range(num_clients):
            signer = oqs.Signature(SIG_ALG)
            pub = signer.generate_keypair()
            client_keys.append({"signer": signer, "public_key": pub})
    elif crypto_mode == "ECDSA":
        client_keys = []
        for i in range(num_clients):
            priv = ec.generate_private_key(ec.SECP256K1())
            pub = priv.public_key()
            client_keys.append({"private_key": priv, "public_key": pub})
    # NONE mode needs no keys

    for rnd in range(start_round, num_rounds):
        round_start = time.time()
        local_states, local_weights = [], []
        sign_times, verify_times, sig_sizes = [], [], []

        for i, cdata in enumerate(client_data[:num_clients]):
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.01, weight_decay=5e-4)
            local_model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                out = local_model(cdata["x"], cdata["edge_index"])
                loss = F.cross_entropy(out, cdata["y"], weight=class_weights)
                loss.backward()
                optimizer.step()

            state = local_model.state_dict()
            h = hash_model(state)

            if crypto_mode == "PQC":
                t0 = time.perf_counter()
                signature = client_keys[i]["signer"].sign(h)
                t1 = time.perf_counter()
                sign_times.append((t1 - t0) * 1000)
                sig_sizes.append(len(signature))

                t0 = time.perf_counter()
                verifier = oqs.Signature(SIG_ALG)
                is_valid = verifier.verify(h, signature, client_keys[i]["public_key"])
                t1 = time.perf_counter()
                verify_times.append((t1 - t0) * 1000)

            elif crypto_mode == "ECDSA":
                priv = client_keys[i]["private_key"]
                pub = client_keys[i]["public_key"]
                t0 = time.perf_counter()
                signature = priv.sign(h, ec.ECDSA(hashes.SHA256()))
                t1 = time.perf_counter()
                sign_times.append((t1 - t0) * 1000)
                sig_sizes.append(len(signature))

                t0 = time.perf_counter()
                try:
                    pub.verify(signature, h, ec.ECDSA(hashes.SHA256()))
                    is_valid = True
                except InvalidSignature:
                    is_valid = False
                t1 = time.perf_counter()
                verify_times.append((t1 - t0) * 1000)

            else:  # NONE
                sign_times.append(0.0)
                verify_times.append(0.0)
                sig_sizes.append(0)
                is_valid = True

            if not is_valid:
                print(f"  Client {i}: SIGNATURE INVALID — rejected")
                continue

            local_states.append(state)
            local_weights.append(len(cdata["node_idx"]))

        total_weight = sum(local_weights)
        new_state = {}
        for key in local_states[0].keys():
            new_state[key] = sum(local_states[i][key] * (local_weights[i] / total_weight)
                                  for i in range(len(local_states)))
        global_model.load_state_dict(new_state)

        global_model.eval()
        with torch.no_grad():
            out = global_model(x.to(device), edge_index.to(device))
            preds = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)[:, 1]
            test_y = y.to(device)[test_mask_global].cpu()
            test_preds = preds[test_mask_global].cpu()
            test_probs = probs[test_mask_global].cpu()

            test_acc = accuracy_score(test_y, test_preds)
            precision = precision_score(test_y, test_preds, pos_label=1, zero_division=0)
            recall = recall_score(test_y, test_preds, pos_label=1, zero_division=0)
            f1 = f1_score(test_y, test_preds, pos_label=1, zero_division=0)
            try:
                auc = roc_auc_score(test_y, test_probs)
            except ValueError:
                auc = float("nan")

        round_time = time.time() - round_start
        history["round"].append(rnd)
        history["test_acc"].append(test_acc)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["f1"].append(f1)
        history["auc"].append(auc)
        history["avg_sign_time_ms"].append(sum(sign_times) / len(sign_times))
        history["avg_verify_time_ms"].append(sum(verify_times) / len(verify_times))
        history["sig_size_bytes"].append(sig_sizes[0])
        history["round_time_s"].append(round_time)

        print(f"[{crypto_mode}] Round {rnd:2d} | acc {test_acc:.4f} | recall {recall:.4f} | f1 {f1:.4f} | "
              f"auc {auc:.4f} | sign {history['avg_sign_time_ms'][-1]:.3f}ms | "
              f"verify {history['avg_verify_time_ms'][-1]:.3f}ms | sig {sig_sizes[0]}B")

        torch.save({"round": rnd, "model_state": global_model.state_dict()}, ckpt_path)
        with open(history_path, "w") as f:
            json.dump(history, f)

    print(f"[{crypto_mode}] Done.")
    return history

# ---- Run all three configs ----
results = {}
for mode in ["PQC", "ECDSA", "NONE"]:
    results[mode] = run_fl_experiment(mode, num_clients=3, num_rounds=20)
```


### Original notebook cell 10

```python
import matplotlib.pyplot as plt
import pandas as pd

# ---- Summary table (final round for each config) ----
summary_rows = []
for mode in ["PQC", "ECDSA", "NONE"]:
    h = results[mode]
    summary_rows.append({
        "Config": mode,
        "Final Test Acc": h["test_acc"][-1],
        "Final Recall": h["recall"][-1],
        "Final F1": h["f1"][-1],
        "Final AUC": h["auc"][-1],
        "Avg Sign Time (ms)": sum(h["avg_sign_time_ms"]) / len(h["avg_sign_time_ms"]),
        "Avg Verify Time (ms)": sum(h["avg_verify_time_ms"]) / len(h["avg_verify_time_ms"]),
        "Signature Size (B)": h["sig_size_bytes"][-1],
        "Avg Round Time (s)": sum(h["round_time_s"]) / len(h["round_time_s"]) if "round_time_s" in h else None,
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv("/kaggle/working/fl_checkpoints/comparison_summary.csv", index=False)

# ---- Plot 1: Detection performance across rounds ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for mode in ["PQC", "ECDSA", "NONE"]:
    axes[0].plot(results[mode]["round"], results[mode]["test_acc"], label=mode)
    axes[1].plot(results[mode]["round"], results[mode]["f1"], label=mode)
    axes[2].plot(results[mode]["round"], results[mode]["auc"], label=mode)
axes[0].set_title("Test Accuracy"); axes[0].set_xlabel("Round"); axes[0].legend()
axes[1].set_title("F1 Score"); axes[1].set_xlabel("Round"); axes[1].legend()
axes[2].set_title("AUC-ROC"); axes[2].set_xlabel("Round"); axes[2].legend()
plt.tight_layout()
plt.savefig("/kaggle/working/fl_checkpoints/detection_comparison.png", dpi=150)
plt.show()

# ---- Plot 2: Cryptographic overhead bar chart ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
modes = ["PQC", "ECDSA", "NONE"]
colors = ["#1F3864", "#C55A11", "#548235"]

sign_avgs = [sum(results[m]["avg_sign_time_ms"])/len(results[m]["avg_sign_time_ms"]) for m in modes]
verify_avgs = [sum(results[m]["avg_verify_time_ms"])/len(results[m]["avg_verify_time_ms"]) for m in modes]
sig_sizes = [results[m]["sig_size_bytes"][-1] for m in modes]

axes[0].bar(modes, sign_avgs, color=colors); axes[0].set_title("Avg Sign Time (ms)")
axes[1].bar(modes, verify_avgs, color=colors); axes[1].set_title("Avg Verify Time (ms)")
axes[2].bar(modes, sig_sizes, color=colors); axes[2].set_title("Signature Size (Bytes)")
plt.tight_layout()
plt.savefig("/kaggle/working/fl_checkpoints/crypto_overhead_comparison.png", dpi=150)
plt.show()
```


### Original notebook cell 11

```python
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, 2)
        self.dropout = 0.4

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)
```


### Original notebook cell 12

```python
# ============================================================
# CELL 1: Imports and Data Loading
# ============================================================
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import hashlib, time, os, json, copy
import oqs

feat_path = "/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
class_path = "/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
edge_path = "/kaggle/input/datasets/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"

features = pd.read_csv(feat_path, header=None)
classes = pd.read_csv(class_path)
edges = pd.read_csv(edge_path)

features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
classes.columns = ["txId", "class"]

merged = features.merge(classes, on="txId", how="left")
merged["class"] = merged["class"].map({"1": 1, "2": 0, "unknown": -1})

node_id_map = {txid: idx for idx, txid in enumerate(merged["txId"])}
edges["src"] = edges["txId1"].map(node_id_map)
edges["dst"] = edges["txId2"].map(node_id_map)
edges = edges.dropna()

x = torch.tensor(merged.iloc[:, 2:167].values, dtype=torch.float)
y = torch.tensor(merged["class"].values, dtype=torch.long)
time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)
edge_index = torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long)

labeled_mask = y != -1
print(f"Loaded: {x.shape[0]} nodes, {edge_index.shape[1]} edges, {labeled_mask.sum().item()} labeled")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# CELL 2: Client Partitioning (simulated institutions)
# ============================================================
NUM_CLIENTS_MAX = 30  # build the max split up front; we'll slice subsets for 3/10/30 experiments later

labeled_idx = torch.where(labeled_mask)[0]
sorted_idx = labeled_idx[torch.argsort(time_step[labeled_idx])]

def build_client_data(num_clients):
    splits = torch.chunk(sorted_idx, num_clients)
    cdata_list = []
    for idxs in splits:
        sub_edge_index, _ = subgraph(idxs, edge_index, relabel_nodes=True, num_nodes=x.shape[0])
        cdata_list.append({
            "node_idx": idxs,
            "x": x[idxs].to(device),
            "y": y[idxs].to(device),
            "edge_index": sub_edge_index.to(device),
        })
    return cdata_list

client_data_3 = build_client_data(3)
print(f"Built 3-client split: {[len(c['node_idx']) for c in client_data_3]}")

test_mask_global = (labeled_mask & (time_step > 34)).to(device)

# ============================================================
# CELL 3: Tuned GCN Model
# ============================================================
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, 2)
        self.dropout = 0.4

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)

class_weights = torch.tensor([1.0, 6.0]).to(device)
SIG_ALG = "ML-DSA-65"

def hash_model(state_dict):
    buf = b"".join(t.cpu().numpy().tobytes() for t in state_dict.values())
    return hashlib.sha3_256(buf).digest()

# ============================================================
# CELL 4: Reusable FL Experiment Function
# ============================================================
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

def run_fl_experiment(crypto_mode, client_data, num_rounds=40, local_epochs=10,
                       ckpt_dir="/kaggle/working/fl_tuned", tag="", seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = f"{crypto_mode}_{tag}_seed{seed}" if tag else f"{crypto_mode}_seed{seed}"
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    history_path = os.path.join(ckpt_dir, f"{run_name}_history.json")

    num_clients = len(client_data)
    global_model = GCN(in_dim=x.shape[1]).to(device)
    history = {"round": [], "test_acc": [], "precision": [], "recall": [], "f1": [], "auc": [],
               "avg_sign_time_ms": [], "avg_verify_time_ms": [], "sig_size_bytes": [], "round_time_s": []}
    start_round = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        global_model.load_state_dict(ckpt["model_state"])
        start_round = ckpt["round"] + 1
        with open(history_path, "r") as f:
            history = json.load(f)
        print(f"[{run_name}] Resumed from round {start_round}")
    else:
        print(f"[{run_name}] Starting fresh ({num_clients} clients)")

    if crypto_mode == "PQC":
        client_keys = []
        for i in range(num_clients):
            signer = oqs.Signature(SIG_ALG)
            pub = signer.generate_keypair()
            client_keys.append({"signer": signer, "public_key": pub})
    elif crypto_mode == "ECDSA":
        client_keys = []
        for i in range(num_clients):
            priv = ec.generate_private_key(ec.SECP256K1())
            pub = priv.public_key()
            client_keys.append({"private_key": priv, "public_key": pub})

    for rnd in range(start_round, num_rounds):
        round_start = time.time()
        local_states, local_weights = [], []
        sign_times, verify_times, sig_sizes = [], [], []

        for i, cdata in enumerate(client_data):
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.01, weight_decay=5e-4)
            local_model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                out = local_model(cdata["x"], cdata["edge_index"])
                loss = F.cross_entropy(out, cdata["y"], weight=class_weights)
                loss.backward()
                optimizer.step()

            state = local_model.state_dict()
            h = hash_model(state)

            if crypto_mode == "PQC":
                t0 = time.perf_counter()
                signature = client_keys[i]["signer"].sign(h)
                t1 = time.perf_counter()
                sign_times.append((t1 - t0) * 1000)
                sig_sizes.append(len(signature))
                t0 = time.perf_counter()
                verifier = oqs.Signature(SIG_ALG)
                is_valid = verifier.verify(h, signature, client_keys[i]["public_key"])
                t1 = time.perf_counter()
                verify_times.append((t1 - t0) * 1000)

            elif crypto_mode == "ECDSA":
                priv = client_keys[i]["private_key"]
                pub = client_keys[i]["public_key"]
                t0 = time.perf_counter()
                signature = priv.sign(h, ec.ECDSA(hashes.SHA256()))
                t1 = time.perf_counter()
                sign_times.append((t1 - t0) * 1000)
                sig_sizes.append(len(signature))
                t0 = time.perf_counter()
                try:
                    pub.verify(signature, h, ec.ECDSA(hashes.SHA256()))
                    is_valid = True
                except InvalidSignature:
                    is_valid = False
                t1 = time.perf_counter()
                verify_times.append((t1 - t0) * 1000)

            else:  # NONE
                sign_times.append(0.0)
                verify_times.append(0.0)
                sig_sizes.append(0)
                is_valid = True

            if not is_valid:
                print(f"  Client {i}: SIGNATURE INVALID — rejected")
                continue

            local_states.append(state)
            local_weights.append(len(cdata["node_idx"]))

        total_weight = sum(local_weights)
        new_state = {}
        for key in local_states[0].keys():
            new_state[key] = sum(local_states[j][key] * (local_weights[j] / total_weight)
                                  for j in range(len(local_states)))
        global_model.load_state_dict(new_state)

        global_model.eval()
        with torch.no_grad():
            out = global_model(x.to(device), edge_index.to(device))
            preds = out.argmax(dim=1)
            probs = F.softmax(out, dim=1)[:, 1]
            test_y = y.to(device)[test_mask_global].cpu()
            test_preds = preds[test_mask_global].cpu()
            test_probs = probs[test_mask_global].cpu()

            test_acc = accuracy_score(test_y, test_preds)
            precision = precision_score(test_y, test_preds, pos_label=1, zero_division=0)
            recall = recall_score(test_y, test_preds, pos_label=1, zero_division=0)
            f1 = f1_score(test_y, test_preds, pos_label=1, zero_division=0)
            try:
                auc = roc_auc_score(test_y, test_probs)
            except ValueError:
                auc = float("nan")

        round_time = time.time() - round_start
        history["round"].append(rnd)
        history["test_acc"].append(test_acc)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["f1"].append(f1)
        history["auc"].append(auc)
        history["avg_sign_time_ms"].append(sum(sign_times) / len(sign_times))
        history["avg_verify_time_ms"].append(sum(verify_times) / len(verify_times))
        history["sig_size_bytes"].append(sig_sizes[0])
        history["round_time_s"].append(round_time)

        print(f"[{run_name}] Round {rnd:2d} | acc {test_acc:.4f} | recall {recall:.4f} | "
              f"f1 {f1:.4f} | auc {auc:.4f}")

        torch.save({"round": rnd, "model_state": global_model.state_dict()}, ckpt_path)
        with open(history_path, "w") as f:
            json.dump(history, f)

    print(f"[{run_name}] Done.")
    return history

# ============================================================
# CELL 5: RUN THIS — Quick tuning test (PQC only, 3 clients)
# ============================================================
results_tuned = {}
results_tuned["PQC"] = run_fl_experiment("PQC", client_data_3, num_rounds=40,
                                          local_epochs=10, ckpt_dir="/kaggle/working/fl_tuned")

# Print best F1 achieved
best_idx = int(np.argmax(results_tuned["PQC"]["f1"]))
print(f"\nBest F1: {results_tuned['PQC']['f1'][best_idx]:.4f} at round {results_tuned['PQC']['round'][best_idx]} "
      f"(recall={results_tuned['PQC']['recall'][best_idx]:.4f}, auc={results_tuned['PQC']['auc'][best_idx]:.4f})")
```


### Original notebook cell 13

```python
# ============================================================
# CELL 6: Build client splits for 3, 10, 30 clients
# ============================================================
client_data_10 = build_client_data(10)
client_data_30 = build_client_data(30)
print(f"10-client split sizes: {[len(c['node_idx']) for c in client_data_10]}")
print(f"30-client split sizes: {[len(c['node_idx']) for c in client_data_30]}")

CLIENT_SPLITS = {3: client_data_3, 10: client_data_10, 30: client_data_30}

# ============================================================
# CELL 7: Full experiment matrix — RUN THIS (will take a while)
# ============================================================
CONFIGS = ["PQC", "ECDSA", "NONE"]
CLIENT_COUNTS = [3, 10, 30]
SEEDS = [0, 1, 2]
NUM_ROUNDS = 40
LOCAL_EPOCHS = 10

all_results = {}

for num_clients in CLIENT_COUNTS:
    cdata = CLIENT_SPLITS[num_clients]
    for mode in CONFIGS:
        for seed in SEEDS:
            key = f"{mode}_c{num_clients}_s{seed}"
            print(f"\n{'='*60}\nRunning: {key}\n{'='*60}")
            all_results[key] = run_fl_experiment(
                crypto_mode=mode,
                client_data=cdata,
                num_rounds=NUM_ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                ckpt_dir="/kaggle/working/fl_full_matrix",
                tag=f"c{num_clients}",
                seed=seed
            )

# Save the full raw results as one JSON for safekeeping
with open("/kaggle/working/fl_full_matrix/all_results.json", "w") as f:
    json.dump(all_results, f)

print("\n\nFULL MATRIX COMPLETE:", len(all_results), "runs finished")
```


### Original notebook cell 14

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CONFIGS = ["PQC", "ECDSA", "NONE"]
CLIENT_COUNTS = [3, 10, 30]
SEEDS = [0, 1, 2]

rows = []
for num_clients in CLIENT_COUNTS:
    for mode in CONFIGS:
        for seed in SEEDS:
            key = f"{mode}_c{num_clients}_s{seed}"
            h = all_results[key]
            rows.append({
                "crypto_mode": mode, "num_clients": num_clients, "seed": seed,
                "final_test_acc": h["test_acc"][-1],
                "final_recall": h["recall"][-1],
                "final_f1": h["f1"][-1],
                "final_auc": h["auc"][-1],
                "avg_sign_ms": np.mean(h["avg_sign_time_ms"]),
                "avg_verify_ms": np.mean(h["avg_verify_time_ms"]),
                "sig_size_bytes": h["sig_size_bytes"][-1],
                "avg_round_time_s": np.mean(h["round_time_s"]),
            })

raw_df = pd.DataFrame(rows)
raw_df.to_csv("/kaggle/working/fl_full_matrix/raw_results_27runs.csv", index=False)

# ---- Aggregated mean ± std across seeds ----
agg = raw_df.groupby(["crypto_mode", "num_clients"]).agg(
    acc_mean=("final_test_acc", "mean"), acc_std=("final_test_acc", "std"),
    recall_mean=("final_recall", "mean"), recall_std=("final_recall", "std"),
    f1_mean=("final_f1", "mean"), f1_std=("final_f1", "std"),
    auc_mean=("final_auc", "mean"), auc_std=("final_auc", "std"),
    sign_ms_mean=("avg_sign_ms", "mean"),
    verify_ms_mean=("avg_verify_ms", "mean"),
    sig_size=("sig_size_bytes", "first"),
    round_time_mean=("avg_round_time_s", "mean"),
).reset_index()

for col in ["acc", "recall", "f1", "auc"]:
    agg[f"{col}_reported"] = agg.apply(
        lambda r: f"{r[f'{col}_mean']:.4f} ± {r[f'{col}_std']:.4f}", axis=1)

print(agg[["crypto_mode", "num_clients", "acc_reported", "recall_reported",
           "f1_reported", "auc_reported", "sign_ms_mean", "verify_ms_mean",
           "sig_size", "round_time_mean"]].to_string(index=False))

agg.to_csv("/kaggle/working/fl_full_matrix/aggregated_summary.csv", index=False)

# ---- Plot 1: Scalability — F1 and Recall vs client count, per config ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for mode in CONFIGS:
    sub = agg[agg["crypto_mode"] == mode]
    axes[0].errorbar(sub["num_clients"], sub["f1_mean"], yerr=sub["f1_std"],
                      marker="o", label=mode, capsize=4)
    axes[1].errorbar(sub["num_clients"], sub["recall_mean"], yerr=sub["recall_std"],
                      marker="o", label=mode, capsize=4)
axes[0].set_xlabel("Number of Clients"); axes[0].set_ylabel("F1 Score")
axes[0].set_title("F1 vs. Client Count"); axes[0].legend()
axes[1].set_xlabel("Number of Clients"); axes[1].set_ylabel("Recall")
axes[1].set_title("Recall vs. Client Count"); axes[1].legend()
plt.tight_layout()
plt.savefig("/kaggle/working/fl_full_matrix/scalability_plot.png", dpi=150)
plt.show()

# ---- Plot 2: Crypto overhead vs client count ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for mode in ["PQC", "ECDSA"]:
    sub = agg[agg["crypto_mode"] == mode]
    axes[0].plot(sub["num_clients"], sub["sign_ms_mean"], marker="o", label=mode)
    axes[1].plot(sub["num_clients"], sub["verify_ms_mean"], marker="o", label=mode)
for mode in CONFIGS:
    sub = agg[agg["crypto_mode"] == mode]
    axes[2].plot(sub["num_clients"], sub["round_time_mean"], marker="o", label=mode)
axes[0].set_title("Avg Sign Time (ms) vs Clients"); axes[0].set_xlabel("Clients"); axes[0].legend()
axes[1].set_title("Avg Verify Time (ms) vs Clients"); axes[1].set_xlabel("Clients"); axes[1].legend()
axes[2].set_title("Avg Round Time (s) vs Clients"); axes[2].set_xlabel("Clients"); axes[2].legend()
plt.tight_layout()
plt.savefig("/kaggle/working/fl_full_matrix/overhead_scalability_plot.png", dpi=150)
plt.show()

print("\nAll outputs saved to /kaggle/working/fl_full_matrix/")
```


---
## 4. Newer federated-learning experiment (v2)


### Original notebook cell 17

```python
#new run with changes
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np, torch, torch.nn.functional as F, time, copy, json, os, hashlib, oqs
from torch_geometric.utils import subgraph
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidSignature

tr_mask = labeled_mask & (time_step <= 27)
val_mask_g = labeled_mask & (time_step >= 28) & (time_step <= 34)
test_mask_g = labeled_mask & (time_step >= 35)
print("train/val/test labeled:", tr_mask.sum().item(), val_mask_g.sum().item(), test_mask_g.sum().item())

tr_idx = torch.where(tr_mask)[0]
sorted_idx = tr_idx[torch.argsort(time_step[tr_idx])]

def build_client_data(n):
    out = []
    for idxs in torch.chunk(sorted_idx, n):
        sei, _ = subgraph(idxs, edge_index, relabel_nodes=True, num_nodes=x.shape[0])
        out.append({"node_idx": idxs, "x": x[idxs].to(device), "y": y[idxs].to(device), "edge_index": sei.to(device)})
    return out

CLIENT_SPLITS = {n: build_client_data(n) for n in [3, 10, 30]}
X_ALL, EI_ALL = x.to(device), edge_index.to(device)
yc = y.cpu().numpy(); tsc = time_step.cpu().numpy()
vm, tm = val_mask_g.numpy(), test_mask_g.numpy()

def metrics_at(yt, pt, t):
    pred = (pt > t).astype(int)
    return dict(acc=accuracy_score(yt, pred), prec=precision_score(yt, pred, zero_division=0),
                rec=recall_score(yt, pred, zero_division=0), f1=f1_score(yt, pred, zero_division=0))

def evaluate_probs(p):
    """p: numpy illicit-probability for ALL nodes. Threshold chosen on val, applied to test."""
    ths = np.linspace(0.05, 0.95, 91)
    bt = max(ths, key=lambda t: f1_score(yc[vm], (p[vm] > t).astype(int), zero_division=0))
    yt, pt = yc[tm], p[tm]
    r = {"thr": float(bt), "auc": float(roc_auc_score(yt, pt))}
    r["at_0.5"] = metrics_at(yt, pt, 0.5)
    r["tuned"] = metrics_at(yt, pt, bt)
    ts = tsc[tm]
    for name, sel in [("steps35_42", ts <= 42), ("steps43_49", ts >= 43)]:
        r[name] = metrics_at(yt[sel], pt[sel], bt)
    return r
```


### Original notebook cell 18

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
Xn = x.numpy()
trm = tr_mask.numpy()
baselines = {}
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0).fit(Xn[trm], yc[trm])
baselines["RF (centralized)"] = evaluate_probs(rf.predict_proba(Xn)[:, 1])
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xn[trm], yc[trm])
baselines["LogReg (centralized)"] = evaluate_probs(lr.predict_proba(Xn)[:, 1])
for k, v in baselines.items():
    print(k, "AUC %.3f" % v["auc"], "| tuned F1 %.3f rec %.3f" % (v["tuned"]["f1"], v["tuned"]["rec"]),
          "| @0.5 F1 %.3f rec %.3f" % (v["at_0.5"]["f1"], v["at_0.5"]["rec"]))
json.dump(baselines, open("/kaggle/working/baselines_central.json", "w"))
```


### Original notebook cell 19

```python
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.c1, self.c2, self.c3 = GCNConv(in_dim, hidden), GCNConv(hidden, hidden // 2), GCNConv(hidden // 2, 2)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.c1(x, ei)), 0.4, self.training)
        x = F.dropout(F.relu(self.c2(x, ei)), 0.4, self.training)
        return self.c3(x, ei)

class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.l1, self.l2, self.l3 = torch.nn.Linear(in_dim, hidden), torch.nn.Linear(hidden, hidden // 2), torch.nn.Linear(hidden // 2, 2)
    def forward(self, x, ei=None):
        x = F.dropout(F.relu(self.l1(x)), 0.4, self.training)
        x = F.dropout(F.relu(self.l2(x)), 0.4, self.training)
        return self.l3(x)

MODELS = {"GCN": GCN, "MLP": MLP}
CLASS_W = torch.tensor([1.0, 4.0]).to(device)
CRYPTO_MODES = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "ECDSA", "NONE"]

def hash_model(sd):
    return hashlib.sha3_256(b"".join(t.cpu().numpy().tobytes() for t in sd.values())).digest()
```


### Original notebook cell 20

```python
def make_signers(n):
    ks = {}
    for m in CRYPTO_MODES:
        if m.startswith("ML-DSA"):
            lst = []
            for _ in range(n):
                s = oqs.Signature(m); pub = s.generate_keypair(); lst.append((s, oqs.Signature(m), pub))
            ks[m] = lst
        elif m == "ECDSA":
            ks[m] = [ec.generate_private_key(ec.SECP256K1()) for _ in range(n)]
    return ks

def sign_verify(mode, ks, i, h):
    if mode == "NONE":
        return 0.0, 0.0, 0, True
    if mode == "ECDSA":
        priv = ks[mode][i]; pub = priv.public_key()
        t0 = time.perf_counter(); sig = priv.sign(h, ec.ECDSA(hashes.SHA256())); t1 = time.perf_counter()
        try: pub.verify(sig, h, ec.ECDSA(hashes.SHA256())); ok = True
        except InvalidSignature: ok = False
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, len(sig), ok
    s, v, pub = ks[mode][i]
    t0 = time.perf_counter(); sig = s.sign(h); t1 = time.perf_counter()
    ok = v.verify(h, sig, pub); t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3, len(sig), ok

def run_fl_v2(model_name, n_clients, seed, rounds=60, local_epochs=10, out_dir="/kaggle/working/fl_v2"):
    os.makedirs(out_dir, exist_ok=True)
    name = f"{model_name}_c{n_clients}_s{seed}"
    path, hpath = f"{out_dir}/{name}.pt", f"{out_dir}/{name}.json"
    torch.manual_seed(seed); np.random.seed(seed)
    gm = MODELS[model_name](x.shape[1]).to(device)
    hist = {"eval": [], "train_s": [], "crypto": {m: [] for m in CRYPTO_MODES}}
    start = 0
    if os.path.exists(path):
        gm.load_state_dict(torch.load(path, map_location=device)["m"]); hist = json.load(open(hpath)); start = len(hist["eval"])
        print(f"[{name}] resumed at round {start}")
    ks = make_signers(n_clients)
    cdata = CLIENT_SPLITS[n_clients]
    for rnd in range(start, rounds):
        t_round = time.time(); states, weights = [], []
        cr = {m: {"sign": 0.0, "verify": 0.0, "bytes": 0, "ok": True} for m in CRYPTO_MODES}
        for i, c in enumerate(cdata):
            lm = copy.deepcopy(gm); opt = torch.optim.Adam(lm.parameters(), lr=0.01, weight_decay=5e-4); lm.train()
            for _ in range(local_epochs):
                opt.zero_grad()
                F.cross_entropy(lm(c["x"], c["edge_index"]), c["y"], weight=CLASS_W).backward()
                opt.step()
            sd = lm.state_dict(); h = hash_model(sd)
            for m in CRYPTO_MODES:
                s_ms, v_ms, nb, ok = sign_verify(m, ks, i, h)
                cr[m]["sign"] += s_ms; cr[m]["verify"] += v_ms; cr[m]["bytes"] = nb; cr[m]["ok"] &= ok
            states.append(sd); weights.append(len(c["node_idx"]))
        crypto_total_ms = sum(cr[m]["sign"] + cr[m]["verify"] for m in CRYPTO_MODES)
        train_s = time.time() - t_round - crypto_total_ms / 1e3   # training+aggregation time excluding all crypto
        tw = sum(weights)
        gm.load_state_dict({k: sum(states[j][k] * (weights[j] / tw) for j in range(len(states))) for k in states[0]})
        gm.eval()
        with torch.no_grad():
            p = F.softmax(gm(X_ALL, EI_ALL), dim=1)[:, 1].cpu().numpy()
        ev = evaluate_probs(p); hist["eval"].append(ev); hist["train_s"].append(train_s)
        for m in CRYPTO_MODES:
            hist["crypto"][m].append({"sign_ms_per_client": cr[m]["sign"] / n_clients, "verify_ms_per_client": cr[m]["verify"] / n_clients,
                                      "sig_bytes": cr[m]["bytes"], "total_ms": cr[m]["sign"] + cr[m]["verify"], "all_valid": cr[m]["ok"]})
        if rnd % 5 == 0 or rnd == rounds - 1:
            print(f"[{name}] r{rnd:2d} AUC {ev['auc']:.3f} | @0.5 F1 {ev['at_0.5']['f1']:.3f} | tuned F1 {ev['tuned']['f1']:.3f} rec {ev['tuned']['rec']:.3f} thr {ev['thr']:.2f}")
        torch.save({"m": gm.state_dict()}, path); json.dump(hist, open(hpath, "w"))
    return hist
```


### Original notebook cell 21

```python
h = run_fl_v2("MLP", 3, 0, rounds=30)
h = run_fl_v2("GCN", 3, 0, rounds=30)
```


### Original notebook cell 22

```python
results_v2 = {}
for model_name in ["GCN", "MLP"]:
    for n in [3, 10, 30]:
        for seed in [0, 1, 2]:
            results_v2[f"{model_name}_c{n}_s{seed}"] = run_fl_v2(model_name, n, seed, rounds=60)
```


### Original notebook cell 23

```python
import pandas as pd
rows = []
for key, h in results_v2.items():
    mdl, c, s = key.split("_"); ev = h["eval"][-1]
    rows.append({"model": mdl, "clients": int(c[1:]), "seed": int(s[1:]), "auc": ev["auc"],
                 "f1_0.5": ev["at_0.5"]["f1"], "rec_0.5": ev["at_0.5"]["rec"],
                 "f1_tuned": ev["tuned"]["f1"], "rec_tuned": ev["tuned"]["rec"], "prec_tuned": ev["tuned"]["prec"],
                 "f1_35_42": ev["steps35_42"]["f1"], "f1_43_49": ev["steps43_49"]["f1"]})
det = pd.DataFrame(rows).groupby(["model", "clients"]).agg(["mean", "std"]).round(4)
print(det.to_string()); det.to_csv("/kaggle/working/fl_v2/detection_summary.csv")

crow = []
for key, h in results_v2.items():
    mdl, c, s = key.split("_")
    tr = np.mean(h["train_s"])
    for m in CRYPTO_MODES:
        L = h["crypto"][m]
        tot = np.mean([r["total_ms"] for r in L])
        crow.append({"model": mdl, "clients": int(c[1:]), "mode": m,
                     "sign_ms": np.mean([r["sign_ms_per_client"] for r in L]), "verify_ms": np.mean([r["verify_ms_per_client"] for r in L]),
                     "sig_bytes": L[-1]["sig_bytes"], "overhead_pct": 100 * tot / (tr * 1e3 + tot), "all_valid": all(r["all_valid"] for r in L)})
cry = pd.DataFrame(crow).groupby(["clients", "mode"]).mean(numeric_only=True).round(4)
print(cry.to_string()); cry.to_csv("/kaggle/working/fl_v2/crypto_summary.csv")
```


---
## 5. Evaluation and analysis


### Original notebook cell 24

```python
print("Split counts:")
print("Train:", tr_mask.sum())
print("Validation:", val_mask_g.sum())
print("Test:", test_mask_g.sum())

print("\nCentralized baselines:")
for name, r in baselines.items():
    print(name, r)
```


### Original notebook cell 25

```python
# Fraud prevalence by time period
for name, lo, hi in [
    ("Train", 1, 27),
    ("Validation", 28, 34),
    ("Test early", 35, 42),
    ("Test late", 43, 49),
]:
    mask = (tsc >= lo) & (tsc <= hi) & (yc != -1)
    labels = yc[mask]
    print(
        name,
        "| nodes:", len(labels),
        "| fraud:", int((labels == 1).sum()),
        "| fraud rate:", round((labels == 1).mean() * 100, 2), "%"
    )
```


### Original notebook cell 26

```python
#cell 8
from torch_geometric.utils import subgraph
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import torch
import torch.nn.functional as F
import os

def predict_temporal(model):
    model.eval()
    probs = np.full(len(yc), np.nan, dtype=float)

    with torch.no_grad():
        for step in range(28, 50):
            # Only nodes available by this time step
            nodes = torch.where(time_step <= step)[0]
            targets = torch.where(time_step == step)[0]

            ei, _ = subgraph(
                nodes, edge_index,
                relabel_nodes=True,
                num_nodes=x.shape[0]
            )

            # Map global node IDs to local subgraph IDs
            mapping = torch.full((x.shape[0],), -1, dtype=torch.long)
            mapping[nodes] = torch.arange(len(nodes))
            local_targets = mapping[targets]

            logits = model(
                x[nodes].to(device),
                ei.to(device)
            )

            probs[targets.numpy()] = F.softmax(
                logits[local_targets.to(device)], dim=1
            )[:, 1].cpu().numpy()

    return probs


def evaluate_temporal(p):
    # Select threshold ONLY from validation data
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = max(
        thresholds,
        key=lambda t: f1_score(
            yc[vm], (p[vm] > t).astype(int),
            zero_division=0
        )
    )

    for name, mask in [
        ("Validation", vm),
        ("Test overall", tm),
        ("Test steps 35-42", tm & (tsc <= 42)),
        ("Test steps 43-49", tm & (tsc >= 43))
    ]:
        yt = yc[mask]
        pt = p[mask]

        valid = np.isfinite(pt)
        yt, pt = yt[valid], pt[valid]

        pred = (pt > best_threshold).astype(int)

        print(
            f"{name}: "
            f"F1={f1_score(yt, pred, zero_division=0):.4f}, "
            f"Recall={recall_score(yt, pred, zero_division=0):.4f}, "
            f"Precision={precision_score(yt, pred, zero_division=0):.4f}"
        )

    print("Validation-selected threshold:", best_threshold)
```


### Original notebook cell 27

```python
#cell 9
from sklearn.metrics import precision_score, recall_score

name = "GCN_c30_s0"
model = MODELS["GCN"](x.shape[1]).to(device)
ckpt = torch.load(
    f"/kaggle/working/fl_v2/{name}.pt",
    map_location=device
)
model.load_state_dict(ckpt["m"])

p = predict_temporal(model)
evaluate_temporal(p)
```


### Original notebook cell 28

```python
#cell 10

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
import numpy as np
import torch

def evaluate_checkpoint(model_name, clients, seed):
    model = MODELS[model_name](x.shape[1]).to(device)
    path = f"/kaggle/working/fl_v2/{model_name}_c{clients}_s{seed}.pt"

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["m"])
    p = predict_temporal(model)

    print(f"\n{model_name} | clients={clients} | seed={seed}")

    # Threshold chosen using validation data only
    valid_val = vm & np.isfinite(p)
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold = max(
        thresholds,
        key=lambda t: f1_score(
            yc[valid_val], (p[valid_val] >= t).astype(int),
            zero_division=0
        )
    )
    print(f"Validation-selected threshold: {threshold:.2f}")

    for label, mask in [
        ("Validation", vm),
        ("Test overall", tm),
        ("Test 35-42", tm & (tsc <= 42)),
        ("Test 43-49", tm & (tsc >= 43))
    ]:
        mask = mask & np.isfinite(p)
        yt, pt = yc[mask], p[mask]

        if len(np.unique(yt)) < 2:
            print(label, ": only one class; skipping AUC")
            continue

        print(
            f"{label}: "
            f"AUC={roc_auc_score(yt, pt):.4f}, "
            f"PR-AUC={average_precision_score(yt, pt):.4f}, "
            f"F1@0.5={f1_score(yt, pt >= 0.5, zero_division=0):.4f}, "
            f"F1@val_threshold={f1_score(yt, pt >= threshold, zero_division=0):.4f}, "
            f"Precision={precision_score(yt, pt >= threshold, zero_division=0):.4f}, "
            f"Recall={recall_score(yt, pt >= threshold, zero_division=0):.4f}"
        )

for model_name in ["GCN", "MLP"]:
    for clients in [3, 10, 30]:
        for seed in [0, 1, 2]:
            evaluate_checkpoint(model_name, clients, seed)
```
