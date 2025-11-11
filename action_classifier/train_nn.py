"""
Train a small neural network on SentenceTransformer embeddings built from action
metadata.
- Reuses the same metadata->text flattening as `train_classifier.py`.
- Uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings.
- Trains a tiny PyTorch MLP classifier.
- Saves the trained PyTorch model and the embedding model name used.
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical


def accuracy_score(y_true, y_pred):
    """Lightweight accuracy scorer used when sklearn is not available.

    Returns fraction of matching labels between y_true and y_pred.
    Works for iterables of hashable items (strings, ints).
    """
    # convert to lists to allow multiple iterator types
    yt = list(y_true)
    yp = list(y_pred)
    if len(yt) == 0:
        return 0.0
    # only compare up to the shorter length to avoid throwing
    n = min(len(yt), len(yp))
    matches = sum(1 for i in range(n) if yt[i] == yp[i])
    return matches / n

# ------------------ Configuration (disable parser) ------------------


def set_seed(seed: int = 42):
    """Make runs reproducible across torch/numpy/random where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic behaviour for cuDNN (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 8
EPOCHS = 6  # best found
LR = 2e-3  # best found
HIDDEN = 256  # best found
FORCE_CPU = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
set_seed()

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "sample_actions.json"
MODEL_PATH = ROOT / "emb_nn_model.pt"
_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))
    return _EMBED_MODEL


def get_label(entry):
    """Get 2 scores: harmful_score, unethical_score."""
    classification = (entry.get("classification") or "").lower()
    risk_level = (entry.get("risk_level") or "").lower()

    score = 0.0
    if risk_level == "low":
        score = 1.0
    elif risk_level == "medium":
        score = 2.0
    elif risk_level == "high":
        score = 3.0

    harmful_score = 0.0
    unethical_score = 0.0

    if classification == "harmful":
        # Harmful actions are also unethical
        harmful_score = score
        unethical_score = score
    elif classification == "unethical":
        unethical_score = score
    
    return [harmful_score, unethical_score]


def classify_from_scores(harmful_score, unethical_score):
    """Classify as safe, harmful, or unethical based on scores."""
    if harmful_score <= 0.5 and unethical_score <= 0.5:
        return "safe"
    elif harmful_score > unethical_score:
        return "harmful"
    else:
        return "unethical"


def flatten_action_to_text(entry):
    """Flatten action metadata to text."""
    parts = []
    parts.append(entry.get("label", ""))
    parts.append(entry.get("resource", ""))
    action_meta = entry.get("action", {}) or {}
    parts.append(action_meta.get("server_label", ""))
    server_url = action_meta.get("server_url", "")
    if server_url:
        try:
            host = re.sub(r"^https?://", "", server_url)
            host = host.split("/")[0]
            parts.append(host)
        except Exception:
            parts.append(server_url)
    parts.append(action_meta.get("require_approval", ""))
    params = action_meta.get("parameters") or {}
    if isinstance(params, dict):
        parts.extend(sorted(list(params.keys())))
    else:
        parts.append(str(type(params)))
    return " ".join([str(p) for p in parts if p])


class SimpleMLP(nn.Module):
    """Simple MLP for multi-class classification."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 2),  # 2 scores: harmful, unethical
        )

    def forward(self, x):
        """Forward pass."""
        return self.net(x)


def load_texts_and_labels():
    """Load texts and labels from dataset."""
    with open(DATA_PATH, encoding="utf-8") as f:
        j = json.load(f)
    items = j.get("actions", [])
    texts = []
    labels = []
    classes = []
    for it in items:
        texts.append(flatten_action_to_text(it))
        lbl = get_label(it)
        labels.append(lbl)
        # prefer explicit classification field when present, else derive from scores
        cls = (it.get("classification") or "").lower()
        if not cls:
            cls = classify_from_scores(lbl[0], lbl[1])
        classes.append(cls)
    return texts, labels, classes


def make_embeddings(texts):
    """Generate embeddings for texts."""
    # sentence-transformers returns numpy arrays
    model = _get_embed_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embs)


loss_fn = nn.MSELoss()

def train_one(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
              yte: np.ndarray, hidden: int, lr: float, epochs: int):
    """Train the model for one configuration.

    Returns the trained model, classification accuracy (based on mapping scores
    back to class labels), and the raw predicted scores for the test set.
    """
    set_seed()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleMLP(in_dim=Xtr.shape[1], hidden=hidden)
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    with torch.inference_mode():
        model.eval()
        logits = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE))
        preds = torch.round(logits).cpu().numpy()
        yte_np = np.array(yte)
        # Compute regression-style match (both scores within 0.5)
        reg_correct = np.all(np.abs(preds - yte_np) < 0.5, axis=1)
        reg_acc = np.mean(reg_correct)

        # Compute classification accuracy by mapping scores back to labels
        pred_classes = [classify_from_scores(float(p[0]), float(p[1])) for p in preds]
        true_classes = [classify_from_scores(float(t[0]), float(t[1])) for t in yte_np]
        cls_acc = accuracy_score(true_classes, pred_classes)

    # return classification accuracy as primary metric (used by hyperparam tuning)
    return model, cls_acc, preds


def _class_to_int(cls: str) -> int:
    mapping = {"safe": 0, "harmful": 1, "unethical": 2}
    return mapping.get(cls, 0)


def rl_fine_tune(
    xtr_embs: np.ndarray,
    xte_embs: np.ndarray,
    ytr_classes: list,
    yte_classes: list,
    hidden: int = 256,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 32,
    pretrained_state_dict: dict | None = None,
):
    """Simple REINFORCE fine-tuning on a small classifier head.

    x*_embs: numpy arrays of embeddings
    y*_classes: list/iterable of string class labels (safe/harmful/unethical)
    Returns (model, acc) where acc is classification accuracy on the test set.
    """
    set_seed()
    # convert classes to ints
    ytr_int = np.array([_class_to_int(c) for c in ytr_classes])
    yte_int = np.array([_class_to_int(c) for c in yte_classes])

    in_dim = xtr_embs.shape[1]
    clf = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden // 2, 3),
    )
    clf.to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    # If a pretrained supervised model state dict is provided, try to load
    # compatible weights (ignore final layer size mismatch).
    if pretrained_state_dict is not None:
        try:
            cur_state = clf.state_dict()
            # copy matching tensors
            for k, v in pretrained_state_dict.items():
                if k in cur_state and cur_state[k].shape == v.shape:
                    cur_state[k] = v
            clf.load_state_dict(cur_state)
            print("Loaded compatible pretrained weights into RL head (partial load).")
        except Exception:
            print("Warning: failed to load pretrained weights into RL head; continuing with random init.")

    n = xtr_embs.shape[0]
    idx = np.arange(n)
    baseline = 0.0
    for epoch in range(1, epochs + 1):
        # shuffle
        np.random.shuffle(idx)
        total_loss = 0.0
        total_samples = 0
        for i in range(0, n, batch_size):
            batch_idx = idx[i : i + batch_size]
            xb = torch.tensor(xtr_embs[batch_idx], dtype=torch.float32).to(DEVICE)
            yb = torch.tensor(ytr_int[batch_idx], dtype=torch.long).to(DEVICE)

            logits = clf(xb)
            probs = torch.softmax(logits, dim=1)
            dist = Categorical(probs)
            actions = dist.sample()
            logp = dist.log_prob(actions)

            rewards = (actions == yb).float()
            adv = rewards - baseline
            loss = -(logp * adv).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

            # update baseline (simple moving average)
            baseline = 0.99 * baseline + 0.01 * rewards.mean().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"RL Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - baseline: {baseline:.4f}")

    # evaluate greedily on test set
    with torch.inference_mode():
        logits = clf(torch.tensor(xte_embs, dtype=torch.float32).to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        acc = (preds == yte_int).mean()

    return clf, float(acc)


def train_model():
    """Train the model with best hyperparameters."""
    # Ensure reproducible split and training
    set_seed()
    texts, labels, classes = load_texts_and_labels()
    Xtr_texts, Xte_texts, ytr, yte = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=classes
    )

    print("Generating embeddings...")
    # embeddings should be deterministic given seed and model
    set_seed()
    Xtr_embs = make_embeddings(Xtr_texts)
    Xte_embs = make_embeddings(Xte_texts)

    # Use best values discovered interactively
    print("Training...")
    cfg = {"hidden": HIDDEN, "lr": LR, "epochs": EPOCHS}
    model, acc, _ = train_one(
        Xtr_embs, ytr, Xte_embs, yte, hidden=HIDDEN, lr=LR, epochs=EPOCHS
    )
    print(f"-> acc: {acc:.4f}")

    # save best model and embedding info
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": Xtr_embs.shape[1],
            "config": cfg,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    # Run grid search with top-level configuration variables
    train_model()
