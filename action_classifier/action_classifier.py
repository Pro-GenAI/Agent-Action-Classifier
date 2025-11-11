"""
Action Classifier Module
Loads the trained neural network model to classify agent actions as harmful or safe.
"""

import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn

# Configuration
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FORCE_CPU = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "emb_nn_model.pt"


class SimpleMLP(nn.Module):
    """Simple MLP for binary classification."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class ActionClassifier:
    """Classifier for AI agent actions using embeddings and MLP."""

    def __init__(self, load_model: bool = True):
        """If load_model is False, heavy dependencies (embedding/model) are not created.
        Tests can instantiate with load_model=False and then set attributes manually.
        """
        self.embed_model = None
        self.model = None
        self.in_dim = None
        if load_model:
            # create heavy components only on demand
            self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))
            self.load_model()

    def load_model(self):
        """Load the trained model from file."""
        """Load the trained model from file."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        self.in_dim = checkpoint["in_dim"]
        config = checkpoint["config"]
        hidden = config["hidden"]

        self.model = SimpleMLP(in_dim=self.in_dim, hidden=hidden)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

    def flatten_action_to_text(self, entry):
        """Flatten action metadata to text, same as in training."""
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
            except (ValueError, AttributeError):
                parts.append(server_url)
        parts.append(action_meta.get("require_approval", ""))
        params = action_meta.get("parameters") or {}
        if isinstance(params, dict):
            parts.extend(sorted(list(params.keys())))
        else:
            parts.append(str(type(params)))
        return " ".join([str(p) for p in parts if p])

    def predict(self, action_dict):
        """
        Predict if an action is harmful.
        Returns: (is_harmful: bool, confidence: float)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        text = self.flatten_action_to_text(action_dict)
        if self.embed_model is None:
            raise RuntimeError("Embedding model not initialized.")
        embedding = self.embed_model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        embedding = np.array(embedding)

        if self.model is None:
            raise RuntimeError(
                "Classifier model not loaded. Create ActionClassifier(load_model=True) "
                "or set model manually for tests."
            )

        with torch.inference_mode():
            logits = self.model(torch.tensor(embedding, dtype=torch.float32).to(DEVICE))
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0, int(pred_class)].item()

        is_harmful = pred_class == 1
        return is_harmful, confidence


# Lazy global classifier accessor
_GLOBAL_CLASSIFIER = None


def get_classifier():
    """Get the global classifier instance."""
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None:
        _GLOBAL_CLASSIFIER = ActionClassifier()
    return _GLOBAL_CLASSIFIER


def is_action_harmful(action_dict):
    """Convenience function to check if an action is harmful."""
    clf = get_classifier()
    return clf.predict(action_dict)
