"""A deliberately tiny 1D-CNN for classifying trend-curve shape."""
import numpy as np
import torch
import torch.nn as nn

SEED = 0
EPOCHS = 300
LR = 1e-3
DROPOUT = 0.2


class TrendCNN(nn.Module):
    """1 input channel -> two small conv layers -> global avg pool -> logit."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # global average pool over time
        )
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        # x: (batch, length) -> (batch, 1, length)
        x = x.unsqueeze(1)
        z = self.net(x).squeeze(-1)    # (batch, 16)
        return self.head(self.drop(z)).squeeze(-1)  # (batch,)


def train_fold(X_train, y_train):
    """Train a TrendCNN on (n, length) curves. Returns the trained model."""
    torch.manual_seed(SEED)
    model = TrendCNN()
    Xt = torch.tensor(np.asarray(X_train), dtype=torch.float32)
    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32)

    # class weighting for the positive class (handles 20-vs-30 imbalance)
    n_pos = max(float(yt.sum()), 1.0)
    n_neg = max(float((yt == 0).sum()), 1.0)
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    return model


def predict(model, X):
    """Return probabilities for (n, length) curves."""
    model.eval()
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    with torch.no_grad():
        return torch.sigmoid(model(Xt)).numpy()
