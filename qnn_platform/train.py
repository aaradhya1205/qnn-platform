"""Mini-batch training loop for the QNN, with configurable optimizer/lr and
per-epoch train/validation metric tracking."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_metric: list = field(default_factory=list)
    val_metric: list = field(default_factory=list)


def _metric(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if task == "classification":
        return float((y_true == y_pred).mean())
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)  # R^2


def train_model(model, X_train, y_train, X_val, y_val, task: str, n_classes: int = 2,
                 epochs: int = 20, batch_size: int = 16, optimizer_name: str = "adam",
                 lr: float = 0.05, progress_cb=None) -> TrainingHistory:
    """Train `model` (a torch.nn.Module, e.g. QNNModel) with mini-batches.

    progress_cb(epoch, total_epochs, history) is called after every epoch so a UI
    (e.g. Streamlit) can render live progress.
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    if task == "classification":
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        loss_fn = torch.nn.BCEWithLogitsLoss() if n_classes <= 2 else torch.nn.CrossEntropyLoss()
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        loss_fn = torch.nn.MSELoss()

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    opt_cls = OPTIMIZERS.get(optimizer_name, torch.optim.Adam)
    optimizer = opt_cls(model.parameters(), lr=lr)

    history = TrainingHistory()

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            if task == "classification" and n_classes <= 2:
                loss = loss_fn(out.squeeze(-1), yb.float())
            else:
                loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_out = model(X_train_t)
            val_out = model(X_val_t)
            if task == "classification" and n_classes <= 2:
                train_loss = loss_fn(train_out.squeeze(-1), y_train_t.float()).item()
                val_loss = loss_fn(val_out.squeeze(-1), y_val_t.float()).item()
                train_pred = (torch.sigmoid(train_out.squeeze(-1)) > 0.5).long().numpy()
                val_pred = (torch.sigmoid(val_out.squeeze(-1)) > 0.5).long().numpy()
            elif task == "classification":
                train_loss = loss_fn(train_out, y_train_t).item()
                val_loss = loss_fn(val_out, y_val_t).item()
                train_pred = train_out.argmax(dim=1).numpy()
                val_pred = val_out.argmax(dim=1).numpy()
            else:
                train_loss = loss_fn(train_out, y_train_t).item()
                val_loss = loss_fn(val_out, y_val_t).item()
                train_pred = train_out.numpy().ravel()
                val_pred = val_out.numpy().ravel()

        history.train_loss.append(float(np.mean(batch_losses)))
        history.val_loss.append(val_loss)
        history.train_metric.append(_metric(task, y_train, train_pred))
        history.val_metric.append(_metric(task, y_val, val_pred))

        if progress_cb is not None:
            progress_cb(epoch + 1, epochs, history)

    return history


def evaluate(model, X, y, task: str, n_classes: int = 2):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        out = model(X_t)
        if task == "classification" and n_classes <= 2:
            pred = (torch.sigmoid(out.squeeze(-1)) > 0.5).long().numpy()
        elif task == "classification":
            pred = out.argmax(dim=1).numpy()
        else:
            pred = out.numpy().ravel()
    return pred, _metric(task, y, pred)
