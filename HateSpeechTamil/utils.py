import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def set_seed(seed=42):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics_for_task(preds, labels):
    """
    Computes accuracy, precision, recall, and macro F1 score for a single task.
    """
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
