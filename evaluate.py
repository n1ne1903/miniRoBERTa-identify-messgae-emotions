import torch

def get_cls_acc(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()