import torch


def compute_confusion(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).bool()
    t = targets.bool()
    tp = int((preds & t).sum().item())
    fp = int((preds & ~t).sum().item())
    fn = int((~preds & t).sum().item())
    tn = int((~preds & ~t).sum().item())
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)
    return {
        "IoU": round(iou, 4),
        "Dice": round(dice, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 6),
    }
