"""Turn decision logits into the two outputs of the plug-in layer: a gated image score and a window-weighted anomaly map."""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def gate_score(score, logit, exponent=0.5):
    """Image-level output S = score * sigmoid(logit) ** exponent.  `logit` is the mean of the reference-free and the reference-conditioned logit."""
    return score * sigmoid(logit) ** exponent


def window_weights(boxes, logits, shape, gamma=0.25):
    """Weight map W(x) = sigmoid(window logit) ** gamma; overlapping windows are averaged.  `boxes` are normalised (x0, y0, x1, y1), `shape` = (H, W) of the anomaly map."""
    H, W = shape
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    w = sigmoid(logits) ** gamma
    for (x0, y0, x1, y1), wi in zip(boxes, w):
        ys, ye, xs, xe = int(y0 * H), int(round(y1 * H)), int(x0 * W), int(round(x1 * W))
        num[ys:ye, xs:xe] += wi
        den[ys:ye, xs:xe] += 1
    return num / np.maximum(den, 1)


def apply_windows(anomaly_map, boxes, logits, gamma=0.25):
    """Pixel-level output M' = M * W."""
    return anomaly_map * window_weights(boxes, logits, anomaly_map.shape[-2:], gamma)
