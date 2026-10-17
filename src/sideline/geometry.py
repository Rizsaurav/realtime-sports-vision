"""Small numeric helpers shared by the live pipeline."""
import cv2
import numpy as np


def kmeans2(X, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), 2, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), 1)
        new = np.stack([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    return c


def kmeans2_init(X, c, iters=10):
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), 1)
        new = np.stack([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
        if np.allclose(new, c):
            break
        c = new
    return c


def torso_feature(frame, x, y, w, h):
    """CIELAB mean and std of the non-grass torso pixels (15-55% of the box height)."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    x1, x2 = max(0, x + w // 4), min(frame.shape[1], x + 3 * w // 4)
    y1, y2 = max(0, y + int(h * 0.15)), min(frame.shape[0], y + int(h * 0.55))
    crop = frame[y1:y2, x1:x2]
    if crop.size < 12:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    grass = (hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 60)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = lab[~grass] if (~grass).sum() >= 6 else lab
    return np.concatenate([keep.mean(0), keep.std(0)])


def project(H, pts):
    p = cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H)
    return p.reshape(-1, 2)


def foot_points(H, x, y, w, h):
    """Pitch positions (metres) of the bottom-left, bottom-middle and bottom-right box points."""
    return project(H, [(x, y + h), (x + w / 2, y + h), (x + w, y + h)]).reshape(-1).tolist()


def on_pitch(p, margin=5.0):
    return p is not None and abs(p[2]) <= 52.5 + margin and abs(p[3]) <= 34.0 + margin
