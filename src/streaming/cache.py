"""Frame feature cache (YOLOV-style) for the efficient-ViT streaming path.

Idea: the backbone is the expensive part. On keyframes (every k frames)
run the full backbone and cache its feature maps. On intermediate frames,
reuse cached features with a lightweight update instead of a full forward
pass. Detector head + tracker run every frame on (cached or fresh) features.

This is the experiment no public repo has published for an efficient-ViT
backbone in a tracking pipeline. The accuracy cost of caching vs the FPS
gain is one of the headline curves in the tradeoff study.

Reference: YOLOV (YuHengsss/YOLOV) for the keyframe/propagation design.
Week 2 wires this to EfficientViTDetector.backbone_features().
"""


class FrameFeatureCache:
    def __init__(self, keyframe_interval=3, ema_decay=0.0):
        """
        ema_decay: 0.0 = hard replace on keyframes; >0 blends cached features
                   (ablation for the report: does smoothing help MOTA?).
        """
        self.k = keyframe_interval
        self.ema_decay = ema_decay
        self._cache = None
        self._t = 0
        self.keyframes_used = 0

    def is_keyframe(self):
        return self._t % self.k == 0

    def update(self, features):
        """Store backbone features from a keyframe.

        features: tensor from EfficientViTDetector.backbone_features().
        None-safe until Week 2 wires real features (falls back gracefully).
        """
        if features is None:
            self.keyframes_used += 1
            return
        if self._cache is None or self.ema_decay == 0.0:
            self._cache = features
        else:
            self._cache = (self.ema_decay * self._cache
                           + (1 - self.ema_decay) * features)
        self.keyframes_used += 1

    def get(self):
        return self._cache

    def step(self):
        self._t += 1

    def reset(self):
        self._cache, self._t, self.keyframes_used = None, 0, 0

    def stats(self):
        total = self._t
        return {
            "frames": total,
            "keyframes": self.keyframes_used,
            "cache_hit_rate": 1 - self.keyframes_used / max(total, 1),
        }
