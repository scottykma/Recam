"""Pure-signal frame quality metrics (no ML model).

Currently exposes only colorfulness; add metrics here when needed so training
and data preparation share the same implementation.
"""
import numpy as np


def colorfulness(frame: np.ndarray) -> float:
    """Hasler-Süsstrunk (2003) color richness score for a uint8 RGB frame (H,W,3).

    rg = R-G, yb = 0.5(R+G)-B; result = std(rg,yb) + 0.3*mean(rg,yb).
    Low values (~<18) indicate gray/dull frames; near-zero means monochrome.
    """
    R = frame[:, :, 0].astype(np.float32)
    G = frame[:, :, 1].astype(np.float32)
    B = frame[:, :, 2].astype(np.float32)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)
