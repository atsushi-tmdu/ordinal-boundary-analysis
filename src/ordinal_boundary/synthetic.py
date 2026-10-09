from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class Simulation:
    size: int = 1800
    lower: int = 2
    upper: int = 9
    clusters: int = 60
    seed: int = 941

    def draw(self) -> pd.DataFrame:
        if self.size < 100 or self.clusters < 2 or self.lower >= self.upper:
            raise ValueError("Invalid simulation design")
        rng = np.random.default_rng(self.seed)
        values = rng.integers(self.lower, self.upper + 1, size=self.size)
        confounder = rng.normal(size=self.size)
        batch = rng.integers(0, self.clusters, size=self.size)
        group_shift = rng.normal(0, 0.15, size=self.clusters)
        logits = -0.1 - 0.13 * (values - 5) + 0.29 * confounder + 0.17 * (values <= 5) + group_shift[batch]
        outcomes = rng.binomial(1, expit(logits))
        return pd.DataFrame({"ordinal": values, "target": outcomes, "adjustment": confounder, "group": batch})
