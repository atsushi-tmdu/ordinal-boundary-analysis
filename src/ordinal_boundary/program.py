from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from .contracts import AnalysisSpec, require
from .data import AnalysisSample, Design
from .models import BinaryFit, BinaryGLM, FixedEffectPool, StandardizedContrast
from .workflow import BoundaryScan


class Functional(Protocol):
    def evaluate(self, fit: BinaryFit) -> dict: ...


@dataclass(frozen=True)
class Estimand:
    name: str
    functional: Functional

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "Empty estimand name")


@dataclass(frozen=True)
class FitRequest:
    name: str
    spec: AnalysisSpec
    estimands: tuple[Estimand, ...] = ()
    boundaries: tuple[float, ...] = ()
    include_reference_statistics: bool = True

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "Empty request name")
        require(len({e.name for e in self.estimands}) == len(self.estimands), "Duplicate estimand names")
        require(len(set(self.boundaries)) == len(self.boundaries), "Duplicate boundaries")


@dataclass(frozen=True)
class ExecutionGraph:
    requests: tuple[FitRequest, ...]
    estimator: BinaryGLM = field(default_factory=BinaryGLM)

    def __post_init__(self) -> None:
        require(bool(self.requests), "Empty execution graph")
        require(len({r.name for r in self.requests}) == len(self.requests), "Duplicate request names")

    def execute(self, frame) -> dict:
        records = {}
        for request in self.requests:
            sample = AnalysisSample.from_frame(frame, request.spec)
            design = Design.reference(sample)
            fit = self.estimator.fit(design)
            row = {
                "sample_size": fit.n,
                "event_count": sample.events,
                "cluster_count": sample.active_clusters,
                "estimands": {e.name: e.functional.evaluate(fit) for e in request.estimands},
                "comparisons": [],
            }
            if request.include_reference_statistics:
                row["reference"] = {"AIC": fit.aic, "BIC": fit.bic, "parameters": fit.k}
            if request.boundaries:
                row["comparisons"] = BoundaryScan(
                    request.spec, request.boundaries, self.estimator
                ).run(frame)
            records[request.name] = row
        return records


@dataclass(frozen=True)
class ContrastFactory:
    @staticmethod
    def difference(first: float, second: float) -> StandardizedContrast:
        return StandardizedContrast((first, second), (1.0, -1.0))

    @staticmethod
    def weighted(scores: tuple[float, ...], weights: tuple[float, ...]) -> StandardizedContrast:
        return StandardizedContrast(scores, weights)

    @staticmethod
    def local_excess(lower: float, left: float, right: float, upper: float) -> StandardizedContrast:
        require(np.isfinite([lower, left, right, upper]).all(), "Invalid ordered contrast")
        require(lower < left < right < upper, "Contrast scores are not strictly increasing")
        return StandardizedContrast((lower, left, right, upper), (-0.5, 1.5, -1.5, 0.5))


@dataclass(frozen=True)
class IndependentPool:
    def inverse_variance(self, estimates: tuple[float, ...], standard_errors: tuple[float, ...]) -> dict:
        return FixedEffectPool(estimates, standard_errors).evaluate()


@dataclass(frozen=True)
class LogOddsContrast:
    first: float
    second: float

    def evaluate(self, fit: BinaryFit) -> dict:
        from scipy import stats
        design = fit.design
        vector = np.mean(design.intervention(self.first) - design.intervention(self.second), axis=0)
        estimate = float(vector @ fit.coefficients)
        variance = float(vector @ fit.covariance @ vector)
        require(np.isfinite(variance) and variance > 0, "Invalid log-odds contrast variance")
        standard_error = float(np.sqrt(variance))
        critical = float(stats.norm.ppf(0.975))
        return {
            "first": self.first,
            "second": self.second,
            "log_odds": estimate,
            "standard_error": standard_error,
            "odds_ratio": float(np.exp(estimate)),
            "odds_ratio_low": float(np.exp(estimate - critical * standard_error)),
            "odds_ratio_high": float(np.exp(estimate + critical * standard_error)),
            "p_normal": float(2 * stats.norm.sf(abs(estimate / standard_error))),
        }
