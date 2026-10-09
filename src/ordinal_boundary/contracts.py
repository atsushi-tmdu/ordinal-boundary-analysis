from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Protocol
import numpy as np

class ContractError(RuntimeError):
    pass

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)

def identifier(name: str) -> str:
    require(bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)), "Invalid column identifier")
    return name

def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class AffinePower:
    source: str
    target: str
    center: float = 0.0
    scale: float = 1.0
    power: int = 1
    def __post_init__(self) -> None:
        identifier(self.source); identifier(self.target)
        require(np.isfinite([self.center, self.scale]).all() and self.scale != 0, "Invalid transform scale/center")
        require(isinstance(self.power, int) and self.power >= 1, "Transform power must be positive integer")
    def apply(self, frame):
        frame[self.target] = ((frame[self.source] - self.center) / self.scale) ** self.power

@dataclass(frozen=True)
class NumericRange:
    column: str
    lower: float
    upper: float
    integer: bool = False
    allow_missing: bool = False
    def __post_init__(self):
        identifier(self.column)
        require(np.isfinite([self.lower,self.upper]).all() and self.lower<=self.upper,"Invalid input rule")
    def validate(self,frame):
        require(self.column in frame,"Missing constrained input column")
        values=frame[self.column]
        require(self.allow_missing or values.notna().all(),"Missing constrained input value")
        values=values.dropna().to_numpy(float)
        require(np.isfinite(values).all() and np.all((values>=self.lower)&(values<=self.upper)),"Input range constraint failed")
        require(not self.integer or np.all(values%1==0),"Input integer constraint failed")

class Basis(Protocol):
    def term(self, exposure: str) -> str: ...
    def spec(self) -> dict: ...

@dataclass(frozen=True)
class NaturalSpline:
    lower: float
    upper: float
    knots: tuple[float, ...]
    def __post_init__(self) -> None:
        require(self.lower < self.upper and np.isfinite([self.lower, self.upper, *self.knots]).all(), "Invalid spline bounds")
        require(tuple(sorted(set(self.knots))) == self.knots, "Spline knots must be unique and ordered")
        require(all(self.lower < k < self.upper for k in self.knots), "Spline knot outside bounds")
    def term(self, exposure: str) -> str:
        return f'cr({identifier(exposure)}, knots={self.knots!r}, lower_bound={self.lower!r}, upper_bound={self.upper!r}, constraints="center")'
    def spec(self) -> dict:
        return {"kind": "natural_spline", **asdict(self)}

@dataclass(frozen=True)
class Polynomial:
    degree: int = 4
    center: float = 0.0
    scale: float = 1.0
    def __post_init__(self) -> None:
        require(isinstance(self.degree, int) and self.degree >= 1, "Invalid polynomial degree")
        require(np.isfinite([self.center,self.scale]).all() and self.scale != 0, "Invalid polynomial center/scale")
    def term(self, exposure: str) -> str:
        x = f'(({identifier(exposure)} - {self.center!r}) / {self.scale!r})'
        return " + ".join(f"I({x} ** {k})" for k in range(1, self.degree+1))
    def spec(self) -> dict:
        return {"kind":"polynomial", **asdict(self)}

@dataclass(frozen=True)
class Categorical:
    def term(self, exposure: str) -> str:
        return f'C({identifier(exposure)})'
    def spec(self) -> dict:
        return {"kind":"categorical"}

@dataclass(frozen=True)
class LinearHinge:
    boundary: float
    def __post_init__(self) -> None:
        require(np.isfinite(self.boundary), "Invalid hinge boundary")
    def term(self, exposure: str) -> str:
        x=identifier(exposure)
        return f'I({x}-{self.boundary!r}) + I(np.maximum({x}-{self.boundary!r}, 0))'
    def spec(self) -> dict:
        return {"kind":"linear_hinge", **asdict(self)}

@dataclass(frozen=True)
class AnalysisSpec:
    exposure: str
    outcome: str
    cluster: str
    covariates: tuple[str, ...]
    basis: Basis
    lower: float
    upper: float
    transforms: tuple[AffinePower, ...] = ()
    id_column: str | None = None
    required_scores: tuple[float, ...] = ()
    expected_rank: int | None = None
    label: str = "analysis"
    input_rules: tuple[NumericRange, ...] = ()
    def __post_init__(self) -> None:
        for s in (self.exposure,self.outcome,self.cluster,*self.covariates): identifier(s)
        if self.id_column: identifier(self.id_column)
        require(len(set(self.covariates)) == len(self.covariates), "Duplicate covariates")
        require(not set(self.covariates) & {self.exposure,self.outcome,self.cluster,self.id_column}, "Exposure/outcome/cluster/id cannot also be a covariate")
        require(np.isfinite([self.lower,self.upper]).all() and self.lower < self.upper, "Invalid analysis range")
        targets=[t.target for t in self.transforms]
        require(len(set(targets))==len(targets), "Duplicate transform targets")
        require(all(t.source==self.exposure or t.source not in targets for t in self.transforms), "Chained transforms are unsupported")
        require(not set(targets) & {self.exposure,self.outcome,self.cluster,self.id_column}, "Transform would overwrite protected role")
        require(len(set(self.required_scores))==len(self.required_scores), "Duplicate required scores")
        require(all(self.lower<=s<=self.upper for s in self.required_scores), "Required score outside range")
        require(self.expected_rank is None or self.expected_rank>0, "Invalid expected rank")
    @property
    def formula(self) -> str:
        return "1 + " + self.basis.term(self.exposure) + (" + " + " + ".join(self.covariates) if self.covariates else "")
    def spec(self) -> dict:
        d=asdict(self); d["basis"]=self.basis.spec()
        return d
    @property
    def fingerprint(self) -> str:
        return digest(self.spec())

@dataclass(frozen=True)
class FitOptions:
    maxiter: int = 100
    tolerance: float = 1e-8
    threads: int = 1
    def __post_init__(self) -> None:
        require(self.maxiter>0 and self.threads>0 and 0<self.tolerance<1, "Invalid fit options")


def basis_from_dict(d: dict) -> Basis:
    a=dict(d); k=a.pop("kind")
    classes={"natural_spline":NaturalSpline,"polynomial":Polynomial,"categorical":Categorical,"linear_hinge":LinearHinge}
    require(k in classes,"Unknown basis kind")
    if "knots" in a: a["knots"]=tuple(a["knots"])
    return classes[k](**a)

def spec_from_dict(d: dict) -> AnalysisSpec:
    a=dict(d);a["basis"]=basis_from_dict(a["basis"])
    a["covariates"]=tuple(a.get("covariates",()))
    a["required_scores"]=tuple(a.get("required_scores",()))
    a["transforms"]=tuple(AffinePower(**t) for t in a.get("transforms",()))
    a["input_rules"]=tuple(NumericRange(**t) for t in a.get("input_rules",()))
    return AnalysisSpec(**a)
