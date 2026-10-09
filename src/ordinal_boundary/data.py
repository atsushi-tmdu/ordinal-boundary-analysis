from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
import numpy as np
import pandas as pd
import patsy
from .contracts import AnalysisSpec, require

@dataclass(frozen=True)
class AnalysisSample:
    spec: AnalysisSpec
    _frame: pd.DataFrame = field(repr=False)
    y: np.ndarray = field(repr=False)
    groups: np.ndarray = field(repr=False)
    fingerprint: str
    source_rows: int
    @classmethod
    def from_frame(cls, source: pd.DataFrame, spec: AnalysisSpec):
        d=source.copy(deep=True)
        require(d.columns.is_unique,"Duplicate input column labels")
        require(d.index.is_unique,"Duplicate input row index")
        if spec.id_column:
            require(spec.id_column in d, "Missing identifier column")
            ids=d[spec.id_column]
            require(ids.notna().all() and ids.astype(str).str.strip().ne('').all() and not ids.astype(str).duplicated().any(), "Invalid or duplicate participant keys")
        for rule in spec.input_rules:rule.validate(d)
        for t in spec.transforms:
            require(t.source in d,"Missing transform source")
            require(t.target not in source.columns, "Transform would overwrite existing column")
            require(pd.api.types.is_numeric_dtype(d[t.source]),"Transform source must be numeric")
            t.apply(d)
        need=[spec.exposure,spec.outcome,spec.cluster,*spec.covariates]
        require(set(need)<=set(d.columns),"Missing required analysis columns")
        numeric=[spec.exposure,spec.outcome,*spec.covariates]
        require(all(pd.api.types.is_numeric_dtype(d[c]) for c in numeric),"Analysis variables must be numeric")
        mask=d[spec.exposure].between(spec.lower,spec.upper) & d[need].notna().all(axis=1)
        d=d.loc[mask].copy()
        require(len(d)>0,"Empty complete-case sample")
        require(np.isfinite(d[numeric].to_numpy(float)).all(),"Nonfinite retained value; no silent post-hoc exclusion")
        require(set(d[spec.outcome].unique())=={0,1},"Outcome must contain binary events and nonevents")
        require(d[spec.cluster].astype(str).str.strip().ne('').all(),"Empty cluster label")
        if spec.required_scores:
            require(set(d[spec.exposure].unique())==set(spec.required_scores),"Score support differs from contract")
        groups,levels=pd.factorize(d[spec.cluster].astype(str),sort=True)
        require(len(levels)>1,"Cluster covariance requires at least two clusters")

        hashcols=need+([spec.id_column] if spec.id_column else [])
        h=sha256(spec.fingerprint.encode())
        h.update(pd.util.hash_pandas_object(d[hashcols],index=True).values.tobytes())
        h.update(mask.to_numpy(np.uint8).tobytes())
        y=d[spec.outcome].to_numpy(float); groups=np.asarray(groups,int)
        y.flags.writeable=False;groups.flags.writeable=False
        return cls(spec,d,y,groups,h.hexdigest(),len(source))
    @property
    def n(self): return len(self.y)
    @property
    def events(self): return int(self.y.sum())
    @property
    def active_clusters(self): return len(np.unique(self.groups))
    def frame(self) -> pd.DataFrame: return self._frame.copy(deep=True)
    def summary(self) -> dict:
        return {"n":self.n,"events":self.events,"active_G":self.active_clusters,
                "source_rows":self.source_rows,"sample_fingerprint":self.fingerprint}

@dataclass(frozen=True)
class Design:
    matrix: np.ndarray = field(repr=False)
    columns: tuple[str,...]
    info: object = field(repr=False)
    sample: AnalysisSample = field(repr=False)
    boundary: float | None = None
    @classmethod
    def reference(cls,sample: AnalysisSample):
        x=patsy.dmatrix(sample.spec.formula,sample.frame(),return_type="dataframe",NA_action="raise")
        require(len(x)==sample.n,"Design construction changed sample")
        a=np.asarray(x,float);a.flags.writeable=False
        return cls(a,tuple(x.columns),x.design_info,sample)
    def with_step(self,boundary:float):
        require(self.boundary is None,"Cannot append a second boundary step")
        require(np.isfinite(boundary) and self.sample.spec.lower<boundary<self.sample.spec.upper,"Boundary outside analysis support")
        step=(self.sample._frame[self.sample.spec.exposure].to_numpy(float)<=boundary).astype(float)
        require(len(np.unique(step))==2,"Boundary has an empty side")
        a=np.column_stack([self.matrix,step]);a.flags.writeable=False
        return Design(a,self.columns+("step_lower_score",),self.info,self.sample,float(boundary))
    def intervention(self,score:float) -> np.ndarray:
        require(np.isfinite(score) and self.sample.spec.lower<=score<=self.sample.spec.upper,"Intervention outside analysis support")
        d=self.sample.frame();d[self.sample.spec.exposure]=score
        for transform in self.sample.spec.transforms:
            transform.apply(d)
        x=np.asarray(patsy.build_design_matrices([self.info],d,NA_action="raise")[0],float)
        if self.boundary is not None:
            x=np.column_stack([x,np.full(len(x),float(score<=self.boundary))])
        require(x.shape==self.matrix.shape,"Intervention changed design shape")
        return x
