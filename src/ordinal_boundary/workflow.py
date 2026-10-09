from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Callable
import importlib.metadata
import hashlib
import numpy as np
from .contracts import AnalysisSpec, FitOptions, require, digest
from .data import AnalysisSample, Design
from .models import BinaryGLM, NestedComparison
from .storage import CheckpointStore, atomic_json

@dataclass(frozen=True)
class BoundaryScan:
    spec:AnalysisSpec
    boundaries:tuple[float,...]
    engine:BinaryGLM=BinaryGLM()
    def __post_init__(self):
        require(len(self.boundaries)>0 and len(set(self.boundaries))==len(self.boundaries),"Empty/duplicate boundary grid")
        require(np.isfinite(self.boundaries).all() and all(self.spec.lower<b<self.spec.upper for b in self.boundaries),"Boundary grid outside support")
    def run(self,frame,checkpoint:Path|None=None,progress:Callable[[dict],None]|None=None,max_seconds:float|None=None):
        require(max_seconds is None or max_seconds>0,"Invalid time budget")
        sample=AnalysisSample.from_frame(frame,self.spec);design=Design.reference(sample)
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','statsmodels','patsy','threadpoolctl']}

        code=hashlib.sha256()
        for p in sorted(Path(__file__).parent.rglob('*.py')):code.update(p.relative_to(Path(__file__).parent).as_posix().encode());code.update(p.read_bytes())
        identity={'sample':sample.fingerprint,'spec':self.spec.spec(),'boundaries':list(self.boundaries),
                  'engine':asdict(self.engine.options),'versions':versions,'source_sha256':code.hexdigest()}
        identity=__import__('json').loads(__import__('json').dumps(identity))
        start=monotonic();rows=[];reference=None
        def perform(store=None):
            nonlocal reference
            for i,b in enumerate(self.boundaries):
                key=digest({'identity':identity,'boundary':b})
                cached=store.read(key) if store else None
                if cached is not None:
                    require(cached['sample_fingerprint']==sample.fingerprint and cached['boundary']==b,"Wrong checkpoint sample/boundary")
                    row=cached;state='REUSED'
                else:
                    require(max_seconds is None or monotonic()-start<max_seconds,"Time budget reached between fits; completed boundaries preserved")
                    if reference is None:reference=self.engine.fit(design)
                    row=NestedComparison(reference,self.engine.fit(design.with_step(b))).record();state='FITTED'
                    if store:store.write(key,row)
                rows.append(row)
                if progress:progress({'completed':i+1,'total':len(self.boundaries),'state':state,'elapsed_seconds':monotonic()-start})
            return rows
        if checkpoint:
            with CheckpointStore(checkpoint).transaction() as store:
                store.bind(identity);result=perform(store)
                atomic_json(store.path/'completion.json',{'status':'ALL_BOUNDARIES_COMPLETE','comparisons':len(result),'identity_sha256':digest(identity)})
                return result
        return perform()
