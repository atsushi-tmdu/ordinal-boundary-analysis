from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from ..contracts import AnalysisSpec, Categorical, LinearHinge, Polynomial, require, spec_from_dict
from ..data import AnalysisSample, Design
from ..models import BinaryGLM, StandardizedContrast
from ..program import LogOddsContrast
from ..workflow import BoundaryScan

TASKS = ('q1','q2','q3_local','q3_full','s18')
BOUNDARIES = {'q3_local':(8.5,), 'q3_full':tuple(i+0.5 for i in range(3,15)), 's18':tuple(i+0.5 for i in range(5,12))}
OUTCOMES = ('severe_head','death','head_surgery','intubation')

@dataclass(frozen=True)
class StudyContract:
    base: AnalysisSpec
    outcomes: tuple[str,...]
    boundaries: tuple[float,...]
    @classmethod
    def from_file(cls, path:Path):
        content=json.loads(Path(path).read_text(encoding='utf-8'))
        outcomes=tuple(content['outcomes'])
        require(outcomes==OUTCOMES,'Unexpected outcome contract')
        return cls(spec_from_dict(content['analysis']),outcomes,tuple(content['boundaries']))
    def for_task(self, task:str, outcome:str) -> AnalysisSpec:
        require(task in TASKS and outcome in self.outcomes,'Unknown task or outcome')
        settings={
            'q1':(Categorical(),8,9,(8,9),12),
            'q2':(Categorical(),3,15,tuple(range(3,16)),23),
            'q3_local':(LinearHinge(8.5),6,10,tuple(range(6,11)),13),
            'q3_full':(Polynomial(4,9,3),3,15,tuple(range(3,16)),15),
            's18':(self.base.basis,5,12,tuple(range(5,13)),15),
        }
        basis,lower,upper,scores,rank=settings[task]
        return replace(self.base,outcome=outcome,basis=basis,lower=lower,upper=upper,required_scores=scores,expected_rank=rank,label=f'{self.base.label}:{task}:{outcome}')

@dataclass(frozen=True)
class StudyRunner:
    contract: StudyContract
    estimator: BinaryGLM=BinaryGLM()
    def execute(self, frame:pd.DataFrame,task:str,outcome:str) -> dict:
        spec=self.contract.for_task(task,outcome)
        if task in BOUNDARIES:
            boundaries=BOUNDARIES[task]
            if task=='s18':require(boundaries==self.contract.boundaries,'S18 boundary grid differs from contract')
            rows=BoundaryScan(spec,boundaries,self.estimator).run(frame)
            return {'task':task,'outcome':outcome,'era':self.contract.base.label,'n_comparisons':len(rows),'results':rows}
        sample=AnalysisSample.from_frame(frame,spec)
        design=Design.reference(sample)
        fit=self.estimator.fit(design)
        if task=='q1':
            risks=StandardizedContrast.pair(8,9).evaluate(fit)
            odds=LogOddsContrast(8,9).evaluate(fit)
            return {'task':task,'outcome':outcome,'era':self.contract.base.label,'n':sample.n,'n8':int((sample._frame[spec.exposure]==8).sum()),'n9':int((sample._frame[spec.exposure]==9).sum()),'risk_difference':risks,'odds_ratio':odds}
        local=StandardizedContrast.local_excess((7,8,9,10)).evaluate(fit)
        return {'task':task,'outcome':outcome,'era':self.contract.base.label,'n':sample.n,'local_excess':local}

def combine_q1(first:dict,second:dict)->dict:
    from ..models import FixedEffectPool
    require(first['task']==second['task']=='q1','Only Q1 results can be combined')
    require(first['era']!=second['era'] and first['outcome']==second['outcome'],'Two distinct eras with identical outcome required')
    r1,r2=first['risk_difference'],second['risk_difference']
    o1,o2=first['odds_ratio'],second['odds_ratio']
    rd=FixedEffectPool((r1['estimate'],r2['estimate']),(r1['SE'],r2['SE'])).evaluate()
    lor=FixedEffectPool((o1['log_odds'],o2['log_odds']),(o1['standard_error'],o2['standard_error'])).evaluate()
    z=(o1['log_odds']-o2['log_odds'])/math.sqrt(o1['standard_error']**2+o2['standard_error']**2)
    return {'outcome':first['outcome'],'eras':(first['era'],second['era']),'risk_difference':rd,'pooled_odds_ratio':math.exp(lor['estimate']),'pooled_odds_ratio_low':math.exp(lor['CI_low']),'pooled_odds_ratio_high':math.exp(lor['CI_high']),'era_difference_p_normal':float(2*stats.norm.sf(abs(z))),'scale':'independent-era fixed-effect summary'}
