from __future__ import annotations
from ..contracts import AnalysisSpec, AffinePower, NaturalSpline, Polynomial, Categorical, LinearHinge, NumericRange, require
from ..workflow import BoundaryScan
from ..models import BinaryGLM
from ..contracts import FitOptions

OUTCOMES=('severe_head','death','head_surgery','intubation')
CENTERS={'2008-2014':2011,'2015-2018':2017,'2019-2023':2021}
COVARIATES=('agec','agec2','male','blunt','alcohol_yes','alcohol_unknown','yearc','sbpc','hrc','rrc')

def s18_spec(era:str,outcome:str)->AnalysisSpec:
    require(era in CENTERS and outcome in OUTCOMES,'Unknown SJTREM era/outcome')
    transforms=(AffinePower('age','agec',65,10),AffinePower('age','agec2',65,10,2),
       AffinePower('year','yearc',CENTERS[era]),AffinePower('sbp','sbpc',130,10),
       AffinePower('hr','hrc',80,10),AffinePower('rr','rrc',18,5))
    return AnalysisSpec('gcs',outcome,'hospital',COVARIATES,NaturalSpline(5,12,(7,9,11)),5,12,
          transforms,'sid',tuple(range(5,13)),15,era,(NumericRange('year',*map(int,era.split('-')),integer=True),))

def s18_scan(era:str,outcome:str)->BoundaryScan:
    return BoundaryScan(s18_spec(era,outcome),tuple(i+.5 for i in range(5,12)),BinaryGLM(FitOptions()))
