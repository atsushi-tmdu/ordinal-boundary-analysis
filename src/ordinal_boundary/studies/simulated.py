from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import expit
from .primary import OUTCOMES

def simulated_eligible(era:str,seed:int=2049,per_score:int=90)->pd.DataFrame:
    if era not in ('2008-2014','2019-2023') or per_score<40:
        raise ValueError('Invalid simulation settings')
    rng=np.random.default_rng(seed)
    gcs=np.repeat(np.arange(3,16),per_score)
    n=len(gcs)
    age=rng.uniform(18,85,n)
    sbp=rng.uniform(111,165,n)
    hr=rng.uniform(55,110,n)
    rr=rng.uniform(11,26,n)
    male=rng.binomial(1,.6,n)
    blunt=rng.binomial(1,.75,n)
    alcohol=rng.choice([0,1,2],n,p=[.75,.15,.10])
    start,stop=map(int,era.split('-'))
    year=rng.integers(start,stop+1,n)
    hospital=rng.integers(0,55,n)
    group_effect=rng.normal(0,.12,55)
    linear=-.55-.14*(gcs-8)+.015*(age-50)+.14*male+.18*blunt+group_effect[hospital]
    frame=pd.DataFrame({'sid':[f'SYNTHETIC_{i}' for i in range(n)],'hospital':[f'FAKE_{v}' for v in hospital],'gcs':gcs,'age':age,'year':year,'male':male,'blunt':blunt,'alcohol_yes':(alcohol==1).astype(int),'alcohol_unknown':(alcohol==2).astype(int),'sbp':sbp,'hr':hr,'rr':rr})
    for j,outcome in enumerate(OUTCOMES):
        frame[outcome]=rng.binomial(1,expit(linear+(j-1.5)*.13))
    return frame
