from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import math
from typing import Iterable
import numpy as np
from scipy import stats
from .contracts import require

def display(value:float|str,digits:int)->str:
    q=Decimal(str(value)).quantize(Decimal(1).scaleb(-digits),rounding=ROUND_HALF_EVEN)
    if q==0:q=abs(q)
    return format(q,f'.{digits}f')

@dataclass(frozen=True)
class MetricRule:
    candidate:str
    reference:str
    scale:str='absolute'
    tolerance:float=0.0
    digits:int|None=None
    def compare(self,new,old)->dict:
        a,b=float(new),float(old)
        require(np.isfinite([a,b]).all(),'Nonfinite validation value')
        require(self.tolerance>=0,'Negative validation tolerance')
        if self.scale=='display':
            require(self.digits is not None,'Display precision is required')
            actual,expected=display(new,self.digits),display(old,self.digits)
            return {'passed':actual==expected,'actual':actual,'expected':expected,'scale':'display'}
        if self.scale=='log':
            require(a>0 and b>0,'Nonpositive logarithmic validation value')
            difference=math.log(a)-math.log(b)
        elif self.scale=='absolute':difference=a-b
        else:raise ValueError('Unknown validation scale')
        return {'passed':abs(difference)<=self.tolerance,'difference':difference,'tolerance':self.tolerance,'scale':self.scale}

@dataclass(frozen=True)
class ResultContract:
    keys:tuple[str,...]
    rules:tuple[MetricRule,...]
    def compare(self,candidate:Iterable[dict],reference:Iterable[dict])->dict:
        require(bool(self.keys) and bool(self.rules),"Validation requires keys and at least one metric rule")
        require(len(set(self.keys))==len(self.keys),"Duplicate validation key fields")
        candidate=list(candidate);reference=list(reference)
        def key(x):
            result=[]
            for k in self.keys:
                v=x[k]
                if k=='boundary':v=float(v)
                if k=='era':v=str(v).replace('–','-').replace('—','-')
                result.append(v)
            return tuple(result)
        c={key(x):x for x in candidate};r={key(x):x for x in reference}
        require(len(c)==len(candidate) and len(r)==len(reference),'Duplicate validation keys')
        require(set(c)==set(r) and bool(c),'Candidate/reference key sets differ or empty')
        rows=[]
        for k in sorted(c):
            checks={rule.candidate:rule.compare(c[k][rule.candidate],r[k][rule.reference]) for rule in self.rules}
            rows.append({'key':k,'checks':checks,'passed':all(q['passed'] for q in checks.values())})
        return {'all_passed':all(row['passed'] for row in rows),'comparisons':len(rows),
                'metric_cells':len(rows)*len(self.rules),'rows':rows}

class InformationCriterionAudit:
    @staticmethod
    def comparison(row:dict,tolerance:float=1e-8)->dict:
        n=float(row['n']);k0=float(row['reference_k']);k1=float(row['step_k'])
        require(n>0 and n.is_integer() and k0.is_integer() and k1.is_integer() and k1>k0,'Invalid information-criterion dimensions')
        ll0=float(row['reference_llf']);ll1=float(row['step_llf']);df=k1-k0;lr=2*(ll1-ll0)
        require(np.isfinite([ll0,ll1]).all() and lr>=-tolerance,'Invalid nested likelihood')
        expected={'delta_BIC':df*math.log(n)-lr,'delta_AIC':2*df-lr,'LR_chi2':lr,
                  'LR_p':float(stats.chi2.sf(max(0,lr),df))}
        delta={k:float(row[k])-v for k,v in expected.items()}
        return {'passed':all(np.isfinite(v) and abs(v)<=tolerance for v in delta.values()),'differences':delta}
    @staticmethod
    def fitted_model(row:dict,tolerance:float=1e-8)->dict:
        n,k=int(row['n']),int(row['k']);ll=float(row['llf'])
        b=np.asarray(row['coefficients'],float);v=np.asarray(row['cluster_covariance'],float)
        require(b.shape==(k,) and v.shape==(k,k) and n>k,'Invalid stored model dimensions')
        se=np.sqrt(np.diag(v));require(np.isfinite([ll]).all() and np.isfinite(b).all() and np.isfinite(v).all(),'Nonfinite stored model')
        errors={'AIC':float(row['AIC'])-(-2*ll+2*k),'BIC':float(row['BIC'])-(-2*ll+math.log(n)*k)}
        if 'standard_errors_cluster' in row:errors['SE_max']=float(np.max(abs(se-row['standard_errors_cluster'])))
        if 'coefficient_normal_Wald_p' in row:errors['P_max']=float(np.max(abs(2*stats.norm.sf(abs(b/se))-row['coefficient_normal_Wald_p'])))
        if 'coefficient_95CI' in row:errors['CI_max']=float(np.max(abs(np.column_stack([b-stats.norm.ppf(.975)*se,b+stats.norm.ppf(.975)*se])-row['coefficient_95CI'])))
        return {'passed':all(np.isfinite(v) and abs(v)<=tolerance for v in errors.values()),'differences':errors}
