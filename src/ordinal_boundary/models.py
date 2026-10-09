from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math
import warnings
import numpy as np
from scipy import stats
from scipy.special import expit
import statsmodels.api as sm
from threadpoolctl import threadpool_limits
from .contracts import FitOptions, require
from .data import Design

@dataclass(frozen=True)
class BinaryFit:
    design: Design = field(repr=False)
    coefficients: np.ndarray = field(repr=False)
    covariance: np.ndarray = field(repr=False)
    llf: float
    diagnostics: dict
    @property
    def n(self): return self.design.sample.n
    @property
    def k(self): return self.design.matrix.shape[1]
    @property
    def aic(self): return -2*self.llf+2*self.k
    @property
    def bic(self): return -2*self.llf+math.log(self.n)*self.k
    def record(self) -> dict:
        return {**self.design.sample.summary(),"k":self.k,"llf":self.llf,"AIC":self.aic,"BIC":self.bic,
                "columns":list(self.design.columns),"coefficients":self.coefficients.tolist(),
                "cluster_covariance":self.covariance.tolist(),"diagnostics":self.diagnostics}

@dataclass(frozen=True)
class BinaryGLM:
    options: FitOptions = FitOptions()
    def fit(self,design:Design) -> BinaryFit:
        x=design.matrix;y=design.sample.y;g=design.sample.groups;n,k=x.shape
        require(np.isfinite(x).all(),"Nonfinite design")
        require(np.linalg.matrix_rank(x)==k,"Rank-deficient design; no pseudo-inverse or regularization allowed")
        er=design.sample.spec.expected_rank
        if er is not None: require(k==er+(design.boundary is not None),"Design rank differs from contract")
        require(n>k,"Residual degrees of freedom not positive")
        with threadpool_limits(limits=self.options.threads),warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            fit=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=self.options.maxiter,tol=self.options.tolerance,
                method="IRLS",cov_type="cluster",cov_kwds={"groups":g,"use_correction":True},use_t=False)
        require(not ws,"Fit warning; review private data without publishing warning values")
        require(bool(fit.converged),"IRLS did not converge")
        b=np.asarray(fit.params,float);v=np.asarray(fit.cov_params(),float);p=np.asarray(fit.fittedvalues,float)
        require(all(np.isfinite(a).all() for a in [b,v,p]),"Nonfinite fit result")
        require(np.all((p>0)&(p<1)),"Boundary fitted probability; review separation")
        eta=x@b;ll=float(fit.llf);ld=float(np.sum(y*eta-np.logaddexp(0,eta)))
        require(abs(ll-ld)<=1e-8,"Bernoulli likelihood identity failed")
        fisher=x.T@((p*(1-p))[:,None]*x);bread=np.linalg.inv(fisher)
        G=design.sample.active_clusters;scoreg=np.zeros((G,k))
        for j in range(k):scoreg[:,j]=np.bincount(g,weights=x[:,j]*(y-p),minlength=G)
        correction=G/(G-1)*(n-1)/(n-k)
        cr1=bread@(scoreg.T@scoreg)@bread*correction
        require(np.allclose(v,cr1,rtol=1e-8,atol=1e-10),"Manual CR1 verification failed")
        require(np.allclose(v,v.T,atol=1e-10),"Asymmetric covariance")
        require(np.linalg.eigvalsh(v).min()>=-1e-9*max(1,float(np.linalg.norm(v,2))),"Covariance not positive semidefinite")
        b.flags.writeable=False;v.flags.writeable=False
        return BinaryFit(design,b,v,ll,{"converged":True,"iterations":int(fit.fit_history["iteration"]),
             "CR1_multiplier":correction,"CR1_max_abs_difference":float(np.max(abs(v-cr1))),
             "llf_direct_error":ll-ld,"ordinary_score_max_abs":float(np.max(abs(x.T@(y-p)))),
             "condition_number":float(np.linalg.cond(x)),"fit_options":asdict(self.options),
             "covariance":"cluster CR1; normal Wald; use_t=False","penalty":None,
             "separation_check_scope":"warnings, convergence and boundary probabilities; not a formal nonexistence proof"})

@dataclass(frozen=True)
class NestedComparison:
    reference: BinaryFit = field(repr=False)
    alternative: BinaryFit = field(repr=False)
    def __post_init__(self):
        r,a=self.reference,self.alternative
        require(r.design.sample.fingerprint==a.design.sample.fingerprint,"Nested comparison sample mismatch")
        require(a.k==r.k+1 and a.design.boundary is not None,"Expected one added boundary parameter")
        require(np.array_equal(r.design.matrix,a.design.matrix[:,:r.k]),"Models are not nested in an identical reference design")
        require(a.llf>=r.llf-1e-8,"Nested alternative has lower likelihood")
    def record(self) -> dict:
        r,a=self.reference,self.alternative;lr=2*(a.llf-r.llf)
        b=float(a.coefficients[-1]);se=float(np.sqrt(a.covariance[-1,-1]))
        require(np.isfinite(se) and se>0,"Invalid boundary standard error")
        z=stats.norm.ppf(.975)
        row={"label":r.design.sample.spec.label,"outcome":r.design.sample.spec.outcome,"boundary":a.design.boundary,
          **r.design.sample.summary(),"reference_k":r.k,"step_k":a.k,"reference_llf":r.llf,"step_llf":a.llf,
          "reference_AIC":r.aic,"step_AIC":a.aic,"reference_BIC":r.bic,"step_BIC":a.bic,
          "delta_BIC":a.bic-r.bic,"delta_AIC":a.aic-r.aic,"LR_chi2":lr,"LR_df":1,
          "LR_p":float(stats.chi2.sf(max(0,lr),1)),"step_log_OR":b,"step_SE_cluster":se,
          "step_OR_cluster":math.exp(b),"step_OR_low":math.exp(b-z*se),"step_OR_high":math.exp(b+z*se),
          "step_p_cluster":float(2*stats.norm.sf(abs(b/se)))}
        require(abs(row['delta_BIC']-(math.log(a.n)-lr))<1e-8,"BIC/LR identity failed")
        require(abs(row['delta_AIC']-(2-lr))<1e-8,"AIC/LR identity failed")
        return row

@dataclass(frozen=True)
class StandardizedContrast:
    scores: tuple[float,...]
    weights: tuple[float,...]
    def __post_init__(self):
        require(len(self.scores)==len(self.weights)>0,"Invalid contrast dimensions")
        require(len(set(self.scores))==len(self.scores),"Duplicate intervention score")
        require(np.isfinite([*self.scores,*self.weights]).all(),"Nonfinite contrast")
    def evaluate(self,fit:BinaryFit) -> dict:
        probabilities=[];gradients=[]
        for s in self.scores:
            x=fit.design.intervention(s);p=expit(x@fit.coefficients)
            probabilities.append(float(p.mean()));gradients.append(np.mean((p*(1-p))[:,None]*x,axis=0))
        g=np.asarray(gradients);joint=g@fit.covariance@g.T;w=np.asarray(self.weights)
        estimate=float(w@probabilities);variance=float(w@joint@w)
        require(variance>=-1e-12,"Negative contrast variance")
        se=math.sqrt(max(0,variance));require(se>0,"Degenerate contrast standard error")
        z=stats.norm.ppf(.975)
        return {"scores":self.scores,"weights":self.weights,"probabilities":probabilities,
           "joint_covariance":joint.tolist(),"estimate":estimate,"SE":se,"CI_low":estimate-z*se,
           "CI_high":estimate+z*se,"p_normal":float(2*stats.norm.sf(abs(estimate/se))),
           "scale":"probability; multiply by 100 for percentage points","standardization_n":fit.n}
    @classmethod
    def pair(cls,lower:float,upper:float): return cls((lower,upper),(1.,-1.))
    @classmethod
    def local_excess(cls,scores:tuple[float,float,float,float]): return cls(scores,(-.5,1.5,-1.5,.5))

@dataclass(frozen=True)
class FixedEffectPool:
    estimates: tuple[float,...]
    standard_errors: tuple[float,...]
    def evaluate(self)->dict:
        e=np.asarray(self.estimates,float);s=np.asarray(self.standard_errors,float)
        require(len(e)==len(s)>=2 and np.isfinite(e).all() and np.isfinite(s).all() and np.all(s>0),"Invalid pooling inputs")
        w=1/s**2;m=float(w@e/w.sum());se=float(1/np.sqrt(w.sum()));q=float(w@((e-m)**2));z=stats.norm.ppf(.975)
        return {"estimate":m,"SE":se,"CI_low":m-z*se,"CI_high":m+z*se,"Q":q,"Q_df":len(e)-1,
          "heterogeneity_p":float(stats.chi2.sf(q,len(e)-1)),"model":"inverse-variance fixed effect",
          "assumption":"independent estimates on the same additive scale"}
