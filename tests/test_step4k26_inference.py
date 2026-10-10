from pathlib import Path
import math
import numpy as np
import pytest
from scipy import stats
import statsmodels.api as sm
from ordinal_boundary.data import AnalysisSample, Design
from ordinal_boundary.models import BinaryGLM, NestedComparison
from ordinal_boundary.studies.primary import StudyContract, StudyRunner, combine_q1
from ordinal_boundary.studies.simulated import simulated_eligible

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('task,boundary',(('q3_local',8.5),('q3_full',8.5),('s18',8.5)))
def test_step_model_based_wald_matches_independent_nonrobust_glm(task,boundary):
    contract=StudyContract.from_file(ROOT/'configs/sjtrem_2008_2014.json')
    spec=contract.for_task(task,'intubation')
    sample=AnalysisSample.from_frame(simulated_eligible('2008-2014',seed=1108),spec)
    reference=Design.reference(sample)
    alternative=reference.with_step(boundary)
    estimator=BinaryGLM()
    row=NestedComparison(estimator.fit(reference),estimator.fit(alternative)).record()
    independent=sm.GLM(sample.y,alternative.matrix,family=sm.families.Binomial()).fit()
    fitted=1/(1+np.exp(-(alternative.matrix@independent.params)))
    fisher=alternative.matrix.T@((fitted*(1-fitted))[:,None]*alternative.matrix)
    axis=np.zeros(alternative.matrix.shape[1])
    axis[-1]=1.
    se_direct=np.sqrt(float(np.linalg.solve(fisher,axis)[-1]))
    p_direct=float(2*stats.norm.sf(abs(float(independent.params[-1])/se_direct)))
    assert row['step_p_model_based']==pytest.approx(p_direct,abs=1e-10)
    assert row['step_SE_model_based']==pytest.approx(se_direct,abs=1e-10)
    assert row['step_p_model_based']==pytest.approx(float(independent.pvalues[-1]),abs=1e-6)
    assert row['step_SE_model_based']==pytest.approx(float(independent.bse[-1]),abs=1e-6)
    assert 0 <= row['step_p_cluster'] <= 1
    assert 0 <= row['LR_p'] <= 1
    assert row['step_SE_model_based'] > 0

@pytest.mark.parametrize('outcome',('intubation','death'))
def test_q1_era_risk_difference_matches_independent_era_contrast(outcome):
    old=StudyRunner(StudyContract.from_file(ROOT/'configs/sjtrem_2008_2014.json')).execute(simulated_eligible('2008-2014',seed=902),'q1',outcome)
    new=StudyRunner(StudyContract.from_file(ROOT/'configs/sjtrem_2019_2023.json')).execute(simulated_eligible('2019-2023',seed=903),'q1',outcome)
    result=combine_q1(old,new)
    a,b=old['risk_difference'],new['risk_difference']
    diff=b['estimate']-a['estimate']
    se=math.sqrt(a['SE']**2+b['SE']**2)
    crit=stats.norm.ppf(.975)
    row=result['era_risk_difference']
    assert row['difference']==pytest.approx(diff)
    assert row['SE']==pytest.approx(se)
    assert row['CI_low']==pytest.approx(diff-crit*se)
    assert row['CI_high']==pytest.approx(diff+crit*se)
    assert row['p_normal']==pytest.approx(2*stats.norm.sf(abs(diff/se)))
    assert row['direction']=='second-minus-first'
    reverse=combine_q1(new,old)['era_risk_difference']
    assert reverse['difference']==pytest.approx(-row['difference'])
    assert reverse['p_normal']==pytest.approx(row['p_normal'])


def test_preserve_existing_bic_identity_with_model_based_wald():
    contract=StudyContract.from_file(ROOT/'configs/sjtrem_2019_2023.json')
    row=StudyRunner(contract).execute(simulated_eligible('2019-2023',seed=950),'q3_full','intubation')['results'][5]
    assert row['delta_BIC']==pytest.approx(math.log(row['n'])-row['LR_chi2'],abs=1e-8)
    assert row['delta_AIC']==pytest.approx(2-row['LR_chi2'],abs=1e-8)
    assert row['step_p_model_based']==pytest.approx(2*stats.norm.sf(abs(row['step_log_OR']/row['step_SE_model_based'])))
