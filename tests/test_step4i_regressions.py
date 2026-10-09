from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from ordinal_boundary import AffinePower, AnalysisSample, AnalysisSpec, BinaryGLM, Categorical, ContractError, Design, MetricRule, Polynomial, ResultContract, StandardizedContrast
from ordinal_boundary.synthetic import Simulation
from ordinal_boundary.studies.primary import StudyContract, StudyRunner, OUTCOMES
from ordinal_boundary.studies.simulated import simulated_eligible

ROOT=Path(__file__).resolve().parents[1]

def test_intervention_recomputes_exposure_transform():
    frame=Simulation(size=900,seed=812).draw()
    spec=AnalysisSpec('ordinal','target','group',('adjustment','ordinal_square'),Categorical(),2,9,transforms=(AffinePower('ordinal','ordinal_square',power=2),))
    design=Design.reference(AnalysisSample.from_frame(frame,spec))
    idx=np.where(frame.ordinal.to_numpy()==5)[0]
    assert np.allclose(design.intervention(5)[:, -1],25)
    assert np.allclose(design.intervention(5)[idx],design.matrix[idx])

def test_empty_validation_rules_rejected():
    with pytest.raises(ContractError):
        ResultContract(('era',),()).compare([{'era':'a','v':1}],[{'era':'a','v':2}])
    with pytest.raises(ContractError):
        ResultContract((),(MetricRule('v','v'),)).compare([{'v':1}],[{'v':1}])

def test_nonempty_validation_rules_work():
    result=ResultContract(('era',),(MetricRule('v','v',tolerance=.01),)).compare([{'era':'a','v':1}],[{'era':'a','v':1.005}])
    assert result['all_passed'] and result['metric_cells']==1

def test_reject_chained_transforms():
    with pytest.raises(ContractError):
        AnalysisSpec('ordinal','target','group',('adjustment','b'),Polynomial(2),2,9,transforms=(AffinePower('ordinal','a'),AffinePower('a','b')))

@pytest.mark.parametrize('era',('2008-2014','2019-2023'))
def test_study_contract_all_outcomes(era):
    path=ROOT/'configs'/('sjtrem_'+era.replace('-','_')+'.json')
    contract=StudyContract.from_file(path)
    for outcome in OUTCOMES:
        for task,rank in [('q1',12),('q2',23),('q3_local',13),('q3_full',15),('s18',15)]:
            spec=contract.for_task(task,outcome)
            assert spec.expected_rank==rank
            assert spec.outcome==outcome

def test_four_principal_tasks_on_simulated_data():
    contract=StudyContract.from_file(ROOT/'configs/sjtrem_2008_2014.json')
    runner=StudyRunner(contract)
    frame=simulated_eligible('2008-2014',seed=299)
    assert runner.execute(frame,'q1','intubation')['n8']>0
    assert np.isfinite(runner.execute(frame,'q2','intubation')['local_excess']['estimate'])
    assert runner.execute(frame,'q3_local','intubation')['n_comparisons']==1
    assert runner.execute(frame,'q3_full','intubation')['n_comparisons']==12
    assert runner.execute(frame,'s18','intubation')['n_comparisons']==7

def test_two_era_q1_pooling():
    from ordinal_boundary.studies.primary import combine_q1
    left=StudyRunner(StudyContract.from_file(ROOT/'configs/sjtrem_2008_2014.json')).execute(simulated_eligible('2008-2014',seed=301),'q1','intubation')
    right=StudyRunner(StudyContract.from_file(ROOT/'configs/sjtrem_2019_2023.json')).execute(simulated_eligible('2019-2023',seed=302),'q1','intubation')
    result=combine_q1(left,right)
    assert result['pooled_odds_ratio']>0
    assert 0<=result['era_difference_p_normal']<=1
    with pytest.raises(ContractError):
        combine_q1(left,left)


@pytest.mark.parametrize('task,expected_size',(('q1',179),('s18',719)))
def test_complete_case_year_missing_handling(task,expected_size):
    contract=StudyContract.from_file(ROOT/'configs/sjtrem_2008_2014.json')
    frame=simulated_eligible('2008-2014',seed=613)
    frame.loc[frame.index[frame.gcs==4][0],'year']=np.nan
    frame.loc[frame.index[frame.gcs==8][0],'year']=np.nan
    spec=contract.for_task(task,'intubation')
    sample=AnalysisSample.from_frame(frame,spec)
    assert sample.n==expected_size
    assert sample._frame['year'].notna().all()


def test_included_out_of_era_year_fails():
    contract=StudyContract.from_file(ROOT/'configs/sjtrem_2008_2014.json')
    frame=simulated_eligible('2008-2014',seed=714)
    frame.loc[frame.index[frame.gcs==8][0],'year']=2019
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(frame,contract.for_task('s18','intubation'))
