from pathlib import Path
import csv
import pytest
from ordinal_boundary.studies.primary import StudyContract, StudyRunner
from ordinal_boundary.studies.simulated import simulated_eligible
from ordinal_boundary.storage import AggregateExporter
from ordinal_boundary.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('task,era,count', [
    ('q3_full','2008-2014',12),
    ('s18','2019-2023',7),
])
def test_revised_boundary_results_are_exportable(tmp_path,task,era,count):
    cfg=StudyContract.from_file(ROOT/'configs'/('sjtrem_'+era.replace('-','_')+'.json'))
    rows=StudyRunner(cfg).execute(simulated_eligible(era,seed=314),task,'intubation')['results']
    assert len(rows)==count
    dest=tmp_path/'out.csv'
    AggregateExporter.write(dest,rows)
    with dest.open(newline='') as f:
        table=list(csv.DictReader(f))
    assert len(table)==count
    assert {'step_SE_model_based','step_p_model_based'} <= set(table[0])
    assert all(0<=float(x['step_p_model_based'])<=1 for x in table)
    AggregateExporter.write(dest,rows)


def test_revised_boundary_export_rejects_unexpected_field(tmp_path):
    cfg=StudyContract.from_file(ROOT/'configs/sjtrem_2008_2014.json')
    row=StudyRunner(cfg).execute(simulated_eligible('2008-2014',seed=315),'s18','intubation')['results'][0].copy()
    row['participant_id']='SHOULD_NOT_EXPORT'
    with pytest.raises(ContractError):
        AggregateExporter.write(tmp_path/'forbidden.csv',[row])
    assert not (tmp_path/'forbidden.csv').exists()
