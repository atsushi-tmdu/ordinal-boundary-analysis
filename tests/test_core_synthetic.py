from pathlib import Path
import math
import numpy as np
import pytest
from ordinal_boundary import (
    AnalysisSample,
    AnalysisSpec,
    BinaryGLM,
    BoundaryScan,
    Categorical,
    ContractError,
    Design,
    FitOptions,
    InformationCriterionAudit,
    NaturalSpline,
    Polynomial,
    StandardizedContrast,
)
from ordinal_boundary.synthetic import Simulation


@pytest.fixture(scope="module")
def data():
    return Simulation(size=1800, seed=29).draw()


def make_spec(basis):
    return AnalysisSpec("ordinal", "target", "group", ("adjustment",), basis, 2, 9)


@pytest.mark.parametrize("basis", [Polynomial(2, 5, 3), Polynomial(4, 5, 3), Categorical(), NaturalSpline(2, 9, (4, 6, 8))])
def test_estimator_validates_cluster_covariance(data, basis):
    fit = BinaryGLM(FitOptions(threads=1)).fit(Design.reference(AnalysisSample.from_frame(data, make_spec(basis))))
    assert fit.llf < 0
    assert fit.n == 1800
    assert np.all(np.linalg.eigvalsh(fit.covariance) >= -1e-10)
    assert fit.diagnostics["CR1_max_abs_difference"] < 1e-8


def test_scan_checkpoint_reuse_and_identity_lock(data, tmp_path):
    spec = make_spec(Polynomial(2, 5, 3))
    scan = BoundaryScan(spec, (4.5, 5.5, 6.5))
    first = scan.run(data, checkpoint=tmp_path/"checkpoint")
    second = scan.run(data, checkpoint=tmp_path/"checkpoint")
    assert first == second
    with pytest.raises(ContractError):
        BoundaryScan(spec, (5.5,)).run(data, checkpoint=tmp_path/"checkpoint")


def test_scan_likelihood_and_information_criteria(data):
    rows = BoundaryScan(make_spec(Polynomial(3, 5, 3)), (3.5, 5.5)).run(data)
    assert all(InformationCriterionAudit.comparison(row)["passed"] for row in rows)


def test_invalid_outcome_fails_closed(data):
    wrong = data.copy()
    wrong.loc[0, "target"] = 2
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(wrong, make_spec(Categorical()))


def test_pair_functional(data):
    fit = BinaryGLM().fit(Design.reference(AnalysisSample.from_frame(data, make_spec(Categorical()))))
    result = StandardizedContrast.pair(5, 6).evaluate(fit)
    assert math.isclose(result["estimate"], result["probabilities"][0]-result["probabilities"][1], abs_tol=1e-12)
