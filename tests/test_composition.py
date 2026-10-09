import math
import numpy as np
import pytest
from ordinal_boundary import (
    AnalysisSample,
    AnalysisSpec,
    BinaryGLM,
    Categorical,
    ContrastFactory,
    ContractError,
    Design,
    Estimand,
    ExecutionGraph,
    FitRequest,
    NaturalSpline,
    LogOddsContrast,
    NestedComparison,
    Polynomial,
)
from ordinal_boundary.synthetic import Simulation


@pytest.fixture(scope="module")
def synthetic():
    return Simulation(size=1800, seed=941).draw()


def spec(basis):
    return AnalysisSpec("ordinal", "target", "group", ("adjustment",), basis, 2, 9)


def test_graph_multiple_models(synthetic):
    graph = ExecutionGraph((
        FitRequest("categorical", spec(Categorical()), (
            Estimand("difference", ContrastFactory.difference(5, 6)),
            Estimand("curvature", ContrastFactory.local_excess(4, 5, 6, 7)),
        )),
        FitRequest("polynomial", spec(Polynomial(3, 5, 3)), (), (4.5, 5.5)),
        FitRequest("spline", spec(NaturalSpline(2, 9, (4, 6, 8))), (), (5.5,)),
    ))
    result = graph.execute(synthetic)
    assert set(result) == {"categorical", "polynomial", "spline"}
    assert len(result["polynomial"]["comparisons"]) == 2
    assert len(result["spline"]["comparisons"]) == 1
    assert result["categorical"]["sample_size"] == len(synthetic)
    assert result["categorical"]["estimands"]["difference"]["SE"] > 0
    assert np.isfinite(result["categorical"]["estimands"]["curvature"]["estimate"])
    for row in result["polynomial"]["comparisons"] + result["spline"]["comparisons"]:
        assert math.isclose(row["delta_BIC"], math.log(row["n"]) - row["LR_chi2"], abs_tol=1e-8)


def test_composition_contrast_equivalence(synthetic):
    fit = BinaryGLM().fit(Design.reference(AnalysisSample.from_frame(synthetic, spec(Categorical()))))
    contrast = ContrastFactory.local_excess(4, 5, 6, 7).evaluate(fit)
    p4, p5, p6, p7 = contrast["probabilities"]
    assert math.isclose(contrast["estimate"], -0.5*p4 + 1.5*p5 -1.5*p6 + 0.5*p7, abs_tol=1e-12)
    assert 0 < contrast["p_normal"] <= 1


def test_reject_unsupported_or_duplicate_studies(synthetic):
    with pytest.raises(ContractError):
        ExecutionGraph((FitRequest("x", spec(Categorical())), FitRequest("x", spec(Categorical()))))
    with pytest.raises(ContractError):
        ContrastFactory.local_excess(5, 4, 6, 7)
    with pytest.raises(ContractError):
        FitRequest("x", spec(Categorical()), boundaries=(5.5, 5.5))


def test_nested_comparison_requires_shared_sample(synthetic):
    sample = AnalysisSample.from_frame(synthetic, spec(Polynomial(2, 5, 3)))
    design = Design.reference(sample)
    engine = BinaryGLM()
    a = engine.fit(design)
    b = engine.fit(design.with_step(5.5))
    assert NestedComparison(a, b).record()["reference_k"] + 1 == b.k


def test_no_patient_data_emitted_by_graph(synthetic):
    result = ExecutionGraph((FitRequest("simple", spec(Polynomial(2, 5, 3))),)).execute(synthetic)
    assert "ordinal" not in result["simple"]
    assert "target" not in result["simple"]
    assert "group" not in result["simple"]
    assert result["simple"]["sample_size"] == len(synthetic)


def test_relative_odds_functional(synthetic):
    fit = BinaryGLM().fit(Design.reference(AnalysisSample.from_frame(synthetic, spec(Categorical()))))
    r = LogOddsContrast(5, 6).evaluate(fit)
    assert r["odds_ratio_low"] < r["odds_ratio"] < r["odds_ratio_high"]
    assert math.isclose(math.log(r["odds_ratio"]), r["log_odds"], abs_tol=1e-12)
