from dataclasses import replace
import json
import numpy as np
import pytest
from scipy.special import expit
from ordinal_boundary import (
    AffinePower, AnalysisSample, AnalysisSpec, AggregateExporter, BinaryGLM,
    BoundaryScan, Categorical, CheckpointStore, ContractError, Design,
    FixedEffectPool, NaturalSpline, NestedComparison, NumericRange, Polynomial,
    StandardizedContrast, spec_from_dict,
)
from ordinal_boundary.synthetic import Simulation


@pytest.fixture(scope="module")
def d():
    return Simulation(size=840, seed=1183).draw()


def s(basis=None):
    return AnalysisSpec("ordinal", "target", "group", ("adjustment",), basis or Polynomial(2, 5, 3), 2, 9)


@pytest.mark.parametrize("name", ["x[0]", "in valid", "x;exec", "0score", "a.b", ""])
def test_invalid_identifiers(name):
    with pytest.raises(ContractError):
        replace(s(), exposure=name)


@pytest.mark.parametrize("knots", [(6, 4), (4, 4), (1, 3), (3, 10)])
def test_reject_invalid_basis(knots):
    with pytest.raises(ContractError):
        NaturalSpline(2, 9, knots)


def test_spec_serialize_roundtrip():
    a = replace(s(NaturalSpline(2, 9, (4, 6, 8))), transforms=(AffinePower("adjustment", "z", power=2),), covariates=("z",))
    b = spec_from_dict(json.loads(json.dumps(a.spec())))
    assert a.fingerprint == b.fingerprint


def test_rule_roundtrip():
    a = replace(s(), input_rules=(NumericRange("ordinal", 2, 9, integer=True),))
    assert spec_from_dict(json.loads(json.dumps(a.spec()))).fingerprint == a.fingerprint


def test_transform_collision(d):
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(d, replace(s(), transforms=(AffinePower("adjustment", "adjustment"),)))


def test_missing_input(d):
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(d.drop(columns="adjustment"), s())


def test_invalid_cluster(d):
    x = d.copy()
    x["group"] = "same"
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(x, s())


def test_duplicate_id(d):
    x = d.copy()
    x["record"] = "same"
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(x, replace(s(), id_column="record"))


def test_missingness_common_mask(d):
    x = d.copy()
    x.loc[0, "adjustment"] = np.nan
    x.loc[1, "target"] = np.nan
    sample = AnalysisSample.from_frame(x, s())
    assert sample.n == len(x) - 2
    r = BoundaryScan(s(), (4.5, 5.5)).run(x)
    assert len({v["sample_fingerprint"] for v in r}) == 1


def test_infinite_value_rejected(d):
    x = d.copy()
    x.loc[0, "adjustment"] = np.inf
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(x, s())


def test_missing_required_score(d):
    x = d[d.ordinal != 3]
    with pytest.raises(ContractError):
        AnalysisSample.from_frame(x, replace(s(), required_scores=tuple(range(2, 10))))


def test_reject_saturated_added_step(d):
    des = Design.reference(AnalysisSample.from_frame(d, s(Categorical())))
    with pytest.raises(ContractError):
        BinaryGLM().fit(des.with_step(5.5))


def test_duplicate_covariate_rank(d):
    x = d.copy()
    x["duplicate"] = x["adjustment"]
    with pytest.raises(ContractError):
        BinaryGLM().fit(Design.reference(AnalysisSample.from_frame(x, replace(s(), covariates=("adjustment", "duplicate")))))


def test_intervention_preserves_design_info(d):
    des = Design.reference(AnalysisSample.from_frame(d, s(NaturalSpline(2, 9, (4, 6, 8)))))
    assert np.array_equal(des.intervention(5)[d.ordinal.to_numpy() == 5], des.matrix[d.ordinal.to_numpy() == 5])


def test_gradient_against_finite_differences(d):
    fit = BinaryGLM().fit(Design.reference(AnalysisSample.from_frame(d, s(Categorical()))))
    q = StandardizedContrast.local_excess((4, 5, 6, 7))
    record = q.evaluate(fit)
    def value(b):
        return sum(w * expit(fit.design.intervention(score) @ b).mean() for score, w in zip(q.scores, q.weights))
    h = 1e-5
    grads = []
    for i in range(fit.k):
        eps = np.zeros(fit.k)
        eps[i] = h
        grads.append((value(fit.coefficients + eps) - value(fit.coefficients - eps)) / (2*h))
    assert np.isclose(record["SE"] ** 2, np.asarray(grads) @ fit.covariance @ np.asarray(grads), rtol=1e-7)


def test_preserve_input(d):
    original = d.copy(deep=True)
    sample = AnalysisSample.from_frame(d, s())
    view = sample.frame()
    view.loc[0, "ordinal"] = 999
    assert d.equals(original)
    assert sample.frame().loc[0, "ordinal"] != 999


def test_checkpoint_data_changes_refused(d, tmp_path):
    scan = BoundaryScan(s(), (4.5, 5.5))
    scan.run(d, tmp_path/"results")
    changed = d.copy()
    changed.loc[0, "adjustment"] += 0.5
    with pytest.raises(ContractError):
        scan.run(changed, tmp_path/"results")


def test_checkpoint_tampering_detected(d, tmp_path):
    scan = BoundaryScan(s(), (4.5,))
    location = tmp_path/"results"
    scan.run(d, location)
    record = next(p for p in location.glob("*.json") if len(p.stem) == 64)
    dct = json.loads(record.read_text())
    dct["record"]["n"] += 3
    record.write_text(json.dumps(dct))
    with pytest.raises(ContractError):
        scan.run(d, location)


def test_export_rejects_unapproved_fields(d, tmp_path):
    row = BoundaryScan(s(), (5.5,)).run(d)
    row[0]["record_ids"] = ["synthetic"]
    with pytest.raises(ContractError):
        AggregateExporter.write(tmp_path/"out.csv", row)


def test_pool_fail_closed():
    assert FixedEffectPool((0.2, 0.5), (0.1, 0.2)).evaluate()["estimate"] > 0.2
    with pytest.raises(ContractError):
        FixedEffectPool((0.2, 0.5), (0.0, 0.2)).evaluate()


def test_mismatched_sample_is_not_nested(d):
    sp = s()
    engine = BinaryGLM()
    ref = engine.fit(Design.reference(AnalysisSample.from_frame(d, sp)))
    alt = engine.fit(Design.reference(AnalysisSample.from_frame(d.iloc[:-1], sp)).with_step(5.5))
    with pytest.raises(ContractError):
        NestedComparison(ref, alt)
