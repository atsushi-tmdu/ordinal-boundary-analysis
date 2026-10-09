from ordinal_boundary import (
    AnalysisSpec,
    Categorical,
    ContrastFactory,
    Estimand,
    ExecutionGraph,
    FitRequest,
    NaturalSpline,
    LogOddsContrast,
    Polynomial,
)
from ordinal_boundary.synthetic import Simulation


def main():
    data = Simulation(size=800).draw()
    common = ("ordinal", "target", "group", ("adjustment",))
    graph = ExecutionGraph((
        FitRequest(
            "categorical",
            AnalysisSpec(*common, Categorical(), 2, 9),
            (Estimand("midpoint_difference", ContrastFactory.difference(5, 6)),
             Estimand("local_shape", ContrastFactory.local_excess(4, 5, 6, 7)),
             Estimand("relative_odds", LogOddsContrast(5, 6))),
        ),
        FitRequest(
            "polynomial",
            AnalysisSpec(*common, Polynomial(3, 5, 3), 2, 9),
            (Estimand("midpoint_difference", ContrastFactory.difference(5, 6)),),
            (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5),
        ),
        FitRequest(
            "natural_spline",
            AnalysisSpec(*common, NaturalSpline(2, 9, (4, 6, 8)), 2, 9),
            (),
            (3.5, 4.5, 5.5, 6.5, 7.5),
        ),
    ))
    result = graph.execute(data)
    print({name: (row["sample_size"], len(row["comparisons"])) for name, row in result.items()})


if __name__ == "__main__":
    main()
