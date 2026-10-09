import numpy as np
import pytest
from ordinal_boundary.synthetic import Simulation


def test_deterministic():
    left=Simulation(size=200,seed=71).draw()
    right=Simulation(size=200,seed=71).draw()
    assert left.equals(right)
    assert set(left.columns)=={"ordinal","target","adjustment","group"}
    assert set(left.target.unique())=={0,1}


@pytest.mark.parametrize("n,groups,lo,hi", [(99,10,2,9),(100,1,2,9),(100,10,8,2)])
def test_invalid_simulation(n,groups,lo,hi):
    with pytest.raises(ValueError):
        Simulation(size=n,clusters=groups,lower=lo,upper=hi).draw()
