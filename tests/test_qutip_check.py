"""Optional QuTiP cross-check: a third independent route to the XX-chain entropy.
Skips cleanly when QuTiP is not installed (the numpy ED path is the guaranteed check).
"""
import pytest

from verify.qutip_check import qutip_available, xx_entropy_qutip
from verify.peschel import xx_entanglement_entropy

pytestmark = pytest.mark.skipif(not qutip_available(), reason="QuTiP not installed")


@pytest.mark.parametrize("N,L", [(2, 1), (4, 2), (6, 3), (8, 4)])
def test_qutip_matches_peschel(N, L):
    s_q = xx_entropy_qutip(N, L)
    s_p = xx_entanglement_entropy(N, L, bc="open")
    assert abs(s_q - s_p) < 1e-9
