import pytest
from topapprox import ImageFilter, available_backends
import numpy as np

cripser = pytest.importorskip("cripser")

pytestmark = pytest.mark.skipif("cpp" not in available_backends(dimensions=3), reason="3D filtering requires the C++ backend.")

def test_compare_cripser_3D():
    '''Compares the persistent diagrams for 0 and 2 homology
    obtained by topapprox and by cripser for a 3D array.'''
    

    n,m,l = 50,50,50
    arr3D = np.random.rand(n,m,l)

    tfi = ImageFilter(arr3D)
    tfi._update_BHT()
    pd_topapprox_0 = tfi.bht.get_persistence()
    tfi_dual = ImageFilter(arr3D, dual=True)
    tfi_dual._update_BHT()
    pd_topapprox_2 = tfi_dual.bht.get_persistence()

    pd_cripser = cripser.computePH(arr3D, maxdim=2)
    pd_cripser_0 = pd_cripser[pd_cripser[:,0]==0]
    pd_cripser_2 = pd_cripser[pd_cripser[:,0]==2]

    assert compare_pd_toapprox_cripser(pd_topapprox_0, pd_cripser_0), error_msg("pd0")

    assert compare_pd_toapprox_cripser(pd_topapprox_2, pd_cripser_2, dual=True), error_msg("pd2")

def compare_pd_toapprox_cripser(pd_topapprox, pd_cripser, *, dual=False) -> bool:
    '''Receives Persistence Diagrams (PD) computed using 
    topapprox and cripser, and compares them. 
    The PD obtained via cripser should be cleaned to have only
    one dimension.

    Parameter:
    ---------
    pd_topapprox: np.ndarray
    pd_cripser: np.ndarray

    Output:
    ------
    result: bool
            True if the two diagrams are the same, and False if not.
    '''

    pd_ta = pd_topapprox[:,0:2]
    pd_cr = pd_cripser[:,1:3] #eliminates the permanent cycle

    #sorting both diagrams lexicographically for comparison
    pd_ta = pd_ta[np.lexsort((pd_ta[:,1], pd_ta[:,0]))]
    pd_cr = pd_cr[np.lexsort((pd_cr[:,1], pd_cr[:,0]))]

    if not dual:
        pd_cr = pd_cr[1:]

    if pd_ta.shape != pd_cr.shape:
        result = False
    else:
        result = np.all(pd_ta == pd_cr)

    return result


def error_msg(err_name):
    '''
    Error msg for pd0 and pd1.

    Parameter:
    ---------
    err_name: string
              Either "pd0" or "pd1"
    '''
    dim = err_name[-1]

    msg = f"""
Error {err_name}: The {dim}-homology persistent diagrams for the 3D case are distinct.
"""
    return(msg)
