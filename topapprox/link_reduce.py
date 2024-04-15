"""main module for topapprox

Todo:
    * docstring
    
"""

__all__ = ["_link_reduce"]

import numpy as np
from numba import njit,f8,i8,prange,types
from numba.typed import List,Dict

#'Tuple((f8[:],f8[:,:]))(f8[:],i8[:,:],f8)'
@njit(parallel=True, fastmath=True) # cache=True
def _link_reduce(birth, edges, epsilon, keep_basin=False):
    """link and reduce

    Todo:
        * rewrite with Union-find with bookkeeping

    """
    persistence = List()
    persistence.append((birth.min(),np.inf,birth.argmin()))
    parent = np.arange(birth.shape[0])
    children = dict()
    #children = Dict.empty(key_type=i8,value_type=i8[:]) # typing makes slower...
    modified = birth.copy()
    for i in range(edges.shape[0]):
        ui,vi = edges[i,0],edges[i,1]
        # # parents of u and v
        up = parent[ui]
        vp = parent[vi]
        death = max(birth[ui],birth[vi])
        if up != vp: # if their bosses are different (i.e., they belong to different families)
            if birth[up] < birth[vp]:
                up,vp = vp,up   # up is the younger and to be killed
            # the basin of the younger family will be merged into older's
            if up not in list(children.keys()):
                region = np.array([up])
            else:
                region = children[up].copy()
                if not keep_basin:
                    del children[up]
            if birth[up] < death: # a cycle is produced
                persistence.append((birth[up],death,up))
                if keep_basin and (up not in list(children.keys())):
                    children[up] = np.array([up])
                if abs(birth[up]-death)<epsilon:
                    modified[region]=death
            parent[region] = vp
            if vp in list(children.keys()):
                children[vp] = np.append(children[vp], region) # can we skip creating a new array?
            else:
                children[vp] = np.append([vp], region)
    return(modified,persistence,children)


