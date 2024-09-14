"""main module for topapprox

Todo:
    * docstring
    
"""

import numpy as np
#from .link_reduce_numba import *   ## numba
from .link_reduce import link_reduce

class TopologicalFilterImage():
    """Base class for topological filtering for images

    Compute topological filtering for a image
        
    Attributes:
        shape (tuple[int,int]): shape of the image
        persistence (np.array): persistent homology. each row indicates (birth,death,brith index)
        dual (bool): flag for duality (PH1)
    """
    
    def __init__(self, img, dual=False):
        self.shape = img.shape
        self.persistence = None
        self.basin = None
        self.dual = dual
        self.parent = None

        # create graph
        n,m = img.shape
        if not dual:
            self.birth = img.ravel().copy() # filtration value for each vertex
            E = [((i,j),(i+1,j)) for i in range(n-1) for j in range(m)]+[((i,j),(i,j+1)) for j in range(m-1) for i in range(n)] #Edges for the grid case
            img_extended = img
        else:
            min_val = -np.inf #-img.max()-1 # this value serves as -infty
            img_extended = np.full((n+2,m+2),min_val)
            img_extended[1:-1,1:-1] = -img # embed the negative of original image with a "frame" filled with -infty
            self.birth = img_extended.ravel()
            self.shape = img_extended.shape
            E = [((i,j),(i+1,j)) for i in range(n+1) for j in range(m+2)]+[((i,j),(i,j+1)) for j in range(m+1) for i in range(n+2)] #Edges for the grid case
            E+= [((i,j),(i+1,j+1)) for i in range(n+1) for j in range(m+1)]+[((i,j),(i-1,j+1)) for i in range(1,n+2) for j in range(m+1)]

        birth_edges = np.max([(img_extended[a[0]],img_extended[a[1]]) for a in E], axis=1) #Birth value for each edge
        min_edges = np.min([(img_extended[a[0]],img_extended[a[1]]) for a in E], axis=1) #Minimum node value for each edge 
        # Sort edges first by birth_edges and then by -min_edges
        sorted_indices = np.lexsort((-min_edges, birth_edges))
        self.edges = np.array([(np.ravel_multi_index(u, self.shape), np.ravel_multi_index(v, self.shape)) for u, v in E])[sorted_indices]

        
    def low_pers_filter(self, epsilon, keep_basin=False):
        """ computes topological high-pass filtering
        Args:
            epsilon (float): cycles having persistence below this value will be eliminated
            keep_basin (bool): if set to True, basin information will be stored for re-use. This makes the computation much slower but effective when filterings for multiple epsilons are computed.
        Returns:
            np.array: a filtered image
        """
        self.epsilon = epsilon
        if self.basin is None:
            modified, persistence, basin, self.parent = link_reduce(self.birth, self.edges, self.epsilon, keep_basin=keep_basin)
            self.persistence=np.array(persistence)
            if keep_basin:
                self.basin = basin
        else:
            # re-compute the modification using the stored basin regions
            modified = self.birth.copy()
            for b,d,p in self.persistence:
                if (d-b) < epsilon:
                    modified[self.basin[int(p)]] = d
        # return the modified image
        modified = modified.reshape(self.shape)
        if(self.dual):
            return(-modified[1:-1,1:-1])
        else:
            return(modified)
        


