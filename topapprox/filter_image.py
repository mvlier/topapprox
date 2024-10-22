"""
Topapprox - Image Filtering Main Module

This module implements topological filtering for images, using persistent homology to perform low persistence filtering.
More explicitely, given a 1D or 2D array, we can compute the persistence diagram for the sublevel filtration of this
array, and by choosing a threshold `epsilon` we can filter out all elements with persistence less than `epsilon`
in the diagram. This module makes it possible to realize this filtered persistence diagram as a function which is
at a distance of at most `epsilon` from the original function in the l-infinity norm.

The module supports three different methods for computing the required operations: 
1. A pure Python implementation
2. A Numba-optimized implementation
3. A C++ extension for higher performance.

Classes:
    TopologicalFilterImage: Base class for applying low persistence filtering to images.

Todo:
    * Adapt `TopologicalFilterImage so that it can filter 0 and 1 homology alternating until now low persistence class exists
    * Remove obsolete attributes and methods, such as `keep_basin` and `persistence` in the constructor.
    * Improve error handling and warnings for fallback scenarios.
    * Develop a class for meshes
    * Edit _link_reduce so that it can compute the BHT without having to compute the modified function.
    * Optimize the way basin_size is computed
"""

import numpy as np
from .mixins.Method_Loader import MethodLoaderMixin
from .bht import BasinHierarchyTree


class TopologicalFilterImage(MethodLoaderMixin):
    """Base class for topological filtering for images

    Compute topological low persistence filtering for images
        
    Attributes:
        shape (tuple[int,int]): shape of the image
        persistence (np.array): persistent homology. each row indicates (birth,death,brith index)
        dual (bool): flag for duality (PH1)
        method (str): Which method to use for `_link_reduce`, options are "python", "numba" or "cpp" (fastest)
    """
    
    # Changed cpp to python
    # changed vert
    def __init__(self, img, *, method="cpp", dual=False, recursive=True, iter_vertex=False):
        self.shape = img.shape
        self.method = self.load_method(method, __package__, iter_vertex=iter_vertex) # python, numba or C++
        self.bht = BasinHierarchyTree(recursive=recursive)
        self.birth = img.ravel().copy() # filtration value for each vertex
        self.dual = dual
        self.edges = None
        self.persistence = None
        self.iter_vertex = iter_vertex
        if dual:
            self.bht.birth = np.concatenate((-self.birth, np.array([-np.inf])))
        else:
            self.bht.birth = self.birth.copy()

        
    #TODO: `keep_basin` now became obsolete, we have to either completely remove it, or include an option to 
    #      save all the basins when it is called. 
    # The reason it is obsolete is that now we have a list of children (`self.children`), and even more a list 
    # of non zero persistence children (`self.persistent_children``)
    def low_pers_filter(self, epsilon, *, size_range = None):
        """ computes topological high-pass filtering
        Args:
            epsilon (float): cycles having persistence below this value will be eliminated
            keep_basin (bool): if set to True, basin information will be stored for re-use. This makes the computation much slower but effective when filterings for multiple epsilons are computed.
            method (string): can be chosen between "python", "numba" or "cpp".
        Returns:
            np.array: a filtered image
        """
        if self.bht.children is None:
            self._update_BHT()

        if size_range is None:
            modified = self.bht._low_pers_filter(epsilon)
        else:
            modified = self.bht._lpf_size_filter(epsilon, size_range=size_range)
        if(self.dual):
            modified = -modified[:-1]
        modified = modified.reshape(self.shape)
        return(modified)

        
    def _update_BHT(self):
        '''Updates BHT via link_reduce method.
        One essential ingredient for obtaining the BHT is the '''
        if self.iter_vertex:
            self.bht.parent, \
                self.bht.children, \
                    self.bht.root, \
                        self.bht.linking_vertex, \
                            self.bht.persistent_children, \
                                self.bht.positive_pers = self._link_reduce(self.bht.birth, self.shape, self.dual)
        else:
            if self.edges is None:
                self._compute_sorted_edges()
            self.bht.parent, \
                self.bht.children, \
                    self.bht.root, \
                        self.bht.linking_vertex, \
                            self.bht.persistent_children, \
                                self.bht.positive_pers = self._link_reduce(self.bht.birth, self.edges, 0)

        
    def _compute_sorted_edges(self):
        ''' Saves the sorted edges in `self.edges`.
        '''
        # create graph
        n,m = self.shape
        edges = [(i*m + j, (i+1)*m + j) for i in range(n-1) for j in range(m)]+[(i*m + j, i*m + j+1) for j in range(m-1) for i in range(n)] #Edges for the grid case
        # edges = np.array([(np.ravel_multi_index(u, self.shape), np.ravel_multi_index(v, self.shape)) for u, v in E])
        # birth_edges = np.max([(self.birth[a[0]],self.birth[a[1]]) for a in edges], axis=1) #Birth value for each edge

        if self.dual:
            edges += [(i*m + j, (i+1)*m + j+1) for i in range(n-1) for j in range(m-1)] + \
                 [(i*m + j, (i-1)*m + j+1) for i in range(1, n) for j in range(m-1)]

            # Add one vertex valued -infty and connect the boundary to the vertex n*m
            edges += [(j, n*m) for j in range(m)] + \
                     [((n-1)*m + j, n*m) for j in range(m)] + \
                     [(i*m, n*m) for i in range(1, n-1)] + \
                     [(i*m + m-1, n*m) for i in range(1, n-1)]
            
            edges = np.array(edges, dtype=np.uint32)
            birth_edges = np.maximum(self.bht.birth[edges[:, 0]], self.bht.birth[edges[:, 1]])
        else:
            edges = np.array(edges, dtype=np.uint32)
            birth_edges = np.maximum(self.birth[edges[:, 0]], self.birth[edges[:, 1]])
        sorted_indices = np.argsort(birth_edges)
        self.edges = edges[sorted_indices]
        # print(f"Edges:{self.edges}")

    def get_BHT(self, *, with_children=False):
        return self.bht._get_BHT(with_children=with_children)
    
    def get_persistence(self, *, reduced=True):
        if self.bht.children is None:
            self._update_BHT()
        return self.bht.get_persistence(reduced=reduced)

        





        


