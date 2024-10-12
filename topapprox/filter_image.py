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


class TopologicalFilterImage(MethodLoaderMixin):
    """Base class for topological filtering for images

    Compute topological low persistence filtering for images
        
    Attributes:
        shape (tuple[int,int]): shape of the image
        persistence (np.array): persistent homology. each row indicates (birth,death,brith index)
        dual (bool): flag for duality (PH1)
        method (str): Which method to use for `_link_reduce`, options are "python", "numba" or "cpp" (fastest)
    """
    
    def __init__(self, img, *, dual=False, method="cpp"):
        self.shape = img.shape
        self.persistence = None
        self.dual = dual
        self.children = None
        self.persistent_children = None
        self.parent = None
        self.positive_pers = None # 1D numpy array to store the list of vertices with positive persistence, apart from the root
        self.method = self.load_method(method, __package__) # python, numba or C++

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

        self.descendants = None #TODO use this to save descendants
        birth_edges = np.max([(img_extended[a[0]],img_extended[a[1]]) for a in E], axis=1) #Birth value for each edge
        sorted_indices = np.argsort(birth_edges)
        self.edges = np.array([(np.ravel_multi_index(u, self.shape), np.ravel_multi_index(v, self.shape)) for u, v in E])[sorted_indices]

        
    #TODO: `keep_basin` now became obsolete, we have to either completely remove it, or include an option to 
    #      save all the basins when it is called. 
    # The reason it is obsolete is that now we have a list of children (`self.children`), and even more a list 
    # of non zero persistence children (`self.persistent_children``)
    def low_pers_filter(self, epsilon, *, keep_basin=False):
        """ computes topological high-pass filtering
        Args:
            epsilon (float): cycles having persistence below this value will be eliminated
            keep_basin (bool): if set to True, basin information will be stored for re-use. This makes the computation much slower but effective when filterings for multiple epsilons are computed.
            method (string): can be chosen between "python", "numba" or "cpp".
        Returns:
            np.array: a filtered image
        """
        self.epsilon = epsilon
        if self.children is None:
            modified = self.update_link_reduce()
        else:
            # compute the modification using the stored children data if available
            modified = self.birth.copy()
            for vertex in self.persistent_children[self.root]:
                self.filter_branch(vertex, self.epsilon, modified)
        modified = modified.reshape(self.shape)
        if(self.dual):
            return(-modified[1:-1,1:-1])
        else:
            return(modified)
        
    def update_link_reduce(self):
        '''Updates BHT via link_reduce method, and returns the modified function'''
        modified, self.parent, self.children, self.root, self.linking_vertex, self.persistent_children, self.positive_pers = self._link_reduce(self.birth, self.edges, self.epsilon)
        return modified
        
    def get_BHT(self, *, with_children=False):
        ''' Returns the BHT as a list of parents, linking vertices and root. 
        Additionally, if `with_children` is True,
        also returns the list of children of each node
        '''
        if self.children == None:
            _ = self.update_link_reduce()
        if with_children:
            return self.parent, self.linking_vertex, self.root, [list(x) for x in self.children]
        return self.parent, self.linking_vertex, self.root
        

    def filter_branch(self, vertex, epsilon, modified):
        ''' Given a `vertex`, a threshold `epsilon` and the current state of the function `modified`,
        alters `modified` by eliminating all the classes in the branch of `v` in the BHT 
        with positive persistence lower than `epsilon`.
        '''
        linking_vertex = self.linking_vertex[vertex]
        persistence = self.birth[linking_vertex] - self.birth[vertex]
        if  0 < persistence < epsilon:
            modified[np.array(self.compute_descendants(vertex, self.children))] = self.birth[linking_vertex]
        elif persistence >= epsilon:
            for child_vertex in self.persistent_children[vertex]:
                self.filter_branch(child_vertex, epsilon, modified)

    def basin_size(self, vertex):
        '''Returns the size of the basin of `vertex`
        The basin is composed of `vertex` itself plus all its descendants'''
        _ = self.get_BHT()
        return len(self.compute_descendants(vertex, self.children))

    def get_persistence(self, *, reduced=True):
        '''Computes the reduced persistence diagram from the BHT'''
        if isinstance(self.persistence, np.ndarray):
            return self.persistence
        
        _ = self.get_BHT()
    
        #(birth, death, birth_location, death_location, basin_size)
        self.persistence = np.array([[self.birth[v], self.birth[self.linking_vertex[v]], v, self.linking_vertex[v], self.basin_size(v)] for v in self.positive_pers])

        if reduced==False:
            permanent_interval = np.array([[self.birth[self.root], np.inf, self.root, -1, np.inf]])
            self.persistence = np.concatenate((self.persistence, permanent_interval))
        
        return self.persistence
        



        


