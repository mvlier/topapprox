import numpy as np
from .filter_image import TopologicalFilterImage
from .filter_graph import TopologicalFilterGraph



class Filtered():

    def __init__(self, signal):
        self.signal = signal



class PersistenceFilter():
    '''Class for performing various kinds of persistence filtering on various data types.
    '''

    def __init__(self) -> None:
        self.type = None
        self.bht = []
        self.filtered = []
        self.filtered_type = [], # each type is a tuple 

    def load_signal(self, signal):
        if isinstance(signal, np.ndarray):
            self.signal = signal.ravel.copy()
            self.type = "array"
            self.dim = signal.dim
        elif isinstance(signal, list):
            if (not len(signal) == 3) or (not isinstance(signal[0], list)) or (not isinstance(signal[1], list) or (not isinstance(signal[2], np.ndarray))):
                raise TypeError("First argument should be either an nd.array or a list of length 2,\
                                 containing a list of faces and a dictionary of signal values.")
            
            self.signal = signal[2]
            self.faces = signal[0]
            self.holes = signal[1]
            self.type = "graph"
    
    def low_pers_filter(self, epsilon, *, iteration_order="01", method="cpp"):
        n_it = len(iteration_order)
        for i in range(n_it):
            j = n_it-i
            current_iteration = iteration_order[:j]
            remaining_iterations = iteration_order[j:]
            if current_iteration in self.filtered_type:
                break
        
        for i in range(j):
            pass







        if self.type == "array":
            if self.dim == 2:
                if dual:
                    if "1" in self.filtered_type:
                        return self.filtered[self.filtered_type.index("1")]
                        self.filter[1] = TopologicalFilterImage(self.signal, dual=dual, method=method)
                        i=1
                else:
                    if self.filter[0] is None:
                        self.filter[0] = TopologicalFilterImage(self.signal, dual=dual, method=method)
                        i=0

                return self.filter[i].low_pers_filter()
                

    
        '''
         - filtered signal
         - filtrations (e.g. (low,0,eps1), (low,1,eps2))

         [(0,eps), (1,eps2), ()]
        '''