"""
Topapprox - Graph Filtering Main Module

This module implements topological filtering for graphs, using persistent homology to perform low persistence filtering.
More explicitely, given a graph with faces embadabble, we can compute the persistence diagram for the sublevel filtration of this
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
import networkx as nx
import itertools
from collections import Counter
from .mixins.Method_Loader import MethodLoaderMixin
from .gwf import GraphWithFaces
from .bht import BasinHierarchyTree



#Class for graph input
class TopologicalFilterGraph(MethodLoaderMixin):
  def __init__(self, input=None, method="cpp", dual=False, recursive=True):
    self.method = self.load_method(method, __package__) # python, numba or C++
    self.modified = None
    self.G = None # graph structure
    self.pos = None #position for drawing graph
    self.diagram = [np.array([None]),np.array([None])] #Persistence diagram, with [PH0, PH1]
    self.children = None
    self.persistent_children = None
    self.parent = None
    self.shape = None
    self.gwf = None
    self.dual = dual
    self.bht = BasinHierarchyTree(recursive=recursive)
    self.compute = "dual" if self.dual else "normal"
    if isinstance(input,np.ndarray):
      self.from_array(input)
    elif input is not None:
      raise ValueError(f"Unknown initialisation type")

  def compute_gwf(self, F: list, H: list, signal: np.ndarray):
    """Computes the graph with faces
    """
    # TODO: adapt this class so that it is possible to compute normal 
    # and dual for the same class to optimize
    self.gwf = GraphWithFaces(F=F, H=H, signal=signal, compute=self.compute)


  ## create a graph with faces from image
  def from_array(self, img: np.array):
    self.shape = img.shape
    if len(img.shape)==2: # 2D case
      n,m = img.shape
      F = [[i*m + j,(i+1)*m +j, (i+1)*m + j+1, i*m + j+1] for i in range(n-1) for j in range(m-1)]
    elif len(img.shape)==1: # 1D case
      n,m = img.shape[0],1
      F = [[i, i+1] for i in range(n-1)]
    else:
      raise ValueError("array dimension should be 1 or 2!")
    boundary_vertices =  [j for j in range(m)] + [i*m + m-1 for i in range(1,n)] + \
                         [(n-1)*m + m - j for j in range(2,m+1)] + [m*(n-i) for i in range(2,n)]
    H = [boundary_vertices]
    self.signal = img.ravel()
    self.compute_gwf(F, H, self.signal)


  def _update_BHT(self):
    self.bht.birth = self.gwf.signal
    if self.dual:
      self.bht.parent, self.bht.children, self.bht.root, self.bht.linking_vertex, self.bht.persistent_children, self.bht.positive_pers = self._link_reduce(self.gwf.signal, self.gwf.dualE, 0)
    else:
      self.bht.parent, self.bht.children, self.bht.root, self.bht.linking_vertex, self.bht.persistent_children, self.bht.positive_pers = self._link_reduce(self.gwf.signal, self.gwf.E, 0)
    
  def low_pers_filter(self, epsilon, *, size_gap = None):
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

      if size_gap is None:
          modified = self.bht._low_pers_filter(epsilon)
      else:
          modified = self.bht._lpf_size_filter(epsilon, size_gap=size_gap)
      if self.shape is not None:
        if(self.dual):
            modified = -modified[:self.shape[0]*self.shape[1]]
        modified = modified.reshape(self.shape)
        return(modified)
      else:
        if(self.dual):
          n = self.gwf.signal.shape[0] - len(self.gwf.F) - len(self.gwf.H)
          modified = -modified[:n]
        return modified


  ## returns modified function values in the original array's shape
  def get_array(self):
    mod_img = np.zeros((self.n,self.m))
    for i in range(len(self.modified)):
      mod_img[self.idx2node[i]] = self.modified[i]
    return(mod_img.reshape(self.shape))

  def get_modified_filtration(self):
    return {u: self.modified[i] for i,u in enumerate(self.idx2node)}

  def _dualPH0_to_PH1(self,dualPH0):
    PH1 = list()
    for x in dualPH0:
      PH1.append((-x[1],-x[0]))
    if PH1 == list():
      return(np.reshape(np.array([]),(0,2)))
    return(np.array(PH1))

  #Returns the persistence diagram as list of numpy arrays [PH0,PH1]
  def get_diagram(self):
    if np.all(self.diagram[0]) == None:
      self.diagram[0] = np.array(self.compute())
    if np.all(self.diagram[1]) == None:
      self.diagram[1] = self._dualPH0_to_PH1(self.compute(dual=True))
    return(self.diagram)


  ## merge families and modify the filtration values.
  # (set epsilon=0 for leaving filtration values intact)
  def link(self, ui,vi, val=np.inf, epsilon=0):
      if ui==vi:
        return
      # apply the "elder rule" to determine who is the parent
      # (TODO) keeping a mapping parent => vertex indices can be more efficient
      if self.birth[ui] >= self.birth[vi]: # ui>vi is guaranteed
        if abs(self.birth[ui]-val)<epsilon:
          self.modified[self.parent == self.parent[ui]]=val
        self.parent[self.parent == self.parent[ui]]=self.parent[vi]
      elif self.birth[ui] < self.birth[vi]:
        if abs(self.birth[vi]-val)<epsilon:
          self.modified[self.parent == self.parent[vi]]=val
        self.parent[self.parent == self.parent[vi]]=self.parent[ui]

  # ## look at an edge and merge end points if necessary
  # def reduce(self,ui,vi):
  #     # parents of u and v
  #     up = self.parent[ui] # self.find(ui)
  #     vp = self.parent[vi] # self.find(vi)
  #     #print(up,vp,self.birth[up], self.birth[vp])
  #     # if self.birth[ui] > self.birth[vi]:
  #     #   killing = ui
  #     # elif (self.birth[ui] == self.birth[vi]) and ui>vi:
  #     #   killing = ui # break tie consistently
  #     # else:
  #     #   killing = vi
  #     # death = self.birth[killing] # filtration value of the added edge
  #     death = max(self.birth[ui],self.birth[vi])
  #     if up != vp: # if their bosses are different (i.e., they belong to different families)
  #       killed = up if self.birth[up] > self.birth[vp] else vp
  #       birth = self.birth[killed] # choose the younger one
  #       if birth < death: # one of families is killed to produce a cycle
  #         self.persistence.append((birth,death)) # (birth,death)
  #         #print(self.persistence[-1])
  #         self.link(up,vp,death,self.epsilon)
  #       else:
  #         self.link(up,vp)

  ## Low persistence filter for Graphs
  def _low_pers_filter(self, epsilon=0, dual=False, verbose=False):
    self.epsilon=epsilon
    ## mapping from node index to node name
    self.idx2node = self.vertices.copy()
    if dual:
      E = self.dualE
      sign = -1
      self.idx2node.extend(self.INF_V)
    else:
      E = self.E
      sign = 1

    ## mapping from node name to node index
    self.node2idx = {u: i for i,u in enumerate(self.idx2node)}
    self.modified = np.array([sign*self.filtration[u] for u in self.idx2node]) # the last entry is for the point at infinity
    E = np.sort([(self.node2idx[u],self.node2idx[v]) for u,v in E],axis=1)[:,::-1] # respect vertex order

    # if use_numba:
    #   ## TODO: still slower with numba
    #   self.modified, self.persistence, basin = self.link_reduce(self.modified, E, self.epsilon)
    # else:
    #   self.birth = self.modified.copy()
    #   self.parent = np.arange(len(self.birth))
    #   self.persistence = [(self.birth.min(),float('inf'))] # list of cycles in the form (birth,death). Initially, the permanent cycle is added.
    #   # iterating over edges
    for u,v in E:
      #print("edge ",(u,v),max(self.birth[ui],self.birth[vi]))
      # self._link_reduce(u,v)
      self.modified, self.parent, self.children, self.root, self.linking_vertex, self.persistent_children, self.positive_pers = self._link_reduce(self.modified, E, self.epsilon)

    if dual:
      self.modified = -self.modified[:-len(self.INF_V)] # remove the infinity point
      # self.persistence = self.persistence[1:] # remove the permanent cycle since Alexander duality looks at the reduced homology
      self.idx2node = self.idx2node[:-len(self.INF_V)]
    # self.persistence = np.where(np.array(self.persistence) > -self.infinite_filtration_value, self.persistence, -np.inf)
    return {u: self.modified[i] for i,u in enumerate(self.idx2node)}





  #######################################################################################
  #######################################################################################
  ################################ FOR DRAWING ##########################################
  #######################################################################################
  #######################################################################################

  def reorient(self):
    k=0
    List=list(range(len(self.faces)-1))
    self.faces = [list(f) for f in self.faces]
    while List:
      for j in List:
        interP=[x for x in self.faces[k] if x in self.faces[j]]
        if len(interP)>1:
          index_P0= self.faces[j].index(interP[0])
          # Determine the next index, or wrap around to the first index if P0 is the last element
          next_index = (index_P0 + 1) % len(self.faces[j])
          # Get the number next to 2
          next_number = self.faces[j][next_index]
          if next_number==interP[1]:
            self.faces[j+1].reverse()
        k=j
        List.remove(j)
        break


  def pos_from_faces(self): ## still buggy
    self.reorient()
    PE = nx.PlanarEmbedding()
    for node in self.boundary:
      faces = [face for face in self.faces if node in face]

      #Getting a neighbour (nb1) of node, which only shares one face with node
      numb_of_faces = {u: len([x for x in faces if u in x]) for u in self.G.neighbors(node)}
      nb1 = [x for x, v in numb_of_faces.items() if v==1][0]

      face = [f for f in faces if nb1 in f][0]
      idx = face.index(node)
      if nb1 == face[idx-1]:
        add_edge = PE.add_half_edge_cw
        next_nbh = lambda f : f[idx-len(f)+1]
      else :
        add_edge = PE.add_half_edge_ccw
        next_nbh = lambda f : f[idx-1]
      nb2 = next_nbh(face)
      add_edge(node,nb1,None)
      add_edge(node,nb2,nb1)
      faces.pop(faces.index(face))
      while len(faces)>0:
        face = [f for f in faces if nb2 in f]
        if len(face)>0:
          face = face[0]
          nb1 = nb2
          idx = face.index(node)
          nb2 = next_nbh(face)
          add_edge(node,nb2,nb1)
          faces.pop(faces.index(face))
        else:
          break

    internal_nodes = set(self.G.nodes) - set(self.boundary)

    for node in internal_nodes:
      add_edge = PE.add_half_edge_cw
      faces = [face for face in self.faces if node in face]
      face = faces[-1]
      idx = face.index(node)
      nb1 = face[idx-1]
      nb2 = face[idx-len(face)+1]
      add_edge(node, nb1, None)
      add_edge(node, nb2, nb1)
      faces.pop()
      while len(faces) > 1:
        face = [f for f in faces if nb2 in f]
        if len(face)>0:
          face = face[0]
          idx = face.index(node)
          nb1 = nb2
          nb2 = face[idx-len(face)+1]
          add_edge(node, nb2, nb1)
          faces.pop(faces.index(face))
        else:
          break
    try:
      PE.check_structure()
      pos = nx.combinatorial_embedding_to_pos(PE)
    except:
      print("NOT GOOD")
      return(None)
    return(pos)  
    
  def draw(self, *, with_filtration=False, with_labels=False, modified=False, ax=None, node_size=600, font_size=8):
    if self.G is None:
      self.G = nx.Graph()
      self.G.add_edges_from(self.E)
    if self.pos == None:
      self.pos = self.pos_from_faces()
    #Drawing
    if ax == None:
      if with_labels:
        if modified:
          mf = self.get_modified_filtration()
          nx.draw(self.G, self.pos, labels={u:f'V{u} : {mf[u]}' for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size)
        elif with_filtration:
          nx.draw(self.G, self.pos, labels={u:f'V{u} : {self.filtration[u]}' for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size)
        else:
          nx.draw(self.G, self.pos, labels = {u:f'V{u}' for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size)
      else:
        if modified:
          mf = self.get_modified_filtration()
          nx.draw(self.G, self.pos, labels={u:mf[u] for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size)
        elif with_filtration:
          nx.draw(self.G, self.pos, labels={u:self.filtration[u] for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size)
        else:
          nx.draw(self.G, self.pos, with_labels = False, node_size=node_size, font_size=font_size)
    else:
      if with_labels:
        if modified:
          mf = self.get_modified_filtration()
          nx.draw(self.G, self.pos, labels={u:f'V{u} : {mf[u]}' for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size, ax=ax)
        elif with_filtration:
          nx.draw(self.G, self.pos, labels={u:f'V{u} : {self.filtration[u]}' for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size, ax=ax)
        else:
          nx.draw(self.G, self.pos, labels = {u:f'V{u}' for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size, ax=ax)
      else:
        if modified:
          mf = self.get_modified_filtration()
          nx.draw(self.G, self.pos, labels={u:mf[u] for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size, ax=ax)
        elif with_filtration:
          nx.draw(self.G, self.pos, labels={u:self.filtration[u] for u in self.G.nodes()}, with_labels = True, node_size=node_size, font_size=font_size, ax=ax)
        else:
          nx.draw(self.G, self.pos, with_labels = False, node_size=node_size, font_size=font_size, ax=ax)