import numpy as np
import networkx as nx
import itertools

## optimised link_reduce
from numba import njit,f8,i8
from numba.typed import List

#'Tuple((f8[:],f8[:,:]))(f8[:],i8[:,:],f8)'
@njit(parallel=True, fastmath=True) # cache=True
def _link_reduce(birth, edges, epsilon, compute_persistence=True):
    persistence = List()
    if compute_persistence:
      persistence.append((birth.min(),np.inf))
    modified = birth.copy()
    parent = np.arange(modified.shape[0])
    children = dict()
    for i in range(edges.shape[0]):
      ui,vi = edges[i,0],edges[i,1]
      # # parents of u and v
      up = parent[ui]
      vp = parent[vi]
      death = max(birth[ui],birth[vi])
      if up != vp: # if their bosses are different (i.e., they belong to different families)
        if birth[up] < birth[vp]:
          up,vp = vp,up   # up is the younger
        if birth[up] < death: # one of families is killed to produce a cycle
          if compute_persistence:
            persistence.append((birth[up],death))
          val=death
        else:
          val=np.inf


        if up in list(children.keys()):
          region = children[up].copy() # .pop(up) does not work...
          del children[up]
        else:
          region = np.array([up])
        #region = (parent == parent[up]) #Old version
        if abs(birth[up]-val)<epsilon:
          modified[region]=val
        parent[region] = vp
        if vp in list(children.keys()):
          children[vp] = np.append(children[vp], region) # can we skip creating a new array?
        else:
          children[vp] = np.append([vp], region)
    return(modified,persistence)



## UFI iterating over edges instead of vertices
class UnionFindImg():
  def __init__(self, img, dual=False, use_numba=True):
      self.shape = img.shape
      self.modified = None
      self.dual = dual # dualize or not
      self.use_numba = use_numba
      self.persistence = None

      # create graph
      if not dual:
          n,m = img.shape
          self.parent = np.arange(img.size) # initially, every vertex is its own parent
          self.birth = img.ravel().copy() # filtration value for each vertex
          E = [((i,j),(i+1,j)) for i in range(n-1) for j in range(m)]+[((i,j),(i,j+1)) for j in range(m-1) for i in range(n)] #Edges for the grid case
          img_extended = img
      else:
          n,m = img.shape
          self.parent = np.arange((n+2)*(m+2)) # initially, every vertex is its own parent
          min_val = -img.max()-1 # this value serves as -infty
          img_extended = np.full((n+2,m+2),min_val)
          img_extended[1:-1,1:-1] = -img # embed the negative of original image with a "frame" filled with -infty
          self.birth = img_extended.ravel().copy()
          self.shape = img_extended.shape
          E = [((i,j),(i+1,j)) for i in range(n+1) for j in range(m+2)]+[((i,j),(i,j+1)) for j in range(m+1) for i in range(n+2)] #Edges for the grid case
          E+= [((i,j),(i+1,j+1)) for i in range(n+1) for j in range(m+1)]+[((i,j),(i-1,j+1)) for i in range(1,n+2) for j in range(m+1)]
          #edges = np.empty((n+2)*(m+1)+(n+1)*(m+2)+2*(n+1)*(m+1),dtype=object)
          #edges[:]=E
      birth_edges = np.max([(img_extended[a[0]],img_extended[a[1]]) for a in E],axis=1) #Birth value for each edge
      self.edges = (np.sort([(np.ravel_multi_index(u, self.shape),np.ravel_multi_index(v, self.shape)) for u,v in E],axis=1)[:,::-1])[birth_edges.argsort()] #Edges already sorted by birth value


  # returns the modifed filtration values
  def get_f(self):
      modified = self.modified.reshape(self.shape)
      if(self.dual):
        return(-modified[1:-1,1:-1])
      return(modified)

  # returns the persistence if it was computed, otherwise returns None
  def get_presistence(self):
    return self.persistence

  # merge families and modify the filtration values.
  # (set epsilon=0 for leaving filtration values intact)
  def link(self, u, v, val=np.inf, epsilon=0):
      if u==v:
        return
      # apply the "elder rule" to determine who is the parent
      if self.birth[u] >= self.birth[v]:
        if abs(self.birth[u]-val)<epsilon:
          self.modified[self.parent == self.parent[u]]=val
        self.parent[self.parent == self.parent[u]]=v
      elif self.birth[u] < self.birth[v]:
        if abs(self.birth[v]-val)<epsilon:
          self.modified[self.parent == self.parent[v]]=val
        self.parent[self.parent == self.parent[v]]=u
      #print('link:',self.parent)##########################################################

  # look at an edge and merge end points if necessary
  def reduce(self,edge):
      i,j = edge
      #print(edge)##########################################################
      #order the vertices i,j of the edge by birth value
      if(self.birth[i]<self.birth[j]):
        _=i
        i=j
        j=_
      u = self.parent[i]
      v = self.parent[j]
      if u != v: # if their bosses are different (i.e., they belong to different families)
        birth = max(self.birth[u],self.birth[v]) # choose the younger one
        if birth < self.birth[i]: # one of families is killed to produce a cycle
          #x,y = np.unravel_index(i, shape=self.shape)
          self.persistence.append((birth,self.birth[i])) # (birth,death)
          if self.birth[u] == birth:
            self.component.append([j for j in range(len(self.birth)) if self.parent[j]==u])
          else:
            self.component.append([j for j in range(len(self.birth)) if self.parent[j]==v])
          self.link(u,v,self.birth[i],self.epsilon)
        else:
          self.link(u,v)

  # compute PH and store it to `self.persistence`
  # at the same time, smooth the filtration values and store it to `self.modified`
  def compute(self, epsilon=0, compute_persistence=True, verbose=False):
      self.component = ['permanent_component'] # self.component[i] has the list of vertices that merged into another component and created self.persitence[i]
      self.epsilon = epsilon # threshold for elimination of small bump
      wrapper = tqdm if verbose else lambda x:x # for showing progress-bar

      #iterate over the edges, which are already ordered by birth value
      if self.use_numba:
        self.modified, self.persistence = _link_reduce(self.birth, self.edges, self.epsilon, compute_persistence)
      else:
        self.parent = np.arange(self.shape[0]*self.shape[1]) # stores the parent (boss) of each vertex
        self.modified = self.birth.copy() # stores the modified (smoothed) function value
        self.persistence = [(self.birth.min(),float('inf'))] # list of cycles in the form (birth,death). Initially, the permanent cycle is added.
        for edge in wrapper(self.edges):
          #self.reduce((np.ravel_multi_index(edge[0], self.shape),np.ravel_multi_index(edge[1], self.shape)))
          self.reduce(edge)
      self.persistence=np.array(self.persistence)
      return(self.get_f())

  #This can smooth the image without having to compute PH again
  def smooth_image(self, epsilon):
    self.modified = self.birth.copy()
    for i in range(len(self.persistence)):
      x = self.persistence[i]
      if (x[1]-x[0]) < epsilon:
        for j in self.component[i]:
          self.modified[j] = x[1]
    return(self.get_f())





#Class for graph input
class UnionFindGraph():
  def __init__(self, input=None, use_numba=True):
    self.modified = None
    self.pos = None #position for drawing graph
    self.diagram = [np.array([None]),np.array([None])] #Persistence diagram, with [PH0, PH1]
    self.use_numba=use_numba
    if isinstance(input,np.ndarray):
      self.from_array(input)
    elif input is not None:
      raise ValueError(f"Unknown initialisation type")

  def from_faces(self, F: list, filtration: dict, *, boundary_vertices=None, verbose=True):
    # given face data F, create the graph and its dual
    self.G = nx.Graph()
    #given faces F, obtain all Edges E and boundary vertices
    self.faces = F
    facePerEdge = {} #{edge: number of incident faces}
    for f in F:
        for i in range(len(f)):
            edge = tuple(sorted([f[i],f[i-1]]))
            facePerEdge[edge] = facePerEdge.get(edge, 0) + 1
    self.E = list(facePerEdge.keys())
    self.G.add_edges_from(self.E)
    if verbose or (boundary_vertices is None):
        if max(facePerEdge.values()) > 2:
            problematicEdges = [k for k, v in facePerEdge.items() if v > 2]
            raise ValueError(f"Not embeddable in the plane, at least one edge belongs to more than two faces. Problematic edges: {problematicEdges}")
        boundary = [edge for edge, v in facePerEdge.items() if v == 1]
        boundary = set(list(itertools.chain.from_iterable(boundary)))
        if (boundary_vertices is not None) and (set(boundary_vertices) != boundary):
            print("detected boundary:", boundary)
            print("specified boundary:", boundary_vertices)
            raise ValueError("specified boundary_vertices are wrong!")
    else:
        boundary = set(boundary_vertices)
    self.boundary = boundary.copy()
    self.filtration = filtration.copy()
    # check if all vertices are given filtration values
    if set(self.G)!=set(filtration.keys()):
        raise ValueError("give filtration values for all the vertices!")
    # create dual edges
    dualE = []
    for f in F:
        for i in range(2, len(f) - 1):
            edge = tuple(sorted([f[0],f[i]]))
            dualE.append(edge)
        for i in range(1, len(f) - 2):
            for j in range(i+2, len(f)):
                edge = tuple(sorted([f[i],f[j]]))
                dualE.append(edge)
    self.dualE = list(set(dualE))
    self.INF_V = ["infty"] # vertex at infinity for one-point compactification
    self.dualE.extend([(self.INF_V[0],u) for u in self.boundary]) # boundary vertices are connected to the infinity
    self.infinite_filtration_value = max(filtration.values())+1
    self.filtration[self.INF_V[0]] = self.infinite_filtration_value
    # sort edges by the birth value
    filtration_E_max = [max(self.filtration[u],self.filtration[v]) for u,v in self.E]
    #filtration_E_min = [min(self.filtration[u],self.filtration[v]) for u,v in self.E]
    #self.E = [self.E[i] for i in np.lexsort((filtration_E_min,filtration_E_max))]
    self.E = [self.E[i] for i in np.argsort(filtration_E_max)]
    filtration_dualE_max = [max(-self.filtration[u],-self.filtration[v]) for u,v in self.dualE]
    self.dualE = [self.dualE[i] for i in np.argsort(filtration_dualE_max)]

  ## generalisation to include non-gfaces
  def from_faces_nonfaces(self, F: list, nF: list, filtration: dict, erbose=True):
    # given face data F, create the graph and its dual
    self.G = nx.Graph()
    #given faces F, obtain all Edges E and boundary vertices
    self.faces=F
    self.E = list(set(sum([ [frozenset((f[-1],f[0]))]+[frozenset((f[i],f[i+1])) for i in range(len(f)-1)] for f in F+nF],[])))
    self.G.add_edges_from(self.E)
    self.filtration = filtration.copy()
    # check if all vertices are given filtration values
    if set(self.G)!=set(filtration.keys()):
      raise ValueError("give filtration values for all the vertices!")
    # create dual edges
    dualE = sum([[frozenset((f[i],f[j])) for i in range(len(f)-1) for j in range(i+1,len(f))] for f in F],[])
    self.dualE = list(set(dualE))
    self.INF_V=[]
    self.infinite_filtration_value = max(filtration.values())+1
    for i,f in enumerate(nF):
      self.INF_V.append(f"infty{i}") # vertex at infinity
      self.filtration[f"infty{i}"] = self.infinite_filtration_value
      self.dualE.extend([frozenset((f[j],f"infty{i}")) for j in range(len(f))])
    # sort edges by the birth value
    filtration_E_max = [max(self.filtration[u],self.filtration[v]) for u,v in self.E]
    self.E = [self.E[i] for i in np.argsort(filtration_E_max)]
    filtration_dualE_max = [max(-self.filtration[u],-self.filtration[v]) for u,v in self.dualE]
    self.dualE = [self.dualE[i] for i in np.argsort(filtration_dualE_max)]

  ## create a graph with faces from image
  def from_array(self, img: np.array):
    self.shape = img.shape
    if len(img.shape)==2: # 2D case
      n,m = img.shape
      F = [[(i,j),(i+1,j),(i+1,j+1),(i,j+1)] for i in range(n-1) for j in range(m-1)]
    elif len(img.shape)==1: # 1D case
      n,m = img.shape[0],1
      F = [[(i,0),(i+1,0)] for i in range(n-1)]
    else:
      raise ValueError("array dimension should be 1 or 2!")
    boundary_vertices =  [(0,j) for j in range(m)]+[(n-1,j) for j in range(m)]
    boundary_vertices += [(i,0) for i in range(1,n-1)]+[(i,m-1) for i in range(1,n-1)]
    filtration = {(i,j): img.reshape((n,-1))[i,j] for i in range(n) for j in range(m)}
    self.from_faces(F, filtration, boundary_vertices=boundary_vertices)
    self.n = n
    self.m = m

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

  ## look at an edge and merge end points if necessary
  def reduce(self,ui,vi):
      # parents of u and v
      up = self.parent[ui] # self.find(ui)
      vp = self.parent[vi] # self.find(vi)
      #print(up,vp,self.birth[up], self.birth[vp])
      # if self.birth[ui] > self.birth[vi]:
      #   killing = ui
      # elif (self.birth[ui] == self.birth[vi]) and ui>vi:
      #   killing = ui # break tie consistently
      # else:
      #   killing = vi
      # death = self.birth[killing] # filtration value of the added edge
      death = max(self.birth[ui],self.birth[vi])
      if up != vp: # if their bosses are different (i.e., they belong to different families)
        killed = up if self.birth[up] > self.birth[vp] else vp
        #print(self.idx2node[killed])
        birth = self.birth[killed] # choose the younger one
        if birth < death: # one of families is killed to produce a cycle
          self.persistence.append((birth,death)) # (birth,death)
          #print(self.persistence[-1])
          self.link(up,vp,death,self.epsilon)
        else:
          self.link(up,vp)

  ## compute PH along with the smoothed filtration values
  def compute(self, epsilon=0, dual=False, verbose=False):
    self.epsilon=epsilon
    ## mapping from node index to node name
    self.idx2node = list(self.G.nodes())
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

    if self.use_numba:
      ## TODO: still slower with numba
      E = np.sort([(self.node2idx[u],self.node2idx[v]) for u,v in E],axis=1)[:,::-1] # index list
      self.modified, self.persistence = _link_reduce(self.modified, E, self.epsilon)
    else:
      self.birth = self.modified.copy()
      self.parent = np.arange(len(self.birth))
      self.persistence = [(self.birth.min(),float('inf'))] # list of cycles in the form (birth,death). Initially, the permanent cycle is added.
      # iterating over edges
      for u,v in E:
        ui = self.node2idx[u] # u,v denote the node names, and ui,vi are their indices
        vi = self.node2idx[v]
        if ui<vi: # respect vertex order
          ui,vi = vi,ui
        #print("edge ",(u,v),max(self.birth[ui],self.birth[vi]))
        self.reduce(ui,vi)

    if dual:
      self.modified = -self.modified[:-len(self.INF_V)] # remove the infinity point
      self.persistence = self.persistence[1:] # remove the permanent cycle since Alexander duality looks at the reduced homology
      self.idx2node = self.idx2node[:-len(self.INF_V)]
    self.persistence = np.where(np.array(self.persistence) > -self.infinite_filtration_value, self.persistence, -np.inf)
    return(self.persistence)

  #For Drawing

  def reorient(self):
    k=0
    List=list(range(len(self.faces)-1))
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


  def pos_from_faces(self):

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
      while not faces == []:
        face = [f for f in faces if nb2 in f]
        face = face[0]
        nb1 = nb2
        idx = face.index(node)
        nb2 = next_nbh(face)
        add_edge(node,nb2,nb1)
        faces.pop(faces.index(face))

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
        face = [f for f in faces if nb2 in f][0]
        idx = face.index(node)
        nb1 = nb2
        nb2 = face[idx-len(face)+1]
        add_edge(node, nb2, nb1)
        faces.pop(faces.index(face))
    try:
      PE.check_structure()
      pos = nx.combinatorial_embedding_to_pos(PE)
    except:
      print("NOT GOOD")
      return(None)
    return(pos)  
    
  def draw(self, *, with_filtration=False, with_labels=False, modified=False, ax=None, node_size=600, font_size=8):
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