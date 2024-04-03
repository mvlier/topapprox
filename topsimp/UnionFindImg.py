import numpy as np
from topsimp.link_reduce import link_reduce


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
