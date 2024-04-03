def link_reduce(*args, **kwargs):
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
    return _link_reduce(*args, **kwargs)