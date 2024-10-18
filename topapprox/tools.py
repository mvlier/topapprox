import random

class tools():
    '''Class for general tools for the topapprox package
    '''
    @staticmethod
    def gwf_generator(n_faces, *, face_size_range=[3, 12], n_holes=1, unique_faces=True):
        '''Generates a graph with faces and optional holes.

        This function generates a random planar graph structure with `n_faces` faces, 
        where each face is defined by a sequence of vertices. The size of each face is 
        randomly selected within the range `face_size_range`, and the graph may also 
        include a specified number of holes. The faces are built iteratively by modifying 
        the boundary of the current face set, adding new vertices in each step.

        Parameters:
        -----------
        n_faces : int
            The number of faces to be generated in the graph.
        
        face_size_range : list of int, optional
            A list of two integers [n, m] where `n` is the minimum and `m` is the maximum 
            number of vertices that can define a face. Defaults to [3, 12].
        
        n_holes : int, optional
            The number of holes to randomly remove from the generated faces. Each hole 
            is a face removed from the list of faces, representing a "hole" in the graph. 
            Defaults to 1.
        
        unique_faces : bool, optional
            If True, enforces that each generated face contains at least one new vertex 
            not present in previous faces. If False, faces can potentially reuse vertices 
            entirely from the existing boundary, leading to non-unique faces. Defaults to True.

        Returns:
        --------
        faces : list of lists
            A list of faces where each face is represented as a list of vertex indices.
        
        holes : list of lists
            A list of removed faces that represent holes in the graph. Each hole is a face 
            that was removed from the `faces` list.
        
        n_verts : int
            The total number of vertices generated for the graph.

        Notes:
        ------
        - The function begins by generating a single face as the initial boundary, 
        and then iteratively modifies the boundary by adding new segments and new 
        vertices, based on random selections.
        - The parameter `unique_faces` controls whether each face must include new vertices.
        - The final face in the sequence will always be the current boundary, closed as a loop.
        - After all faces are generated, the specified number of holes (`n_holes`) are 
        randomly selected and removed from the list of faces.

        Example:
        --------
        >>> faces, holes, n_verts = gwf_generator(5, face_size_range=[3, 8], n_holes=2)
        >>> print(faces)
        >>> print(holes)
        >>> print(n_verts)
        '''
        n, m = face_size_range[0], face_size_range[1]
        n_verts = random.randint(n,m)
        boundary = [i for i in range(n_verts)]
        faces = [boundary]
        holes = []
        lim = 1 if unique_faces else 0
        # print(f"Initial boundary: {boundary}")
        for i in range(n_faces-1):
            # print(f"\n ###########\nIteration {i}:\n")
            bound_len = len(boundary)
            idx0 = random.randint(0, bound_len-1)
            increment = random.randint(2, min(bound_len-1, m - lim))
            idx1 = idx0 + increment
            n_new_verts = random.randint(max(lim, n - increment), m - increment)
            new_segment = [i for i in range(n_verts, n_verts + n_new_verts)]
            # print(f"New_segmen = {new_segment}")
            face0 = tools._wrap_slice(boundary,idx0, idx1)
            # print(f"face0 = {face0}")
            face = face0 + new_segment
            # print(f"Face = {face}")
            n_verts += n_new_verts
            faces.append(face)
            new_segment.reverse()
            bound_cut = tools._wrap_slice(boundary, idx1-1, idx0+1)
            # print(f"bound_cut = {bound_cut}")
            boundary = bound_cut + new_segment
            # print(f"boundary = {boundary}")
            
        
        faces.append(boundary)

        for i in range(n_holes):
            idx = random.randint(0, len(faces)-1)
            holes.append(faces.pop(idx))

        return faces, holes, n_verts 
    
    @staticmethod
    def _wrap_slice(a, x, y):
        n = len(a)
        x = x%n
        y = y%n
        if x < y:
            return a[x:y]
        elif x >= y:
            return a[x:] + a[:y]