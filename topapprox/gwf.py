"""
Graph with faces class

This class is used to represent a graph with faces.
By passing a signal value the edges are automatically sorted accordingly.
One can compute the dual of the graph with faces by setting `compute` to "dual",
if it is set to "both", then both the normal and dual versions will be computed
"""


import numpy as np

class GraphWithFaces:
    def __init__(self, F=None, H=None, signal=None, compute="normal"):
        """
        Initializes the graph with faces. Faces (F) and holes (H) should be
        lists of lists, where each inner list represents a sequence of vertices
        that form a face or hole.

        :param F: List of faces, each face is a list of vertices.
        :param H: List of holes, each hole is a list of vertices.
        :param signal: A NumPy array of shape (n_vertices,) representing the function values.
        :param compute: A string, can be either "normal", "dual" or "both"
        """
        self.F = F if F is not None else []
        self.H = H if H is not None else []
        self.signal = signal
        self.E_signal = None
        self.E = set()  # Use set to track unique edges
        self.compute = compute
        self.vertex_count = signal.shape[0]
        if compute == "dual" or compute == "both":
            self.dualE = set()
            self.dualE_signal = None
            self.signal = np.concatenate((signal, np.array([max([signal[v] for v in f]) for f in F]), np.full(len(H), np.inf)))

        # Automatically determine edges from F and H if provided
        self._determine_E()

    def _determine_E(self):
        """
        Determines the edge set E based on the faces (F) and holes (H), and sorts it by signal values.
        """
        # Step 1: Add edges from faces and holes
        for face in self.F:
            self._add_edges(face)
        for i in range(len(self.H)-1): # We can skip the final face, since all its edges were already included
            self._add_edges(self.H[i])

        if self.compute == "normal" or self.compute == "both":
            # Convert set of edges to numpy array for efficient operations
            edges = np.array(list(self.E), dtype=np.uint32)
            # Compute the max signal value for each edge
            edge_signals = np.maximum(self.signal[edges[:, 0]], self.signal[edges[:, 1]])
            # Sort edges by signal values
            sorted_indices = np.argsort(edge_signals)
            self.E_signal = edge_signals[sorted_indices]
            self.E = edges[sorted_indices]

        if self.compute == "dual" or self.compute == "both":
            # Convert set of edges to numpy array for efficient operations
            dual_edges = np.array(list(self.dualE), dtype=np.uint32)
            # Compute the max signal value for each edge
            dual_edge_signals = np.maximum(-self.signal[dual_edges[:, 0]], -self.signal[dual_edges[:, 1]])
            # Sort edges by signal values
            dual_sorted_indices = np.argsort(dual_edge_signals)
            self.dualE_signal = dual_edge_signals[dual_sorted_indices]
            self.dualE = dual_edges[dual_sorted_indices]
        

    def _add_edges(self, sequence, *, ishole=False):
        """
        Adds edges for a given sequence of vertices, which can be a face or hole.
        :param sequence: A list of vertices.
        """
        n = len(sequence)
        if self.compute == "normal":
            for i in range(n):
                v1, v2 = sequence[i], sequence[(i + 1) % n]
                if v1 > v2:  # Manually check for edge direction to avoid sorting
                    v1, v2 = v2, v1
                self.E.add((v1, v2))  # Use set for fast duplicate detection
        elif self.compute == "dual":
            v3 = self.vertex_count
            self.vertex_count += 1
            for i in range(n):
                v1, v2 = sequence[i], sequence[(i + 1) % n]
                self.dualE.add((v1, v3))
                if v1 > v2:  # Manually check for edge direction to avoid sorting
                    v1, v2 = v2, v1
                self.dualE.add((v1, v2))
        elif self.compute == "both":
            v3 = self.vertex_count
            self.vertex_count += 1
            for i in range(n):
                v1, v2 = sequence[i], sequence[(i + 1) % n]
                self.dualE.add((v1, v3))
                if v1 > v2:  # Manually check for edge direction to avoid sorting
                    v1, v2 = v2, v1
                self.E.add((v1, v2))
                self.dualE.add((v1, v2))

