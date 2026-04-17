"""Graph-with-faces data model."""

from __future__ import annotations

import numpy as np


class GraphWithFaces:
    """Embedded graph with optional dual-graph construction."""

    def __init__(
        self,
        F=None,
        H=None,
        E=None,
        signal=None,
        *,
        compute: str = "normal",
        is_triangulated: bool = False,
    ) -> None:
        if signal is None:
            raise ValueError("signal must be provided.")
        if compute not in {"normal", "dual", "both"}:
            raise ValueError(f"compute must be 'normal', 'dual', or 'both', got {compute!r}.")

        self.F = [list(face) for face in ([] if F is None else F)]
        self.H = [list(hole) for hole in ([] if H is None else H)]
        self.compute = compute
        self.is_triangulated = bool(is_triangulated)

        self.signal = np.asarray(signal, dtype=float).ravel()
        self.n_nodes = int(self.signal.shape[0])
        self.vertex_count = self.n_nodes

        self.edges = self._edge_array(E) if E is not None else np.empty((0, 2), dtype=np.int64)
        self.dual_edges = np.empty((0, 2), dtype=np.int64)
        self.edge_signal = np.empty(0, dtype=float)
        self.dual_edge_signal = np.empty(0, dtype=float)

        self._extend_signal()
        self._build_graphs()

    @property
    def E(self) -> np.ndarray:
        return self.edges

    @property
    def dualE(self) -> np.ndarray:
        return self.dual_edges

    @property
    def E_signal(self) -> np.ndarray:
        return self.edge_signal

    @property
    def dualE_signal(self) -> np.ndarray:
        return self.dual_edge_signal

    @staticmethod
    def _edge_array(edges) -> np.ndarray:
        if edges is None:
            return np.empty((0, 2), dtype=np.int64)
        edge_array = np.asarray(edges, dtype=np.int64)
        if edge_array.size == 0:
            return edge_array.reshape(0, 2)
        if edge_array.ndim != 2 or edge_array.shape[1] != 2:
            raise ValueError("Expected an edge array of shape (n, 2).")
        edge_array = np.sort(edge_array, axis=1)
        return np.unique(edge_array, axis=0)

    def _extend_signal(self) -> None:
        if self.is_triangulated:
            hole_values = np.full(len(self.H), np.inf, dtype=self.signal.dtype)
            self.signal = np.concatenate((self.signal, hole_values))
            return

        face_values = np.array(
            [np.max(self.signal[np.asarray(face, dtype=np.int64)]) for face in self.F],
            dtype=self.signal.dtype,
        )
        hole_values = np.full(len(self.H), np.inf, dtype=self.signal.dtype)
        self.signal = np.concatenate((self.signal, face_values, hole_values))

    def _build_graphs(self) -> None:
        primal_edges = {tuple(edge) for edge in self.edges.tolist()}
        dual_edges: set[tuple[int, int]] = set()

        if primal_edges:
            self.vertex_count = max(self.vertex_count, int(np.max(self.edges)) + 1)

        if not primal_edges:
            for face in self.F:
                primal_edges.update(self._cycle_edges(face))
            for hole in self.H:
                primal_edges.update(self._cycle_edges(hole))

        if self.compute in {"dual", "both"}:
            if self.is_triangulated:
                dual_edges.update(primal_edges)
                for hole in self.H:
                    dual_edges.update(self._dual_star_edges(hole))
            else:
                dual_edges.update(primal_edges)
                for polygon in [*self.F, *self.H]:
                    dual_edges.update(self._dual_star_edges(polygon))

        self.edges, self.edge_signal = self._sort_edges(primal_edges, negate=False)

        if self.compute in {"dual", "both"}:
            self.dual_edges, self.dual_edge_signal = self._sort_edges(dual_edges, negate=True)

    def _cycle_edges(self, polygon: list[int]) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            if start == end:
                continue
            edges.add((min(start, end), max(start, end)))
        return edges

    def _dual_star_edges(self, polygon: list[int]) -> set[tuple[int, int]]:
        center = self.vertex_count
        self.vertex_count += 1
        return {(min(vertex, center), max(vertex, center)) for vertex in polygon}

    def _sort_edges(self, edges: set[tuple[int, int]], *, negate: bool) -> tuple[np.ndarray, np.ndarray]:
        edge_array = self._edge_array(list(edges))
        if edge_array.shape[0] == 0:
            return edge_array, np.empty(0, dtype=self.signal.dtype)

        values = -self.signal if negate else self.signal
        edge_signal = np.maximum(values[edge_array[:, 0]], values[edge_array[:, 1]])
        order = np.argsort(edge_signal, kind="stable")
        return edge_array[order], edge_signal[order]

    def __repr__(self) -> str:
        return (
            "GraphWithFaces("
            f"n_nodes={self.n_nodes}, n_faces={len(self.F)}, n_holes={len(self.H)}, "
            f"compute={self.compute!r}, is_triangulated={self.is_triangulated})"
        )

    def draw(
        self,
        signal=None,
        pos=None,
        cmap="viridis",
        figsize=(6, 6),
        face_alpha=0.4,
        node_edgecolors="black",
        colorbar=True,
        edge_width=2.5,
        vmin=None,
        vmax=None,
        edge_boundary_color=None,
        linewidth=1.5,
        threshold=None,
        ax=None,
        gray_faces=False,
    ):
        """Plot the primal graph with optional face fills."""

        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import networkx as nx

        if signal is None:
            signal = self.signal[: self.n_nodes]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        else:
            ax.clear()

        graph = nx.Graph()
        graph.add_edges_from(self.edges.tolist())
        if pos is None:
            pos = nx.spring_layout(graph, seed=42)

        edge_signal = np.array([max(signal[u], signal[v]) for u, v in self.edges])
        face_signal = np.array([max(signal[v] for v in face) for face in self.F]) if self.F else np.empty(0)
        hole_signal = np.array([max(signal[v] for v in hole) for hole in self.H]) if self.H else np.empty(0)

        if vmin is None:
            candidates = [signal.min()]
            if edge_signal.size:
                candidates.append(edge_signal.min())
            if face_signal.size:
                candidates.append(face_signal.min())
            vmin = min(candidates)
        if vmax is None:
            candidates = [signal.max()]
            if edge_signal.size:
                candidates.append(edge_signal.max())
            if face_signal.size:
                candidates.append(face_signal.max())
            vmax = max(candidates)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_func = plt.cm.get_cmap(cmap)

        def transparent(color):
            red, green, blue, _ = mcolors.to_rgba(color)
            return (red, green, blue, 0.0)

        for index, hole in enumerate(self.H):
            polygon = np.array([pos[vertex] for vertex in hole])
            alpha = 0.0 if threshold is not None and hole_signal[index] >= threshold else face_alpha
            edge_color = "none" if alpha == 0.0 else "black"
            ax.fill(*zip(*polygon), color="none", alpha=alpha, edgecolor=edge_color, linewidth=linewidth)

        for index, face in enumerate(self.F):
            polygon = np.array([pos[vertex] for vertex in face])
            color = "lightgray" if gray_faces else cmap_func(norm(face_signal[index]))
            if threshold is not None and face_signal[index] >= threshold:
                color = transparent(color)
            edge_color = "none" if threshold is not None and face_signal[index] >= threshold else "black"
            alpha = 0.0 if edge_color == "none" else face_alpha
            ax.fill(*zip(*polygon), color=color, alpha=alpha, edgecolor=edge_color, linewidth=linewidth)

        for index, (start, end) in enumerate(self.edges):
            color = edge_boundary_color if edge_boundary_color is not None else cmap_func(norm(edge_signal[index]))
            if threshold is not None and edge_signal[index] >= threshold:
                color = transparent(color)
            ax.plot(
                [pos[start][0], pos[end][0]],
                [pos[start][1], pos[end][1]],
                color=color,
                linewidth=edge_width,
                solid_capstyle="round",
            )

        node_colors = [signal[node] for node in graph.nodes]
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap=cmap,
            node_size=300,
            edgecolors=node_edgecolors,
            vmin=vmin,
            vmax=vmax,
        )
        nx.draw_networkx_labels(graph, pos, ax=ax)
        ax.set_axis_off()
        ax.set_aspect("equal")
        if colorbar:
            plt.colorbar(nodes, ax=ax)
        return ax
