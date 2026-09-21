"""Matplotlib helpers for inspecting basins and graphs.

``matplotlib`` and ``networkx`` are imported lazily, so the package itself only
depends on NumPy.

Basins are coloured by the **birth value** of their representative.  Parents are
born before their children, so a sequential colour map renders the hierarchy
directly: the colour gap between a basin and its parent is its persistence.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_basin_image", "plot_basin_graph", "draw_graph"]


def _colormap(cmap):
    import matplotlib as mpl

    return mpl.colormaps[cmap] if isinstance(cmap, str) else cmap


def _axes(ax, **kwargs):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax
    return plt.subplots(**kwargs)[1]


def plot_basin_image(image_filter, epsilon: float = 0.0, *, ax=None, cmap="turbo", **kwargs):
    """Show the basin map of an :class:`~topapprox.filters.ImageFilter` as an image."""
    ax = _axes(ax)
    kwargs.setdefault("origin", "upper")
    image = ax.imshow(image_filter.basin_map(epsilon), cmap=cmap, **kwargs)
    ax.set_title(rf"Basin map ($\varepsilon={epsilon}$)")
    return image


def plot_basin_graph(
    graph_filter,
    epsilon: float = 0.0,
    *,
    pos=None,
    ax=None,
    cmap="turbo",
    node_size: int = 300,
    with_labels: bool = False,
    edge_color="gray",
    colorbar: bool = True,
    **kwargs,
):
    """Colour the nodes of a :class:`~topapprox.filters.GraphFilter` by their basin."""
    import matplotlib.pyplot as plt
    import networkx as nx

    labels = np.asarray(graph_filter.basin_map(epsilon)).ravel()
    graph = _networkx_graph(graph_filter.graph, len(labels))
    pos = pos if pos is not None else nx.spring_layout(graph, seed=42)
    ax = _axes(ax)

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_color)
    nodes = nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=labels, cmap=cmap, node_size=node_size, **kwargs
    )
    if with_labels:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    ax.set_title(rf"Basin map ($\varepsilon={epsilon}$)")
    if colorbar:
        plt.colorbar(nodes, ax=ax, label="basin birth")
    return nodes


def _networkx_graph(gwf, n_vertices: int):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_vertices))
    edges = gwf.edges
    inside = (edges[:, 0] < n_vertices) & (edges[:, 1] < n_vertices)
    graph.add_edges_from(map(tuple, edges[inside].tolist()))
    return graph


def draw_graph(
    gwf,
    signal=None,
    *,
    pos=None,
    ax=None,
    cmap="viridis",
    figsize=(6, 6),
    face_alpha: float = 0.4,
    node_size: int = 300,
    node_edgecolors="black",
    edge_width: float = 2.5,
    linewidth: float = 1.5,
    edge_boundary_color=None,
    vmin=None,
    vmax=None,
    threshold=None,
    colorbar: bool = True,
    gray_faces: bool = False,
):
    """Draw a :class:`~topapprox.graph.GraphWithFaces` with its faces filled in.

    ``threshold`` hides every cell whose value reaches it, which shows the
    sub-level set of the signal.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import networkx as nx

    signal = gwf.signal if signal is None else np.asarray(signal, dtype=float).ravel()
    graph = _networkx_graph(gwf, gwf.n_vertices)
    pos = pos if pos is not None else nx.spring_layout(graph, seed=42)
    ax = _axes(ax, figsize=figsize)

    def cell_values(cells):
        return np.array([signal[np.asarray(cell, dtype=np.int64)].max() for cell in cells])

    edge_values = signal[gwf.edges].max(axis=1)
    face_values = cell_values(gwf.faces) if gwf.faces else np.empty(0)
    hole_values = cell_values(gwf.holes) if gwf.holes else np.empty(0)

    finite = np.concatenate((signal, edge_values, face_values))
    vmin = float(finite.min()) if vmin is None else vmin
    vmax = float(finite.max()) if vmax is None else vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = _colormap(cmap)

    def faded(color):
        red, green, blue, _ = mcolors.to_rgba(color)
        return (red, green, blue, 0.0)

    def hidden(value):
        return threshold is not None and value >= threshold

    for hole, value in zip(gwf.holes, hole_values):
        polygon = np.array([pos[vertex] for vertex in hole])
        alpha = 0.0 if hidden(value) else face_alpha
        ax.fill(
            *zip(*polygon),
            color="none",
            alpha=alpha,
            edgecolor="none" if alpha == 0.0 else "black",
            linewidth=linewidth,
        )

    for face, value in zip(gwf.faces, face_values):
        polygon = np.array([pos[vertex] for vertex in face])
        color = "lightgray" if gray_faces else colors(norm(value))
        ax.fill(
            *zip(*polygon),
            color=faded(color) if hidden(value) else color,
            alpha=0.0 if hidden(value) else face_alpha,
            edgecolor="none" if hidden(value) else "black",
            linewidth=linewidth,
        )

    for (start, end), value in zip(gwf.edges.tolist(), edge_values):
        color = edge_boundary_color if edge_boundary_color is not None else colors(norm(value))
        ax.plot(
            [pos[start][0], pos[end][0]],
            [pos[start][1], pos[end][1]],
            color=faded(color) if hidden(value) else color,
            linewidth=edge_width,
            solid_capstyle="round",
        )

    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=[signal[node] for node in graph.nodes],
        cmap=cmap,
        node_size=node_size,
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
