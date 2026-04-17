"""
Visualization helpers for BHT basin maps.

Colour convention
-----------------
Each vertex is coloured by the **birth value** of its basin representative at
the chosen persistence threshold *epsilon*.  Because parents are born earlier
than their children, a sequential colour-map naturally assigns **stronger
(darker) colours to parents** and **lighter colours to children**.  The colour
gap between a parent basin and a child basin is proportional to the persistence
of the child, making the hierarchy visually apparent.
"""

import numpy as np


def plot_basin_image(tfi, epsilon=0, *, ax=None, cmap="turbo", **kwargs):
    """Show the basin map of a :class:`TopologicalFilterImage` as a 2-D image.

    Parameters
    ----------
    tfi : TopologicalFilterImage
        A filter-image instance (already constructed with an image).
    epsilon : float, optional
        Persistence threshold (default 0 shows all basins).
    ax : matplotlib Axes, optional
        Axes to draw on.  Created if *None*.
    cmap : str or Colormap, optional
        Colour-map name (default ``"turbo"``).
    **kwargs
        Forwarded to :func:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    import matplotlib.pyplot as plt

    labels = tfi.basin_map(epsilon)
    if ax is None:
        _, ax = plt.subplots()
    kwargs.setdefault("origin", "upper")
    im = ax.imshow(labels, cmap=cmap, **kwargs)
    ax.set_title(f"Basin map ($\\varepsilon={epsilon}$)")
    return im


def plot_basin_graph(
    tfg,
    epsilon=0,
    *,
    pos=None,
    ax=None,
    cmap="turbo",
    node_size=300,
    with_labels=False,
    edge_color="gray",
    **kwargs,
):
    """Colour graph nodes by their basin birth value.

    Parameters
    ----------
    tfg : TopologicalFilterGraph
        A filter-graph instance with ``compute_gwf`` already called.
    epsilon : float, optional
        Persistence threshold (default 0).
    pos : dict, optional
        Node positions ``{node: (x, y)}``.  If *None*, a spring layout is
        computed via NetworkX.
    ax : matplotlib Axes, optional
        Axes to draw on.  Created if *None*.
    cmap : str or Colormap, optional
        Colour-map name (default ``"turbo"``).
    node_size : int, optional
        Node size (default 300).
    with_labels : bool, optional
        Show vertex labels (default False).
    edge_color : str, optional
        Edge colour (default ``"gray"``).
    **kwargs
        Forwarded to :func:`networkx.draw_networkx_nodes`.

    Returns
    -------
    matplotlib.collections.PathCollection
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    labels = tfg.basin_map(epsilon)
    n = len(labels)

    # Build networkx graph from original edges
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if tfg.gwf is not None:
        for u, v in tfg.gwf.E:
            if u < n and v < n:
                G.add_edge(int(u), int(v))

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    if ax is None:
        _, ax = plt.subplots()

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_color)
    nc = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=labels,
        cmap=cmap,
        node_size=node_size,
        **kwargs,
    )
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_title(f"Basin map ($\\varepsilon={epsilon}$)")
    plt.colorbar(nc, ax=ax, label="basin birth")
    return nc
