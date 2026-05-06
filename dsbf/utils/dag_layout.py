from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch


def topo_sort_levels(G: nx.DiGraph) -> tuple[dict[int, list[str]], dict[str, int]]:
    """
    Perform a topological sort of the DAG and group nodes into hierarchical levels.

    Args:
        G (nx.DiGraph): Directed acyclic graph.

    Returns:
        Tuple containing:
            - levels: A dictionary mapping level index to list of nodes at that level.
            - node_levels: A dictionary mapping each node to its level.
    """
    sorted_nodes = list(nx.topological_sort(G))
    levels: dict[int, list[str]] = defaultdict(list)
    node_levels: dict[str, int] = {}

    for node in sorted_nodes:
        level = 0
        for pred in G.predecessors(node):
            level = max(level, node_levels[pred] + 1)
        node_levels[node] = level
        levels[level].append(node)

    return levels, node_levels


def assign_waterfall_positions(
    levels: dict[int, list[str]],
) -> dict[str, tuple[int, int]]:
    """
    Assign (x, y) positions for visualization:
    - x increases by level (left to right)
    - y assigns siblings top-down

    Args:
        levels (Dict[int, List[str]]): Mapping of levels to their nodes.

    Returns:
        Dictionary of positions {node: (x, y)}
    """
    pos: dict[str, tuple[int, int]] = {}
    for level, nodes in levels.items():
        for i, node in enumerate(nodes):
            pos[node] = (level, -i)  # x = level, y = sibling order
    return pos


def draw_dag(
    G: nx.DiGraph,
    pos: dict[str, tuple[int, int]],
    status: dict[str, str] | None = None,
    figsize: tuple[int, int] = (10, 6),
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """
    Draw a DAG using assigned positions and optional task status coloring.

    Args:
        G (nx.DiGraph): Directed acyclic graph.
        pos (Dict[str, Tuple[int, int]]): Mapping from node to (x, y) positions.
        status (Optional[Dict[str, str]]): Mapping of node to status
            ('success', 'failed', 'pending').
        figsize (Tuple[int, int]): Size of the matplotlib figure.
        title (Optional[str]): Optional plot title.
        save_path (Optional[str]): Optional file path to save the figure.
    """
    status = status or {}
    color_map = {
        "success": "#2ca02c",  # green
        "failed": "#d62728",  # red
        "pending": "#ff7f0e",  # orange
        None: "#d3d3d3",  # light gray
    }
    node_colors = [color_map.get(status.get(n), color_map[None]) for n in G.nodes]

    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        edgecolors="black",
        arrows=True,
        arrowsize=20,
        font_size=10,
    )

    if title:
        plt.title(title)

    legend_items = [Patch(color=c, label=s) for s, c in color_map.items() if s]
    plt.legend(handles=legend_items, loc="lower left")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
