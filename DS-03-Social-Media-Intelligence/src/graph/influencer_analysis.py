"""NetworkX graph analytics starter."""
import networkx as nx


def analyze_edges(edges):
    """Build a directed graph from (source, target) pairs and compute PageRank."""
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    if graph.number_of_nodes() == 0:
        return graph, {}
    return graph, nx.pagerank(graph)
