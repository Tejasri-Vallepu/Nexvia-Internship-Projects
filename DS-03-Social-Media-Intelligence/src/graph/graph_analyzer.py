
import json
import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


INPUT_FILE = Path("data/raw/youtube/comments.json")
OUTPUT_DIR = Path("data/processed/graph")


def anonymize_channel_id(channel_id):
    """Create a stable pseudonymous ID from a channel ID."""

    return hashlib.sha256(
        str(channel_id).encode("utf-8")
    ).hexdigest()[:16]


def load_comments():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        comments = json.load(file)

    if not isinstance(comments, list):
        raise ValueError("Expected a list of comments in JSON.")

    return comments


def build_interaction_graph(comments):
    """Build directed author-to-parent-author reply relationships."""

    graph = nx.DiGraph()

    comment_author_map = {}
    author_comment_counts = Counter()

    # First pass: map comment IDs to anonymized author IDs.
    for comment in comments:
        author_id = comment.get("author_channel_id")
        comment_id = comment.get("comment_id")

        if not author_id or not comment_id:
            continue

        anonymous_id = anonymize_channel_id(author_id)

        comment_author_map[comment_id] = anonymous_id
        author_comment_counts[anonymous_id] += 1

        if not graph.has_node(anonymous_id):
            graph.add_node(anonymous_id)

    # Second pass: create reply edges.
    valid_reply_count = 0
    skipped_reply_count = 0

    for comment in comments:
        if comment.get("comment_type") != "reply":
            continue

        reply_author = comment.get("author_channel_id")
        parent_id = comment.get("parent_id")

        if not reply_author or not parent_id:
            skipped_reply_count += 1
            continue

        parent_author = comment_author_map.get(parent_id)

        if not parent_author:
            skipped_reply_count += 1
            continue

        reply_author_hash = anonymize_channel_id(reply_author)

        if graph.has_edge(reply_author_hash, parent_author):
            graph[reply_author_hash][parent_author]["weight"] += 1
        else:
            graph.add_edge(
                reply_author_hash,
                parent_author,
                weight=1
            )

        valid_reply_count += 1

    # Add comment counts as node attributes.
    for node in graph.nodes():
        graph.nodes[node]["comment_count"] = (
            author_comment_counts[node]
        )

    return graph, valid_reply_count, skipped_reply_count


def calculate_pagerank(graph):
    """Calculate PageRank scores."""

    if graph.number_of_nodes() == 0:
        return {}

    return nx.pagerank(
        graph,
        alpha=0.85,
        weight="weight"
    )


def detect_communities(graph):
    """Detect communities among authors with actual interactions."""

    undirected_graph = graph.to_undirected()

    # Exclude isolated authors from community detection.
    active_nodes = [
        node for node, degree in undirected_graph.degree()
        if degree > 0
    ]

    active_graph = undirected_graph.subgraph(active_nodes).copy()

    if active_graph.number_of_nodes() == 0:
        return []

    if active_graph.number_of_edges() == 0:
        return [{node} for node in active_graph.nodes()]

    return list(
        greedy_modularity_communities(
            active_graph,
            weight="weight"
        )
    )


def save_graph_results(
    graph,
    pagerank_scores,
    communities,
    valid_replies,
    skipped_replies
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save anonymized node information.
    node_rows = []

    for node, attributes in graph.nodes(data=True):
        node_rows.append({
            "anonymous_author_id": node,
            "comment_count": attributes.get("comment_count", 0),
            "in_degree": graph.in_degree(node),
            "out_degree": graph.out_degree(node)
        })

    pd.DataFrame(node_rows).to_csv(
        OUTPUT_DIR / "graph_nodes.csv",
        index=False
    )

    # Save anonymized directed edges.
    edge_rows = []

    for source, target, attributes in graph.edges(data=True):
        edge_rows.append({
            "reply_author": source,
            "parent_author": target,
            "interaction_count": attributes.get("weight", 1)
        })

    pd.DataFrame(edge_rows).to_csv(
        OUTPUT_DIR / "graph_edges.csv",
        index=False
    )

    # Save PageRank ranking.
    pagerank_rows = [
        {
            "anonymous_author_id": node,
            "pagerank_score": score
        }
        for node, score in pagerank_scores.items()
    ]

    pagerank_df = pd.DataFrame(pagerank_rows)

    if not pagerank_df.empty:
        pagerank_df = pagerank_df.sort_values(
            "pagerank_score",
            ascending=False
        )

    pagerank_df.to_csv(
        OUTPUT_DIR / "pagerank_scores.csv",
        index=False
    )

    # Save community assignments.
    community_rows = []
    assigned_nodes = set()

    for community_id, members in enumerate(communities):
        for member in members:
            community_rows.append({
                "anonymous_author_id": member,
                "community_id": community_id
            })
            assigned_nodes.add(member)

    # Isolated authors receive community_id = -1.
    for node in graph.nodes():
        if node not in assigned_nodes:
            community_rows.append({
                "anonymous_author_id": node,
                "community_id": -1
            })

    pd.DataFrame(community_rows).to_csv(
        OUTPUT_DIR / "community_assignments.csv",
        index=False
    )

    # Calculate modularity only on the active interaction network.
    undirected_graph = graph.to_undirected()

    active_nodes = [
        node for node, degree in undirected_graph.degree()
        if degree > 0
    ]

    active_graph = undirected_graph.subgraph(active_nodes).copy()

    if active_graph.number_of_edges() > 0 and communities:
        modularity = nx.community.modularity(
            active_graph,
            communities,
            weight="weight"
        )
    else:
        modularity = None

    active_author_count = active_graph.number_of_nodes()
    isolated_author_count = (
        graph.number_of_nodes() - active_author_count
    )

    summary = {
        "source": "YouTube Data API v3",
        "graph_type": "Directed reply interaction graph",
        "nodes": graph.number_of_nodes(),
        "active_authors": active_author_count,
        "isolated_authors": isolated_author_count,
        "directed_edges": graph.number_of_edges(),
        "valid_reply_relationships": valid_replies,
        "skipped_reply_relationships": skipped_replies,
        "communities_detected": len(communities),
        "average_degree": (
            sum(dict(graph.degree()).values())
            / graph.number_of_nodes()
            if graph.number_of_nodes() else 0
        ),
        "density": nx.density(graph),
        "modularity": modularity,
        "pagerank_top_10": (
            pagerank_df.head(10).to_dict(orient="records")
            if not pagerank_df.empty else []
        ),
        "outputs": [
            "graph_nodes.csv",
            "graph_edges.csv",
            "pagerank_scores.csv",
            "community_assignments.csv",
            "graph_report.json"
        ],
        "privacy": "Channel IDs pseudonymized using SHA-256."
    }

    with open(
        OUTPUT_DIR / "graph_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    return summary


def main():
    print("Loading YouTube comments...")

    comments = load_comments()

    print(f"Loaded {len(comments)} comments.")

    print("Building directed interaction graph...")

    graph, valid_replies, skipped_replies = (
        build_interaction_graph(comments)
    )

    print("Calculating PageRank...")

    pagerank_scores = calculate_pagerank(graph)

    print("Detecting communities...")

    communities = detect_communities(graph)

    print("Saving graph analytics results...")

    summary = save_graph_results(
        graph,
        pagerank_scores,
        communities,
        valid_replies,
        skipped_replies
    )

    print("\nGRAPH ANALYSIS COMPLETED!")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
