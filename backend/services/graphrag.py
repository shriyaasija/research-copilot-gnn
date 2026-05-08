"""
GraphRAG: Citation-graph reasoning path extraction and verbalization.
"""

import logging
import networkx as nx

logger = logging.getLogger(__name__)


def extract_reasoning_paths(
    G: nx.DiGraph,
    query_paper_nodes: list[str],
    retrieved_nodes: list[str],
    max_hops: int = 3,
    max_paths: int = 8,
) -> list[str]:
    """
    Extract and verbalize shortest citation graph paths
    connecting query-adjacent nodes to retrieved answer nodes.
    
    Returns verbalized path strings like:
    "Hamilton et al. 2017 → [cites] → Xu et al. 2019 → [mentions] → node_classification"
    """
    if G is None or not query_paper_nodes or not retrieved_nodes:
        return []

    G_undirected = G.to_undirected()

    query_adjacent = set()
    for qnode in query_paper_nodes:
        if G_undirected.has_node(qnode):
            query_adjacent.update(G_undirected.neighbors(qnode))
    query_adjacent.update(query_paper_nodes)

    raw_paths = []
    for retrieved in retrieved_nodes:
        if not G_undirected.has_node(retrieved):
            continue
        best_path = None
        best_length = float("inf")
        for qa_node in query_adjacent:
            if not G_undirected.has_node(qa_node):
                continue
            try:
                path = nx.shortest_path(G_undirected, source=qa_node, target=retrieved)
                if len(path) <= max_hops + 1 and len(path) < best_length:
                    best_path = path
                    best_length = len(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if best_path is not None:
            raw_paths.append(best_path)

    raw_paths.sort(key=len)

    verbalized = []
    seen = set()
    for path in raw_paths[:max_paths * 2]:
        verbal = _verbalize_path(G, path)
        key = " → ".join(path)
        if key not in seen and verbal:
            seen.add(key)
            verbalized.append(verbal)
        if len(verbalized) >= max_paths:
            break
    return verbalized


def _verbalize_path(G: nx.DiGraph, path: list[str]) -> str:
    if len(path) < 2:
        return ""
    parts = []
    for i, node_id in enumerate(path):
        parts.append(_get_node_label(G, node_id))
        if i < len(path) - 1:
            relation = _get_edge_relation(G, node_id, path[i + 1])
            parts.append(f"[{relation}]")
    return " → ".join(parts)


def _get_node_label(G: nx.DiGraph, node_id: str) -> str:
    if not G.has_node(node_id):
        return str(node_id)
    data = G.nodes[node_id]
    node_type = data.get("type", "unknown")
    if node_type == "unknown":
        nid = str(node_id)
        if nid.startswith("paper_"):
            node_type = "paper"
        elif nid.startswith("concept_") or nid.startswith("kp_"):
            node_type = "concept"
        elif nid.startswith("author_"):
            node_type = "author"

    if node_type == "paper":
        title = data.get("title", str(node_id))
        year = data.get("year", "")
        short = title[:42] + "..." if len(title) > 45 else title
        return f"{short} ({year})" if year else short
    elif node_type == "concept":
        label = data.get("label", data.get("name", str(node_id)))
        return str(label).replace("concept_", "").replace("kp_", "").replace("_", " ")
    elif node_type == "author":
        name = data.get("name", str(node_id))
        return str(name).replace("author_", "").replace("_", " ")
    return str(node_id)


def _get_edge_relation(G: nx.DiGraph, source: str, target: str) -> str:
    if G.has_edge(source, target):
        data = G.edges[source, target]
        return data.get("relation", data.get("type", "related"))
    if G.has_edge(target, source):
        data = G.edges[target, source]
        rel = data.get("relation", data.get("type", "related"))
        rev = {"cites": "cited_by", "cited_by": "cites", "wrote": "written_by",
               "written_by": "wrote", "mentions": "mentioned_in", "mentioned_in": "mentions"}
        return rev.get(rel, rel)
    return "related"


def get_query_adjacent_papers(retrieval_service, query: str, k: int = 3) -> list[str]:
    """Get the top-K paper nodes most similar to a query."""
    results = retrieval_service.search(query, k=k, use_gnn=False)
    paper_nodes = []
    for result in results.get("results", []):
        arxiv_id = result.get("arxiv_id", "")
        for i, paper in enumerate(retrieval_service.papers):
            if paper.get("arxiv_id") == arxiv_id:
                paper_nodes.append(f"paper_{i}")
                break
    return paper_nodes
