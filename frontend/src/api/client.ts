export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface Paper {
  rank: number;
  title: string;
  abstract: string;
  year: number;
  authors: string[];
  keyphrases: string[];
  score: number;
  text_similarity: number;
  gnn_activation: number;
  degree_centrality: number;
  arxiv_id: string;
}

export interface SearchResponse {
  results: Paper[];
  query_time_ms: number;
  model_used: string;
}

export interface SuggestResponse {
  retrieved_papers: Paper[];
  reasoning_paths: string[];
  suggestions: string;
  grounding_note: string;
}

export interface GraphStats {
  paper_nodes: number;
  concept_nodes: number;
  author_nodes: number;
  total_edges: number;
}

export interface GraphData {
  nodes: any[];
  edges: any[];
  stats: GraphStats;
}

export const apiClient = {
  async search(query: string, k: number = 10, use_gnn: boolean = true): Promise<SearchResponse> {
    const res = await fetch(`${API_URL}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k, use_gnn }),
    });
    if (!res.ok) throw new Error("Search failed");
    return res.json();
  },

  async suggest(project_description: string, k: number = 5): Promise<SuggestResponse> {
    const res = await fetch(`${API_URL}/api/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_description, k }),
    });
    if (!res.ok) throw new Error("Suggest failed");
    return res.json();
  },

  async getGraph(): Promise<GraphData> {
    const res = await fetch(`${API_URL}/api/graph`);
    if (!res.ok) throw new Error("Failed to fetch graph data");
    return res.json();
  },

  async getStats(): Promise<any> {
    const res = await fetch(`${API_URL}/api/stats`);
    if (!res.ok) throw new Error("Failed to fetch stats");
    return res.json();
  },
};
