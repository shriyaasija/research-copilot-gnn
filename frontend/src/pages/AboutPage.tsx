import { useEffect, useState } from 'react';
import { ExternalLink, Activity, Link as LinkIcon } from 'lucide-react';
import { apiClient } from '../api/client';

export default function AboutPage() {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    apiClient.getStats().then(setStats).catch(console.error);
  }, []);

  return (
    <div className="flex-1 container mx-auto px-4 py-12 max-w-4xl">
      <div className="glass-panel rounded-3xl p-8 md:p-12">
        <h1 className="text-4xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
          About Research Copilot GNN
        </h1>
        
        <div className="prose prose-invert prose-lg mb-12">
          <p>
            Research Copilot GNN is a citation-graph-aware scientific paper retrieval and research assistant.
            It uses a pre-trained R-GCN (Relational Graph Convolutional Network) model to embed academic papers
            in a heterogeneous citation graph.
          </p>
          <p>
            When querying the system, it uses an ensemble scoring mechanism that combines:
            <br/>1. <strong>Semantic similarity</strong> (via sentence-transformers)
            <br/>2. <strong>GNN activation scores</strong> (graph-aware embeddings)
            <br/>3. <strong>Degree centrality</strong> (structural importance)
          </p>
          <p>
            The AI Copilot uses <strong>GraphRAG</strong> to extract and verbalize shortest citation graph paths 
            connecting query-adjacent nodes to retrieved answer nodes, allowing the LLM to generate highly grounded suggestions.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          <div className="bg-zinc-900/50 p-6 rounded-2xl border border-zinc-800">
            <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
              <Activity className="text-blue-400" />
              Dataset Statistics
            </h3>
            {stats ? (
              <ul className="space-y-3 text-zinc-300">
                <li className="flex justify-between"><span>Paper Nodes</span> <span className="font-mono">{stats.dataset.num_papers}</span></li>
                <li className="flex justify-between"><span>Total Graph Nodes</span> <span className="font-mono">{stats.dataset.graph_nodes}</span></li>
                <li className="flex justify-between"><span>Total Graph Edges</span> <span className="font-mono">{stats.dataset.graph_edges}</span></li>
                <li className="flex justify-between"><span>Node Types</span> <span className="font-mono">{stats.dataset.node_types}</span></li>
                <li className="flex justify-between"><span>Relation Types</span> <span className="font-mono">{stats.dataset.relation_types}</span></li>
              </ul>
            ) : (
              <div className="text-zinc-500">Loading stats...</div>
            )}
          </div>

          <div className="bg-zinc-900/50 p-6 rounded-2xl border border-zinc-800">
            <h3 className="text-xl font-bold mb-4">Project Links</h3>
            <div className="space-y-4">
              <a href="https://github.com/shriyaasija/research-copilot-gnn" target="_blank" rel="noopener noreferrer" 
                 className="flex items-center justify-between p-3 bg-zinc-800 rounded-lg hover:bg-zinc-700 transition-colors">
                <div className="flex items-center gap-3">
                  <LinkIcon />
                  <span>GitHub Repository</span>
                </div>
                <ExternalLink size={16} className="text-zinc-500" />
              </a>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-2xl font-bold mb-6">Model Performance Evaluation</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-400">
                  <th className="py-3 px-4 font-medium">Model</th>
                  <th className="py-3 px-4 font-medium">NDCG@5</th>
                  <th className="py-3 px-4 font-medium">NDCG@10</th>
                  <th className="py-3 px-4 font-medium">P@10</th>
                  <th className="py-3 px-4 font-medium">R@10</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50">
                {stats && Object.entries(stats.models).map(([name, metrics]: [string, any]) => (
                  <tr key={name} className="hover:bg-white/5 transition-colors">
                    <td className="py-3 px-4 font-medium text-blue-300">{name}</td>
                    <td className="py-3 px-4 font-mono">{metrics['NDCG@5'].toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono">{metrics['NDCG@10'].toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono">{metrics['P@10'].toFixed(4)}</td>
                    <td className="py-3 px-4 font-mono">{metrics['R@10'].toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
