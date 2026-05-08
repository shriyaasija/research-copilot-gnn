import React, { useState } from 'react';
import { Sparkles, Loader2, Network } from 'lucide-react';
import { apiClient } from '../api/client';
import type { SuggestResponse } from '../api/client';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

export default function SuggestPage() {
  const [desc, setDesc] = useState('');
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SuggestResponse | null>(null);

  const handleSuggest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!desc.trim()) return;

    setLoading(true);
    try {
      const res = await apiClient.suggest(desc, k);
      setResult(res);
    } catch (err) {
      console.error(err);
      alert('Failed to generate suggestions.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 container mx-auto px-4 py-8 max-w-5xl flex flex-col md:flex-row gap-8">
      
      {/* Left Column: Input Form */}
      <div className="w-full md:w-1/3 flex flex-col gap-6">
        <div className="glass-panel p-6 rounded-2xl">
          <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
            <Sparkles className="text-yellow-400" />
            AI Copilot
          </h2>
          <p className="text-sm text-zinc-400 mb-6">
            Describe your research project in detail. The copilot will use GraphRAG to retrieve relevant papers and generate grounded suggestions for ablations, baselines, and datasets.
          </p>

          <form onSubmit={handleSuggest} className="flex flex-col gap-4">
            <div>
              <label className="block text-sm font-medium text-zinc-300 mb-2">
                Project Description
              </label>
              <textarea
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                placeholder="e.g., I am developing a new graph neural network architecture that uses hyperbolic embeddings for hierarchical classification..."
                className="w-full h-48 bg-zinc-900 border border-zinc-700 rounded-xl p-4 text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-none"
              />
            </div>
            
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <label className="text-sm text-zinc-400">Context Papers:</label>
                <select 
                  value={k} 
                  onChange={(e) => setK(Number(e.target.value))}
                  className="bg-zinc-800 text-zinc-200 border border-zinc-700 rounded-md px-2 py-1 text-sm focus:outline-none"
                >
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                </select>
              </div>

              <button
                type="submit"
                disabled={loading || !desc.trim()}
                className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white px-6 py-2 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-blue-900/20"
              >
                {loading ? <Loader2 className="animate-spin" size={18} /> : <Sparkles size={18} />}
                Generate
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Right Column: Results */}
      <div className="w-full md:w-2/3 flex flex-col gap-6">
        {loading ? (
          <div className="flex-1 glass-panel rounded-2xl p-8 flex flex-col items-center justify-center min-h-[500px]">
            <Loader2 className="animate-spin text-blue-500 mb-4" size={48} />
            <h3 className="text-xl font-semibold mb-2">Analyzing Graph Network...</h3>
            <p className="text-zinc-500 text-center max-w-sm typing-indicator">
              Traversing citation paths and synthesizing recommendations
            </p>
          </div>
        ) : result ? (
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex flex-col gap-6"
          >
            {/* LLM Output */}
            <div className="glass-panel p-8 rounded-2xl">
              <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
                <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-yellow-400 to-orange-400">
                  Research Suggestions
                </h3>
                <span className="text-xs bg-blue-500/20 text-blue-300 px-3 py-1 rounded-full border border-blue-500/30">
                  {result.grounding_note}
                </span>
              </div>
              
              <div className="prose prose-invert prose-blue max-w-none">
                <ReactMarkdown>{result.suggestions}</ReactMarkdown>
              </div>
            </div>

            {/* GraphRAG Paths */}
            {result.reasoning_paths.length > 0 && (
              <div className="glass-panel p-6 rounded-2xl">
                <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Network className="text-purple-400" size={20} />
                  Graph Reasoning Paths
                </h4>
                <div className="space-y-3">
                  {result.reasoning_paths.map((path, i) => (
                    <div key={i} className="text-sm bg-zinc-900/80 p-3 rounded-lg border border-zinc-800 text-zinc-300 font-mono">
                      {path.split('→').map((node, j, arr) => (
                        <React.Fragment key={j}>
                          <span className={node.includes('[') ? "text-purple-400" : "text-blue-300"}>
                            {node}
                          </span>
                          {j < arr.length - 1 && <span className="text-zinc-600 mx-1">→</span>}
                        </React.Fragment>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Retrieved Context Papers */}
            <div className="glass-panel p-6 rounded-2xl">
              <h4 className="text-lg font-semibold mb-4 text-zinc-300">Context Papers [{result.retrieved_papers.length}]</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {result.retrieved_papers.map((paper, i) => (
                  <div key={paper.arxiv_id} className="bg-zinc-900/50 p-4 rounded-xl border border-zinc-800">
                    <div className="text-xs text-blue-400 mb-1">[{i + 1}] {paper.year}</div>
                    <h5 className="font-medium text-sm mb-2 line-clamp-2">{paper.title}</h5>
                    <p className="text-xs text-zinc-500 line-clamp-1">{paper.authors.join(', ')}</p>
                  </div>
                ))}
              </div>
            </div>

          </motion.div>
        ) : (
          <div className="flex-1 border-2 border-dashed border-zinc-800 rounded-2xl flex items-center justify-center min-h-[500px]">
            <p className="text-zinc-600 text-center max-w-sm">
              Enter your project description on the left to generate AI-powered research suggestions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
