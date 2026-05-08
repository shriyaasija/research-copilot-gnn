import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import { apiClient } from '../api/client';
import type { Paper } from '../api/client';
import { PaperCard } from '../components/PaperCard';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [useGnn, setUseGnn] = useState(true);
  const [k, setK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Paper[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [stats, setStats] = useState<{ time: number; model: string } | null>(null);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setHasSearched(true);
    try {
      const res = await apiClient.search(query, k, useGnn);
      setResults(res.results);
      setStats({ time: res.query_time_ms, model: res.model_used });
    } catch (err) {
      console.error(err);
      alert('Search failed. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col container mx-auto px-4 py-8 max-w-4xl">
      <motion.div 
        className={cn("flex flex-col transition-all duration-500", hasSearched ? "items-start mb-8" : "items-center justify-center flex-1 min-h-[60vh]")}
        layout
      >
        <motion.div layout className={cn("text-center mb-8", hasSearched ? "text-left" : "text-center")}>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4 text-transparent bg-clip-text bg-gradient-to-br from-white to-zinc-400">
            Graph-Powered Paper Discovery
          </h1>
          <p className="text-lg text-zinc-400 max-w-2xl mx-auto">
            Find relevant research using a hybrid approach of semantic text similarity and citation graph structural features.
          </p>
        </motion.div>

        <motion.form 
          layout
          onSubmit={handleSearch} 
          className="w-full max-w-3xl glass-panel p-2 rounded-2xl flex flex-col sm:flex-row gap-2 relative z-10"
        >
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-zinc-400" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe your research project or idea..."
              className="w-full bg-transparent border-none text-white px-12 py-4 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/50 text-lg placeholder:text-zinc-500"
            />
          </div>
          
          <div className="flex items-center gap-2 px-4 sm:border-l border-white/10">
            <select 
              value={k} 
              onChange={(e) => setK(Number(e.target.value))}
              className="bg-zinc-800 text-zinc-200 border border-zinc-700 rounded-lg px-2 py-2 focus:outline-none"
            >
              <option value={1}>Top 1</option>
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
              <option value={20}>Top 20</option>
            </select>

            <button
              type="button"
              onClick={() => setUseGnn(!useGnn)}
              className={cn(
                "px-3 py-2 rounded-lg text-sm font-medium transition-colors border",
                useGnn 
                  ? "bg-purple-500/20 text-purple-300 border-purple-500/30" 
                  : "bg-zinc-800 text-zinc-400 border-zinc-700"
              )}
            >
              GNN
            </button>
            
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center min-w-[100px]"
            >
              {loading ? <Loader2 className="animate-spin" size={20} /> : "Search"}
            </button>
          </div>
        </motion.form>
      </motion.div>

      {hasSearched && (
        <div className="flex-1 w-full max-w-3xl mx-auto">
          {stats && (
            <div className="flex items-center justify-between mb-6 text-sm text-zinc-500">
              <div>Found {results.length} results in {stats.time}ms</div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-blue-500"></div> Text Match
                </div>
                {useGnn && (
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-purple-500"></div> GNN Activation
                  </div>
                )}
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-teal-500"></div> Degree Centrality
                </div>
              </div>
            </div>
          )}

          <div className="space-y-4">
            {results.map((paper) => (
              <PaperCard key={paper.arxiv_id} paper={paper} />
            ))}
            
            {results.length === 0 && !loading && (
              <div className="text-center py-12 text-zinc-500 glass-panel rounded-xl">
                No papers found for this query.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
