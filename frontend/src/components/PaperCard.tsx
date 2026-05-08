import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ExternalLink } from 'lucide-react';
import type { Paper } from '../api/client';

interface PaperCardProps {
  paper: Paper;
}

export function PaperCard({ paper }: PaperCardProps) {
  const [expanded, setExpanded] = useState(false);

  // Colors for score bars
  const totalScore = paper.score;
  const textSim = paper.text_similarity;
  const gnnAct = paper.gnn_activation;
  const centrality = paper.degree_centrality;

  // Compute widths based on weights (0.6, 0.25, 0.15)
  // Text (blue), GNN (purple), Centrality (teal)
  const textWidth = Math.max(0, textSim * 0.6 * 100);
  const gnnWidth = Math.max(0, gnnAct * 0.25 * 100);
  const centWidth = Math.max(0, centrality * 0.15 * 100);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="glass-panel rounded-xl p-5 mb-4 hover:border-blue-500/30 transition-colors"
    >
      <div 
        className="flex justify-between items-start cursor-pointer gap-4"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="bg-zinc-800 text-zinc-300 text-xs px-2 py-0.5 rounded font-mono">
              #{paper.rank}
            </span>
            <span className="text-zinc-400 text-sm">{paper.year}</span>
            {paper.arxiv_id && (
              <a 
                href={paper.arxiv_id.startsWith('http') ? paper.arxiv_id : `https://arxiv.org/abs/${paper.arxiv_id}`}
                target="_blank" 
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300 flex items-center gap-1 text-xs"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink size={12} />
                arXiv
              </a>
            )}
          </div>
          <h3 className="text-lg font-semibold text-zinc-100 leading-tight mb-2">
            {paper.title}
          </h3>
          <p className="text-sm text-zinc-400 mb-3 truncate max-w-2xl">
            {paper.authors.join(', ')}
          </p>

          {!expanded && (
            <p className="text-sm text-zinc-500 line-clamp-2">
              {paper.abstract}
            </p>
          )}
        </div>

        <div className="flex flex-col items-end min-w-[120px]">
          <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-br from-blue-400 to-purple-400">
            {(totalScore * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-zinc-500 mb-2">Match Score</div>
          
          {/* Score breakdown bar */}
          <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden flex" title={`Text: ${(textSim*100).toFixed(0)}% | GNN: ${(gnnAct*100).toFixed(0)}% | Centrality: ${(centrality*100).toFixed(0)}%`}>
            <div className="h-full bg-blue-500" style={{ width: `${textWidth}%` }} />
            <div className="h-full bg-purple-500" style={{ width: `${gnnWidth}%` }} />
            <div className="h-full bg-teal-500" style={{ width: `${centWidth}%` }} />
          </div>
        </div>
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="pt-4 mt-4 border-t border-white/5">
              <h4 className="text-sm font-medium text-zinc-300 mb-2">Abstract</h4>
              <p className="text-sm text-zinc-400 leading-relaxed mb-4">
                {paper.abstract}
              </p>
              
              {paper.keyphrases && paper.keyphrases.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-zinc-300 mb-2">Keyphrases</h4>
                  <div className="flex flex-wrap gap-2">
                    {paper.keyphrases.slice(0, 8).map((kp, i) => (
                      <span key={i} className="text-xs bg-zinc-800/80 text-zinc-300 px-2 py-1 rounded-md border border-zinc-700/50">
                        {kp}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
