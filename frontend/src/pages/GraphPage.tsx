import React, { useEffect, useState, useRef, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import type { ForceGraphMethods } from 'react-force-graph-2d';
import { apiClient } from '../api/client';
import type { GraphData } from '../api/client';
import { Loader2, Maximize2, Filter } from 'lucide-react';
import { motion } from 'framer-motion';

export default function GraphPage() {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<any | null>(null);
  
  // Filters
  const [showPapers, setShowPapers] = useState(true);
  const [showConcepts, setShowConcepts] = useState(true);
  const [showAuthors, setShowAuthors] = useState(true);
  
  const graphRef = useRef<ForceGraphMethods | undefined>();

  useEffect(() => {
    async function loadGraph() {
      try {
        const graphData = await apiClient.getGraph();
        setData(graphData);
      } catch (err) {
        console.error("Failed to load graph", err);
      } finally {
        setLoading(false);
      }
    }
    loadGraph();
  }, []);

  const filteredData = React.useMemo(() => {
    if (!data) return { nodes: [], edges: [] };
    
    const allowedTypes = new Set();
    if (showPapers) allowedTypes.add('paper');
    if (showConcepts) allowedTypes.add('concept');
    if (showAuthors) allowedTypes.add('author');
    
    const nodes = data.nodes.filter(n => allowedTypes.has(n.type));
    const nodeIds = new Set(nodes.map(n => n.id));
    
    const edges = data.edges.filter(e => 
      nodeIds.has(typeof e.source === 'object' ? e.source.id : e.source) && 
      nodeIds.has(typeof e.target === 'object' ? e.target.id : e.target)
    );
    
    return { nodes, links: edges };
  }, [data, showPapers, showConcepts, showAuthors]);

  const handleNodeClick = useCallback(async (node: any) => {
    if (node.type === 'paper') {
      try {
        const paperDetails = await apiClient.search(node.title, 1, false);
        if (paperDetails.results.length > 0) {
           setSelectedNode({...node, ...paperDetails.results[0]});
        } else {
           setSelectedNode(node);
        }
      } catch(e) {
        setSelectedNode(node);
      }
    } else {
      setSelectedNode(node);
    }
    
    // Center node
    if (graphRef.current) {
      graphRef.current.centerAt(node.x, node.y, 1000);
      graphRef.current.zoom(4, 1000);
    }
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Loader2 className="animate-spin text-blue-500" size={48} />
      </div>
    );
  }

  if (!data) {
    return <div className="flex-1 flex items-center justify-center">Failed to load graph data.</div>;
  }

  const getNodeColor = (node: any) => {
    switch (node.type) {
      case 'paper': return '#3b82f6'; // blue-500
      case 'concept': return '#10b981'; // emerald-500
      case 'author': return '#f59e0b'; // amber-500
      default: return '#71717a';
    }
  };

  const getEdgeColor = (edge: any) => {
    switch (edge.relation) {
      case 'cites': return 'rgba(239, 68, 68, 0.4)'; // red
      case 'mentions': return 'rgba(20, 184, 166, 0.4)'; // teal
      case 'wrote': return 'rgba(113, 113, 122, 0.4)'; // zinc
      default: return 'rgba(255, 255, 255, 0.1)';
    }
  };

  return (
    <div className="flex-1 flex relative overflow-hidden bg-[#09090b]">
      {/* Graph Container */}
      <div className="flex-1 w-full h-full absolute inset-0">
        <ForceGraph2D
          ref={graphRef}
          graphData={filteredData}
          nodeLabel={(node: any) => node.title || node.label || node.name || node.id}
          nodeColor={getNodeColor}
          nodeRelSize={4}
          nodeVal={(node: any) => Math.sqrt(node.degree || 1)}
          linkColor={getEdgeColor}
          linkWidth={(link: any) => link.relation === 'cites' ? 1.5 : 0.5}
          linkDirectionalArrowLength={(link: any) => link.relation === 'cites' ? 3 : 0}
          linkDirectionalArrowRelPos={1}
          onNodeClick={handleNodeClick}
          backgroundColor="#09090b"
        />
      </div>

      {/* Overlay Controls */}
      <div className="absolute top-4 left-4 z-10 glass-panel p-4 rounded-xl flex flex-col gap-4">
        <h3 className="font-bold text-lg flex items-center gap-2">
          <Filter size={18} /> Filters
        </h3>
        <div className="flex flex-col gap-2 text-sm">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={showPapers} onChange={e => setShowPapers(e.target.checked)} className="accent-blue-500" />
            <span className="w-3 h-3 rounded-full bg-blue-500 inline-block" /> Papers ({data.stats.paper_nodes})
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={showConcepts} onChange={e => setShowConcepts(e.target.checked)} className="accent-emerald-500" />
            <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" /> Concepts ({data.stats.concept_nodes})
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={showAuthors} onChange={e => setShowAuthors(e.target.checked)} className="accent-amber-500" />
            <span className="w-3 h-3 rounded-full bg-amber-500 inline-block" /> Authors ({data.stats.author_nodes})
          </label>
        </div>
        <div className="text-xs text-zinc-500 pt-2 border-t border-white/10 mt-2">
          Total edges: {data.stats.total_edges}
        </div>
        <button 
          onClick={() => {
            if(graphRef.current) {
              graphRef.current.zoomToFit(400);
            }
          }}
          className="text-xs bg-white/5 hover:bg-white/10 py-1.5 rounded transition-colors flex items-center justify-center gap-1 mt-2"
        >
          <Maximize2 size={14} /> Zoom to Fit
        </button>
      </div>

      {/* Node Info Panel */}
      {selectedNode && (
        <motion.div 
          initial={{ x: 300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="absolute top-4 right-4 bottom-4 w-80 glass-panel rounded-xl flex flex-col overflow-hidden z-10 border border-white/10 shadow-2xl"
        >
          <div className="p-4 border-b border-white/10 flex justify-between items-center bg-black/40">
            <h3 className="font-bold text-lg capitalize">{selectedNode.type} Details</h3>
            <button onClick={() => setSelectedNode(null)} className="text-zinc-500 hover:text-white">&times;</button>
          </div>
          
          <div className="p-4 overflow-y-auto flex-1">
            <h4 className="text-xl font-semibold mb-2">
              {selectedNode.title || selectedNode.label || selectedNode.name}
            </h4>
            
            {selectedNode.type === 'paper' && (
              <div className="space-y-4 mt-4">
                {selectedNode.year && (
                  <div><span className="text-zinc-500 text-sm">Year</span> <p>{selectedNode.year}</p></div>
                )}
                {selectedNode.authors && (
                  <div><span className="text-zinc-500 text-sm">Authors</span> <p className="text-sm">{selectedNode.authors.join(', ')}</p></div>
                )}
                <div><span className="text-zinc-500 text-sm">Degree Centrality</span> <p>{selectedNode.degree}</p></div>
                {selectedNode.abstract && (
                  <div><span className="text-zinc-500 text-sm">Abstract</span> <p className="text-xs leading-relaxed text-zinc-300">{selectedNode.abstract}</p></div>
                )}
              </div>
            )}
            
            {selectedNode.type !== 'paper' && (
              <div className="space-y-4 mt-4">
                <div><span className="text-zinc-500 text-sm">ID</span> <p className="font-mono text-xs break-all">{selectedNode.id}</p></div>
                <div><span className="text-zinc-500 text-sm">Connections</span> <p>{selectedNode.degree}</p></div>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
}
