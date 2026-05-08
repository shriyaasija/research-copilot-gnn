"""
FastAPI REST API for Research Copilot GNN.
"""

import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.services.retrieval import RetrievalService
from backend.services.graphrag import extract_reasoning_paths, get_query_adjacent_papers
from backend.services.llm import LLMService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Research Copilot GNN API",
    description="Citation-graph-aware scientific paper retrieval and research assistant powered by GNNs and GraphRAG.",
    version="1.0.0",
)

# Enable CORS for all origins (as requested)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
retrieval_service = None
llm_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global retrieval_service, llm_service
    logger.info("Initializing RetrievalService...")
    retrieval_service = RetrievalService()
    logger.info("Initializing LLMService...")
    llm_service = LLMService()

# Models
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    use_gnn: bool = True

class SuggestRequest(BaseModel):
    project_description: str
    k: int = 5

# Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    return {
        "status": "ok",
        "model_loaded": retrieval_service.model_loaded,
        "gnn_available": retrieval_service.gnn_available,
    }

@app.post("/api/search")
async def search_papers(request: SearchRequest):
    """Search for papers by semantic + graph-structural similarity."""
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialized")
    
    try:
        results = retrieval_service.search(
            query=request.query,
            k=request.k,
            use_gnn=request.use_gnn
        )
        return results
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/suggest")
async def generate_suggestions(request: SuggestRequest):
    """Generate LLM-grounded research suggestions from top-K retrieved papers + graph reasoning paths."""
    if retrieval_service is None or llm_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # 1. Retrieve top-K papers using both semantic & GNN signals
        search_results = retrieval_service.search(
            query=request.project_description,
            k=request.k,
            use_gnn=True
        )
        retrieved_papers = search_results.get("results", [])
        
        # 2. Extract GraphRAG reasoning paths
        retrieved_nodes = [f"paper_{p.get('index', i)}" for i, p in enumerate(retrieved_papers)]
        query_paper_nodes = get_query_adjacent_papers(
            retrieval_service, 
            request.project_description, 
            k=3
        )
        
        reasoning_paths = extract_reasoning_paths(
            G=retrieval_service.graph,
            query_paper_nodes=query_paper_nodes,
            retrieved_nodes=retrieved_nodes,
            max_hops=3,
            max_paths=8
        )
        
        # 3. Generate LLM suggestions
        suggestions = llm_service.generate_suggestions(
            project_description=request.project_description,
            retrieved_papers=retrieved_papers,
            reasoning_paths=reasoning_paths
        )
        
        grounding_note = "Generated using GraphRAG paths and semantic retrieval."
        if not llm_service.enabled:
            grounding_note = "LLM generation disabled due to missing GROQ_API_KEY."
            
        return {
            "retrieved_papers": retrieved_papers,
            "reasoning_paths": reasoning_paths,
            "suggestions": suggestions,
            "grounding_note": grounding_note
        }
    except Exception as e:
        logger.error(f"Error in suggest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph")
async def get_graph():
    """Return the full graph structure as a JSON-serialisable node-link format."""
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialized")
    
    try:
        return retrieval_service.get_graph_data()
    except Exception as e:
        logger.error(f"Error getting graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/paper/{paper_id}")
async def get_paper(paper_id: str):
    """Get full metadata for a specific paper node."""
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialized")
    
    paper = retrieval_service.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
    
    return paper

@app.get("/api/stats")
async def get_stats():
    """Return model performance statistics."""
    if retrieval_service is None:
        raise HTTPException(status_code=503, detail="Retrieval service not initialized")
    
    return retrieval_service.get_stats()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
