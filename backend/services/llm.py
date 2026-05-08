"""
LLM integration service for generating grounded research suggestions.
Uses Groq API with Llama 3.1 8B Instant.
"""

import os
import logging
from typing import Optional

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Groq LLM API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = None
        self.enabled = False

        if HAS_GROQ and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.enabled = True
                logger.info("Groq LLM client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        else:
            if not HAS_GROQ:
                logger.warning("groq package not installed. Run `pip install groq`.")
            if not self.api_key:
                logger.warning("GROQ_API_KEY environment variable not set. LLM suggestions disabled.")

    def generate_suggestions(
        self,
        project_description: str,
        retrieved_papers: list[dict],
        reasoning_paths: list[str],
        model: str = "llama-3.1-8b-instant",
    ) -> str:
        """
        Generate research suggestions grounded in retrieved papers and graph paths.
        
        Args:
            project_description: User's project description
            retrieved_papers: Top retrieved papers from semantic/graph search
            reasoning_paths: Verbalized citation-graph paths (GraphRAG context)
            model: Groq model name
            
        Returns:
            LLM-generated suggestions string
        """
        if not self.enabled:
            return (
                "**LLM Generation Disabled**\n\n"
                "To enable AI research suggestions, please provide a valid "
                "`GROQ_API_KEY` in your environment variables and restart the API."
            )

        # Format retrieved papers for context
        papers_context = ""
        for i, paper in enumerate(retrieved_papers):
            title = paper.get("title", "Unknown Title")
            authors = ", ".join(paper.get("authors", []))[:50]
            year = paper.get("year", "Unknown Year")
            abstract = paper.get("abstract", "")[:500] + "..."
            
            papers_context += f"[{i+1}] {title}\n"
            papers_context += f"Authors: {authors} | Year: {year}\n"
            papers_context += f"Abstract snippet: {abstract}\n\n"

        # Format graph paths for context
        paths_context = "\n".join(f"- {path}" for path in reasoning_paths)
        if not paths_context:
            paths_context = "No direct citation paths found among the top results."

        prompt = f"""
You are Research Copilot GNN, an expert AI assistant for scientific researchers. 
The user is working on a new research project and needs grounded suggestions.
You have retrieved relevant academic papers using a Graph Neural Network and GraphRAG system.

USER'S PROJECT DESCRIPTION:
{project_description}

RETRIEVED RELEVANT PAPERS:
{papers_context}

CITATION GRAPH REASONING PATHS (GraphRAG):
These paths show how the retrieved papers connect to each other or to specific concepts in the scientific literature network:
{paths_context}

YOUR TASK:
Generate actionable research suggestions based EXCLUSIVELY on the provided papers and graph paths. 
Structure your response exactly with these three headings:

### Ablations
Suggest 1-2 ablation studies or architectural variants the user should try, explicitly citing the [Number] of the paper that inspired the idea or referring to a concept in the graph paths.

### Baselines
Recommend 2-3 standard baselines the user should compare their method against, based on the retrieved papers.

### Datasets / Setup
Identify any datasets, metrics, or experimental setups mentioned in the retrieved papers that the user should consider adopting.

Keep your response concise, academic, and highly grounded in the provided context. Do not invent papers or citations.
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic research assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model,
                temperature=0.3,
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return f"An error occurred while generating suggestions with the LLM API: {str(e)}"
