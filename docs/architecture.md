# Research Copilot GNN Architecture

This document describes the high-level architecture of the Research Copilot GNN application.

## System Overview

The system is a full-stack AI application designed to assist researchers in finding relevant papers and generating novel ideas. It leverages both semantic text similarity and structural graph features.

### Key Components

1.  **Heterogeneous Citation Graph**: A network representing academic papers, concepts (keyphrases), and authors. Edges represent relations like "cites", "mentions", and "wrote".
2.  **R-GCN (Relational Graph Convolutional Network)**: A Graph Neural Network model trained using contrastive learning to generate dense embeddings (128-dimensional) for each node in the graph, capturing structural relationships.
3.  **Semantic Embedding**: Papers and user queries are embedded using a pre-trained transformer model (`sentence-transformers/all-MiniLM-L6-v2`) into 384-dimensional vectors.
4.  **Ensemble Scorer**: Combines semantic similarity (60%), GNN activation (25%), and degree centrality (15%) to rank papers for a given query.
5.  **GraphRAG Reasoning**: Extracts shortest paths in the citation graph connecting query-adjacent nodes to retrieved papers to provide context for the LLM.
6.  **LLM Backend**: Uses Groq (Llama 3.1 8B Instant) to generate research suggestions (ablations, baselines, datasets) grounded in the retrieved papers and GraphRAG paths.

## Backend (FastAPI)

The backend provides a REST API with the following endpoints:

*   `POST /api/search`: Semantic + Graph-structural search.
*   `POST /api/suggest`: LLM-generated research suggestions using GraphRAG.
*   `GET /api/graph`: Returns the full graph structure for visualization.
*   `GET /api/paper/{paper_id}`: Metadata for a specific paper.
*   `GET /api/stats`: Evaluation metrics for the models.

### Fallback Mechanism

Due to the size of the pre-trained embeddings and model weights, they are not included in the repository. The backend is designed to degrade gracefully:
*   If GNN embeddings (`rgcn_embeddings.npy`) are missing, it falls back to a text-only retrieval mode.
*   If raw embeddings (`raw_embeddings.npy`) are missing, it generates random embeddings for a set of placeholder papers to ensure the API remains functional for demonstration purposes.

## Frontend (React + Vite)

The frontend is a modern single-page application built with React, Vite, TypeScript, and Tailwind CSS.

*   **Search Page**: Interface for searching papers with a toggle for GNN enhancement.
*   **Suggest Page (Copilot)**: Interface for entering project descriptions and receiving AI-generated suggestions.
*   **Graph Page**: Interactive 2D visualization of the citation network using `react-force-graph-2d`.
*   **About Page**: Project information and model evaluation statistics.

## Deployment

*   **Backend**: Deployed on Render (Free Tier) as a Python web service.
*   **Frontend**: Deployed on Vercel as a static site.
*   **CI/CD**: GitHub Actions workflows run tests on pushes and trigger Render deployments on new tags.
