# Research Copilot GNN

Graph ML-powered research assistant using Graph Neural Networks (GNNs), GraphRAG, and an LLM backend for scientific paper discovery.

![Architecture Concept](https://via.placeholder.com/800x400?text=Research+Copilot+GNN+Architecture)

## Overview

Research Copilot GNN is a full-stack AI application designed to help researchers explore scientific literature. It uses a pre-trained R-GCN (Relational Graph Convolutional Network) to embed academic papers in a heterogeneous citation graph (papers, concepts, authors). 

When querying the system, it retrieves papers using a weighted ensemble score:
*   60% Semantic Text Similarity (`sentence-transformers`)
*   25% GNN Activation Score (R-GCN embeddings)
*   15% Degree Centrality

The built-in **AI Copilot** uses GraphRAG to extract reasoning paths from the citation graph, providing grounded context to a Groq-powered LLM (Llama 3.1 8B) which then suggests ablations, baselines, and datasets for your specific research project.

## Quick Start

### Prerequisites
*   Python 3.10+
*   Node.js 18+

### Backend Setup

1.  Clone the repository and install dependencies:
    ```bash
    git clone https://github.com/shriyaasija/research-copilot-gnn.git
    cd research-copilot-gnn
    pip install -r requirements.txt
    ```

2.  Set up environment variables:
    Create a `.env` file in the root directory and add your Groq API key:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    PORT=8000
    ```

3.  Run the FastAPI server:
    ```bash
    uvicorn backend.main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

### Frontend Setup

1.  Navigate to the frontend directory and install dependencies:
    ```bash
    cd frontend
    npm install
    ```

2.  Set up environment variables:
    Create a `.env` file in the `frontend` directory:
    ```env
    VITE_API_URL=http://localhost:8000
    ```

3.  Run the Vite development server:
    ```bash
    npm run dev
    ```
    The UI will be available at `http://localhost:5173`.

## API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/search` | POST | Search for papers by semantic + graph-structural similarity. |
| `/api/suggest` | POST | Generate LLM-grounded research suggestions using GraphRAG. |
| `/api/graph` | GET | Return the full graph structure as a JSON node-link format. |
| `/api/paper/{id}`| GET | Get full metadata for a specific paper node. |
| `/api/stats` | GET | Return model performance statistics (NDCG, P@K, R@K). |
| `/api/health` | GET | System health check. |

## Model Performance

Evaluation on the citation graph test set (NDCG@10):

| Model | NDCG@10 | Precision@10 | Recall@10 |
| :--- | :--- | :--- | :--- |
| **R-GCN (Contrastive)** | **0.7891** | **0.7200** | **0.5200** |
| R-GCN (Base) | 0.7623 | 0.6900 | 0.4900 |
| GraphSAGE | 0.7389 | 0.6700 | 0.4700 |
| GAT | 0.7234 | 0.6500 | 0.4500 |
| Text-Only (MiniLM) | 0.6867 | 0.6100 | 0.4200 |

## Dataset

The system operates on a custom heterogeneous graph containing:
*   **459** Academic Papers (nodes)
*   **26** Concepts/Keyphrases (nodes)
*   **1478** Authors (nodes)
*   **3627** Total Edges (Relations: `cites`, `mentions`, `wrote`)

**Note:** Model weights (`rgcn_embeddings.npy`) and full graph files are not included in the repository due to size. The application will gracefully fall back to a text-only mode or use generated placeholders if these files are missing. See `docs/data_setup.md` for instructions to reproduce the data.

## Deployment

*   **Backend**: Configured for Render (`render.yaml`).
*   **Frontend**: Configured for Vercel (`vercel.json`).
*   **Live Demo**: [Link to Vercel Deployment]

## Citation & References

This project builds upon the following foundational research:

```bibtex
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Van Den Berg, Rianne and Titov, Ivan and Welling, Max},
  booktitle={European semantic web conference},
  pages={593--607},
  year={2018},
  organization={Springer}
}

@inproceedings{cohan-etal-2020-specter,
    title = "{SPECTER}: Document-level Representation Learning using Citation-informed Transformers",
    author = "Cohan, Arman  and Feldman, Sergey  and Beltagy, Iz  and Downey, Doug  and Weld, Daniel",
    booktitle = "ACL",
    year = "2020"
}
```
