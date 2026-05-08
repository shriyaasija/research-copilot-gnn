# Data Setup Instructions

Model weights and large graph files are not included in the GitHub repository due to size constraints. To fully reproduce the GNN-enhanced features, you must generate or download these files.

## Required Files

The backend expects the following files in these locations relative to the repository root:

*   `outputs/models/rgcn_embeddings.npy`: Pre-computed 128-dimensional embeddings for all graph nodes from the R-GCN model.
*   `outputs/models/raw_embeddings.npy`: Pre-computed 384-dimensional sentence embeddings for all papers.
*   `data/graphs/hetero_graph.pkl`: NetworkX heterogeneous graph containing papers, concepts, and authors.
*   `data/processed/papers_enriched.json`: JSON file containing metadata for all papers.

*(Note: `papers_enriched.json` and `hetero_graph.pkl` are partially included in the data directory. If missing, the app will use placeholders or generate a simple dummy graph).*

## Reproduction Steps

To generate the full dataset and models yourself:

1.  **Data Ingestion**: Run the notebook `notebooks/01_ingestion_and_graph.ipynb` to process raw PDFs, extract metadata, and generate sentence embeddings (`raw_embeddings.npy`).
2.  **Graph Construction**: Run the notebook `notebooks/02_graph_construction.ipynb` to build the heterogeneous citation graph (`hetero_graph.pkl`).
3.  **Model Training**: Run the notebook `notebooks/03_gnn_training.ipynb` to train the R-GCN model and export the node embeddings (`rgcn_embeddings.npy`).

## Fallback Mode

If you start the API without the `rgcn_embeddings.npy` file, the system will automatically fall back to "text-only" retrieval using the raw sentence embeddings. 

If `raw_embeddings.npy` is also missing, the system will generate random embeddings for a set of 5 placeholder papers so the UI can still be tested.
