# ES Semantic search demo

This project consists of an Elasticsearch service and a Node.js Express API. The Express API uses an ONNX embeddings model for processing.

## Prerequisites

- Docker
- Docker Compose
- Node.js
- Python 3.11

## Setup

### 1. Download the ONNX Embeddings Model

First, run the `download_model.py` script to download the ONNX embeddings model. This step is necessary for the Express API to function correctly.

```bash
python download_model.py
```
### 2. Build and Run the Docker Containers
Use Docker Compose to build and run the containers for Elasticsearch and the Express API.
```bash
docker-compose up --build
```
### 3. Access the Services
- Elasticsearch will be accessible at http://localhost:9200
- The Express API will be accessible at http://localhost:3000

## Express API
The Express API provides endpoints to interact with the ONNX embeddings model and Elasticsearch.

### Endpoints
- POST /enter-data: Indexes data into Elasticsearch.
- POST /vector-search: Performs a vector similarity search using the ONNX model.
- POST /text-search: Performs a text-based fuzzy search on ProductName and Description.