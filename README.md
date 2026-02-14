# Amazon Kids Product Search

**Elastic Blogathon 2026 - Production Search System**

## Features
- BBQ Quantization (75% memory savings)
- Hybrid RRF Search (BM25 + Vector)
- JinaAI Reranker v3
- 3-stage pipeline architecture

## Live Demo
[Deployed on Streamlit Cloud]

## Tech Stack
- Elasticsearch 8.16+ (BBQ int8_hnsw)
- Sentence Transformers
- JinaAI Reranker v3
- Streamlit

## Performance
- BM25: ~43ms
- Vector: ~69ms
- Hybrid: ~73ms
- Full Pipeline: ~373ms

## Deploy
1. Fork this repo
2. Deploy on Streamlit Cloud
3. Add secrets (CLOUD_ID, USERNAME, PASSWORD)
