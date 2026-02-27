# Amazon Kids Product Search

**Elastic Blogathon 2026 - Production Search System**
**Link to detailed google colab notebook:**https://colab.research.google.com/drive/1MH8HCEjyEch9_Q0754D3dO7Gz2ZeZlAI?usp=sharing
## Features
- BBQ Quantization (75% memory savings)
- Hybrid RRF Search (BM25 + Vector)
- JinaAI Reranker v3
- 3-stage pipeline architecture
  

## Live Demo
[Deployed on Streamlit Cloud]
https://amazon-kids-appuct-search-wjchxxwedrdklawdflpvju.streamlit.app/#1-lego-storage-brick-4-lime-green
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
