import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import time

st.set_page_config(
    page_title="Amazon Kids Product Search",
    page_icon="🎨",
    layout="wide"
)

# Credentials from Streamlit secrets
try:
    CLOUD_ID = st.secrets["CLOUD_ID"]
    USERNAME = st.secrets["USERNAME"]
    PASSWORD = st.secrets["PASSWORD"]
except:
    st.error("⚠️ Please configure secrets in Streamlit Cloud dashboard")
    st.stop()

INDEX_NAME = "amazon_2020_bbq"

# Initialize connections
@st.cache_resource
def init_elasticsearch():
    return Elasticsearch(
        cloud_id=CLOUD_ID,
        basic_auth=(USERNAME, PASSWORD)
    )

@st.cache_resource
def init_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = init_elasticsearch()
model = init_model()

# Search functions
def bm25_search(query, k=10):
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["product_name^3", "brand^2", "category^1.5", "document_text"],
                    "type": "best_fields"
                }
            },
            "_source": ["product_name", "brand", "price", "category", "image_url"]
        }
    )
    return response["hits"]["hits"]

def vector_search(query, k=10):
    query_vector = model.encode(query).tolist()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 100
            },
            "_source": ["product_name", "brand", "price", "category", "image_url"]
        }
    )
    return response["hits"]["hits"]

def hybrid_rrf_search(query, k=10):
    query_vector = model.encode(query).tolist()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["product_name^3", "brand^2", "category^1.5", "document_text"]
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "embedding",
                                "query_vector": query_vector,
                                "k": 50,
                                "num_candidates": 100
                            }
                        }
                    ],
                    "rank_window_size": 100
                }
            },
            "_source": ["product_name", "brand", "price", "category", "image_url"]
        }
    )
    return response["hits"]["hits"]

def full_pipeline_search(query, k=10):
    query_vector = model.encode(query).tolist()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "retriever": {
                "text_similarity_reranker": {
                    "retriever": {
                        "rrf": {
                            "retrievers": [
                                {
                                    "standard": {
                                        "query": {
                                            "multi_match": {
                                                "query": query,
                                                "fields": ["product_name^3", "brand^2", "category^1.5", "document_text"]
                                            }
                                        }
                                    }
                                },
                                {
                                    "knn": {
                                        "field": "embedding",
                                        "query_vector": query_vector,
                                        "k": 50,
                                        "num_candidates": 100
                                    }
                                }
                            ],
                            "rank_window_size": 100
                        }
                    },
                    "field": "document_text",
                    "inference_id": "jina_reranker_v3",
                    "inference_text": query,
                    "rank_window_size": 50
                }
            },
            "_source": ["product_name", "brand", "price", "category", "image_url"]
        }
    )
    return response["hits"]["hits"]

# UI
st.title("🎨 Amazon Kids Product Search")
st.markdown("### Powered by BBQ Quantization + Hybrid RRF + JinaAI Reranker")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
comparison_mode = st.sidebar.checkbox("🔥 Compare All Methods", value=False)

if not comparison_mode:
    search_method = st.sidebar.selectbox(
        "Search Method",
        ["Full Pipeline", "Hybrid RRF", "Vector Semantic", "BM25 Keyword"]
    )
else:
    search_method = None

num_results = st.sidebar.slider("Results", 3, 12, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Try these:**")
examples = [
    "Hot Wheels race track",
    "Barbie dolls",
    "LEGO Star Wars",
    "educational STEM toys",
    "coloring books",
    "Pokemon plush",
    "board games family",
    "outdoor sports toys"
]

for ex in examples:
    if st.sidebar.button(ex, key=ex):
        st.session_state["query"] = ex

# Search bar
query = st.text_input(
    "Search for kids products:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., LEGO sets, educational toys, board games..."
)

# Comparison mode
if comparison_mode and query:
    st.markdown("### 🔬 Method Comparison")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📝 BM25**")
        start = time.time()
        bm25_res = bm25_search(query, k=num_results)
        bm25_time = (time.time() - start) * 1000
        st.metric("Latency", f"{bm25_time:.0f}ms")
        for i, hit in enumerate(bm25_res, 1):
            s = hit["_source"]
            st.markdown(f"**{i}.** {s['product_name'][:40]}...")
            st.caption(f"${s.get('price', 0):.2f}")
    
    with col2:
        st.markdown("**🧠 Vector**")
        start = time.time()
        vec_res = vector_search(query, k=num_results)
        vec_time = (time.time() - start) * 1000
        st.metric("Latency", f"{vec_time:.0f}ms")
        for i, hit in enumerate(vec_res, 1):
            s = hit["_source"]
            st.markdown(f"**{i}.** {s['product_name'][:40]}...")
            st.caption(f"${s.get('price', 0):.2f}")
    
    with col3:
        st.markdown("**🔀 Hybrid**")
        start = time.time()
        hyb_res = hybrid_rrf_search(query, k=num_results)
        hyb_time = (time.time() - start) * 1000
        st.metric("Latency", f"{hyb_time:.0f}ms")
        for i, hit in enumerate(hyb_res, 1):
            s = hit["_source"]
            st.markdown(f"**{i}.** {s['product_name'][:40]}...")
            st.caption(f"${s.get('price', 0):.2f}")
    
    with col4:
        st.markdown("**🚀 Pipeline**")
        start = time.time()
        pip_res = full_pipeline_search(query, k=num_results)
        pip_time = (time.time() - start) * 1000
        st.metric("Latency", f"{pip_time:.0f}ms")
        for i, hit in enumerate(pip_res, 1):
            s = hit["_source"]
            st.markdown(f"**{i}.** {s['product_name'][:40]}...")
            st.caption(f"${s.get('price', 0):.2f}")
    
    st.markdown("---")
    st.success("💡 Full Pipeline reranks for maximum relevance!")

# Single method mode
elif not comparison_mode and query and search_method:
    funcs = {
        "Full Pipeline": full_pipeline_search,
        "Hybrid RRF": hybrid_rrf_search,
        "Vector Semantic": vector_search,
        "BM25 Keyword": bm25_search
    }
    
    with st.spinner(f"Searching..."):
        start = time.time()
        results = funcs[search_method](query, k=num_results)
        latency = (time.time() - start) * 1000
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Latency", f"{latency:.0f}ms")
    col2.metric("📊 Results", len(results))
    col3.metric("🔧 Method", search_method)
    col4.metric("🎯 Storage", "BBQ int8")
    
    st.markdown("---")
    
    if results:
        for i, hit in enumerate(results, 1):
            s = hit["_source"]
            
            col_img, col_info = st.columns([1, 4])
            
            with col_img:
                img = s.get("image_url", "")
                if img and img.startswith("http"):
                    try:
                        st.image(img, width=120)
                    except:
                        st.write("🖼️")
                else:
                    st.write("🖼️")
            
            with col_info:
                st.markdown(f"### {i}. {s['product_name']}")
                d1, d2, d3, d4 = st.columns(4)
                d1.markdown(f"**Brand:** {s.get('brand', 'N/A') if s.get('brand') != 'nan' else 'N/A'}")
                d2.markdown(f"**Price:** ${s.get('price', 0):.2f}")
                d3.markdown(f"**Category:** {s.get('category', 'N/A')[:40] if s.get('category') != 'nan' else 'N/A'}")
                d4.markdown(f"**Score:** {hit['_score']:.3f}")
            
            st.markdown("---")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Architecture")
st.sidebar.markdown("""
**3-Stage Pipeline:**
1. Hybrid RRF (BM25 + Vector)
2. JinaAI Reranker v3
3. Final Results

**Features:**
- BBQ int8_hnsw (75% savings)
- Semantic + keyword fusion
- Cross-encoder reranking
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Elastic Blogathon 2026**")
