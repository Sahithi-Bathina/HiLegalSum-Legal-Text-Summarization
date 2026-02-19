import streamlit as st
import numpy as np
import pandas as pd
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================
# 1. SETUP & MODEL CACHING
# ============================================
st.set_page_config(page_title="HiLegalSum UI", layout="wide")

@st.cache_resource
def load_resources():
    # Load the embedding model once
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # Initialize a basic vectorizer (can be refined with more data)
    vectorizer = TfidfVectorizer(max_features=4000, stop_words="english")
    return model, vectorizer

model, vectorizer = load_resources()

LEGAL_KEYWORDS = ["court", "plaintiff", "defendant", "act", "law", "section", "clause", "judgment", "order", "appeal"]

# ============================================
# 2. CORE LOGIC FUNCTIONS
# ============================================
def simple_sent_tokenize(text):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def run_hi_legal_sum(text, k, w_sem, w_pos, w_tfidf, lambda_param):
    sents = simple_sent_tokenize(text)
    if len(sents) <= k:
        return sents

    # Fit vectorizer on current text for local importance
    tfidf_matrix = vectorizer.fit_transform(sents).toarray()
    
    # Embeddings & Similarity
    emb = model.encode(sents, convert_to_numpy=True)
    sims = cosine_similarity(emb)

    # Graph-based centrality
    G = nx.from_numpy_array(sims)
    try:
        centrality = nx.eigenvector_centrality_numpy(G)
        centrality_scores = np.array([centrality[i] for i in range(len(sents))])
    except:
        centrality_scores = np.ones(len(sents))

    # Scoring Components
    pos_scores = np.array([1/np.sqrt(i+1) for i in range(len(sents))])
    tfidf_scores = tfidf_matrix.sum(axis=1)
    keyword_scores = np.array([sum([sent.lower().count(w) for w in LEGAL_KEYWORDS]) for sent in sents])

    # Combined Score
    total_score = (w_sem * centrality_scores) + (w_pos * pos_scores) + (w_tfidf * tfidf_scores) + (0.5 * keyword_scores)
    
    # MMR (Redundancy Reduction)
    candidates = list(range(len(sents)))
    final_idx = []
    while len(final_idx) < k and candidates:
        mmr_scores = []
        for idx in candidates:
            rel = total_score[idx]
            div = 0 if not final_idx else np.max(sims[idx][final_idx])
            mmr_scores.append(lambda_param * rel - (1 - lambda_param) * div)
        best_idx = candidates[np.argmax(mmr_scores)]
        final_idx.append(best_idx)
        candidates.remove(best_idx)
    
    return [sents[i] for i in sorted(final_idx)]

# ============================================
# 3. STREAMLIT UI LAYOUT
# ============================================
st.title("🧾 HiLegalSum")
st.subheader("Legal-Aware Extractive Summarization")

# Sidebar for Hyperparameters
with st.sidebar:
    st.header("⚙️ Algorithm Settings")
    k_val = st.slider("Sentences in Summary", 1, 15, 5)
    
    st.markdown("---")
    st.label_visibility = "visible"
    w_sem = st.slider("Semantic Weight ($w_{sem}$)", 0.0, 2.0, 1.0)
    w_pos = st.slider("Position Weight ($w_{pos}$)", 0.0, 1.0, 0.15)
    w_tfidf = st.slider("TF-IDF Weight ($w_{tfidf}$)", 0.0, 1.0, 0.25)
    lambda_p = st.slider(r"MMR Diversity ($\lambda$)", 0.0, 1.0, 0.7)

# Main Input Area
input_text = st.text_area("Paste your legal document or bill text here:", height=400, placeholder="Example: SECTION 1. SHORT TITLE. This Act may be cited as...")

if st.button("Generate Legal Summary", type="primary"):
    if not input_text.strip():
        st.error("Please provide some text to summarize.")
    else:
        with st.spinner("Processing legal nodes and graph centrality..."):
            summary = run_hi_legal_sum(input_text, k_val, w_sem, w_pos, w_tfidf, lambda_p)
            
            st.success("Summary Generated!")
            st.markdown("### 🔷 HiLegalSum Output")
            for sent in summary:
                st.markdown(f"**•** {sent}")
                
            # Bonus: Show statistics
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("Original Sentences", len(simple_sent_tokenize(input_text)))
            col2.metric("Summary Sentences", len(summary))