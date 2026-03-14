import streamlit as st
import numpy as np
import pandas as pd
import re
import networkx as nx
import pdfplumber
from docx import Document


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize


# ============================================
# 1. SETUP & MODEL CACHING
# ============================================

st.set_page_config(page_title="HiLegalSum UI", layout="wide")

@st.cache_resource
def load_resources():
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    vectorizer = TfidfVectorizer(max_features=4000, stop_words="english")
    return model, vectorizer

model, vectorizer = load_resources()

LEGAL_KEYWORDS = [
    "court","plaintiff","defendant","act","law",
    "section","clause","judgment","order","appeal"
]


# ============================================
# 2. HELPER FUNCTIONS
# ============================================

def simple_sent_tokenize(text):
    return sent_tokenize(text)


def extract_text_from_file(uploaded_file):

    text = ""

    if uploaded_file.type == "application/pdf":

        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":

        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text


# ============================================
# 3. CORE SUMMARIZATION ALGORITHM
# ============================================

def run_hi_legal_sum(text, k, w_sem, w_pos, w_tfidf, lambda_param):

    sents = simple_sent_tokenize(text)

    if len(sents) <= k:
        return sents

    tfidf_matrix = vectorizer.fit_transform(sents).toarray()

    emb = model.encode(sents, convert_to_numpy=True)
    sims = cosine_similarity(emb)

    G = nx.from_numpy_array(sims)

    try:
        centrality = nx.eigenvector_centrality_numpy(G)
        centrality_scores = np.array([centrality[i] for i in range(len(sents))])
    except:
        centrality_scores = np.ones(len(sents))

    pos_scores = np.array([1/np.sqrt(i+1) for i in range(len(sents))])

    tfidf_scores = tfidf_matrix.sum(axis=1)

    keyword_scores = np.array([
        sum([sent.lower().count(w) for w in LEGAL_KEYWORDS])
        for sent in sents
    ])

    total_score = (
        (w_sem * centrality_scores)
        + (w_pos * pos_scores)
        + (w_tfidf * tfidf_scores)
        + (0.5 * keyword_scores)
    )

    candidates = list(range(len(sents)))
    final_idx = []

    while len(final_idx) < k and candidates:

        mmr_scores = []

        for idx in candidates:

            rel = total_score[idx]

            if not final_idx:
                div = 0
            else:
                div = np.max(sims[idx][final_idx])

            mmr = lambda_param * rel - (1 - lambda_param) * div
            mmr_scores.append(mmr)

        best_idx = candidates[np.argmax(mmr_scores)]

        final_idx.append(best_idx)
        candidates.remove(best_idx)

    return [sents[i] for i in sorted(final_idx)]


# ============================================
# 4. STREAMLIT UI
# ============================================

st.title("🧾 HiLegalSum")
st.subheader("Legal-Aware Extractive Summarization")


# Sidebar
with st.sidebar:

    st.header("⚙️ Algorithm Settings")

    k_val = st.slider("Sentences in Summary", 1, 15, 5)

    st.markdown("---")

    w_sem = st.slider("Semantic Weight ($w_{sem}$)", 0.0, 2.0, 1.0)
    w_pos = st.slider("Position Weight ($w_{pos}$)", 0.0, 1.0, 0.15)
    w_tfidf = st.slider("TF-IDF Weight ($w_{tfidf}$)", 0.0, 1.0, 0.25)

    lambda_p = st.slider("MMR Diversity ($λ$)", 0.0, 1.0, 0.7)


# ============================================
# 5. INPUT SECTION (CHATGPT STYLE)
# ============================================

st.markdown("### 📝 Input Legal Document")

col1, col2 = st.columns([4,1])

with col1:

    input_text = st.text_area(
        "Paste legal document text:",
        height=350,
        placeholder="Example: SECTION 1. SHORT TITLE..."
    )

with col2:

    uploaded_file = st.file_uploader(
        "Add File",
        type=["pdf","docx"]
    )


# If file uploaded → override pasted text

if uploaded_file is not None:

    extracted_text = extract_text_from_file(uploaded_file)

    input_text = extracted_text

    st.success("Document loaded successfully!")


# ============================================
# 6. RUN SUMMARIZATION
# ============================================

if st.button("Generate Legal Summary", type="primary"):

    if not input_text.strip():

        st.error("Please provide text or upload a document.")

    else:

        with st.spinner("Processing legal nodes and graph centrality..."):

            summary = run_hi_legal_sum(
                input_text,
                k_val,
                w_sem,
                w_pos,
                w_tfidf,
                lambda_p
            )

            st.success("Summary Generated!")

            st.markdown("### 🔷 HiLegalSum Output")

            for sent in summary:
                st.markdown(f"• {sent}")

            st.divider()

            col1, col2 = st.columns(2)

            col1.metric(
                "Original Sentences",
                len(simple_sent_tokenize(input_text))
            )

            col2.metric(
                "Summary Sentences",
                len(summary)
            )