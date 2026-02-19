# ⚖️ HiLegalSum — Legal Document Summarization System

## 🔹 Overview

HiLegalSum is a hybrid extractive summarization system designed to generate concise summaries of long and complex legal documents.
It combines semantic, statistical, and structural features to identify the most informative sentences while preserving legal meaning and context.

The system is accompanied by an interactive web interface that allows users to input legal text and generate summaries in real time.

🔗 **Live Demo:** https://hilegalsum.streamlit.app/
🔗 **Source Code:** https://github.com/Sahithi-Bathina/HiLegalSum-Legal-Text-Summarization

---

## 🔹 Project Type

**Group Academic Research Project**

This project was developed collaboratively as part of an academic research initiative focused on legal-domain natural language processing.

---

## 🔹 My Contribution

**Research, Methodology Design & Documentation**

My primary contributions were research-oriented and included:

* Conducting literature review on legal text summarization approaches
* Designing the hybrid summarization methodology
* Defining evaluation strategy and metrics
* Analyzing experimental results and system behavior
* Writing and refining the research paper and documentation
* Contributing to system design decisions

> Note: The Python implementation was developed collaboratively by the project team.

---

## 🔹 Key Features

* 📄 Extractive summarization for long legal documents
* ⚖️ Legal-aware sentence importance scoring
* 🧠 Hybrid semantic + statistical approach
* 🎚️ Adjustable algorithm parameters via UI
* 🔍 Redundancy reduction using MMR
* 💻 Interactive Streamlit web interface
* ⚡ Real-time summary generation

---

## 🔹 Methodology Overview

HiLegalSum follows a multi-stage extractive pipeline:

1. **Preprocessing**

   * Sentence segmentation
   * Text normalization and cleaning

2. **Semantic Representation**

   * Sentence embeddings using Sentence-BERT

3. **Graph-Based Analysis**

   * Semantic similarity graph construction
   * Sentence centrality scoring

4. **Relevance Scoring**

   * TF-IDF weighting
   * Positional importance
   * Legal keyword awareness

5. **Redundancy Reduction**

   * Maximal Marginal Relevance (MMR)

6. **Summary Generation**

   * Top-ranked sentences selected based on combined score

This design emphasizes **interpretability**, **domain relevance**, and **factual consistency**, which are critical for legal applications.

---

## 🔹 Dataset

The system is evaluated using the **BillSum dataset**, a collection of U.S. legislative bills paired with professionally written summaries.

This dataset reflects real-world legal language, structure, and document length, making it suitable for evaluating legal summarization systems.

---

## 🔹 Results Summary

Experimental evaluation demonstrates that HiLegalSum:

* Outperforms classical extractive baselines such as Lead-3, Random, LexRank, and TextRank
* Achieves improved ROUGE and BERTScore metrics
* Effectively compresses long legal texts while preserving key information
* Produces coherent and contextually relevant summaries

Detailed quantitative analysis is provided in the accompanying research documentation.

---

## 🔹 Tech Stack

* **Language:** Python
* **Libraries:** NumPy, Scikit-learn, NLTK, Sentence-Transformers
* **Framework:** Streamlit
* **Techniques:** NLP, Semantic Similarity, Graph-Based Ranking

---

## 🔹 How to Run Locally

```bash
git clone https://github.com/Sahithi-Bathina/HiLegalSum-Legal-Text-Summarization.git
cd HiLegalSum-Legal-Text-Summarization
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔹 Applications

* Legal document review
* Legislative analysis
* Contract summarization
* Legal research assistance
* Academic study of legal NLP systems

---

## 🔹 Repository Structure

```
code/        → Core implementation and notebooks  
images/      → Figures and sample outputs  
paper/       → Research paper and documentation  
app.py       → Streamlit application  
requirements.txt → Dependencies  
```

---

## 🔹 Future Enhancements

* Abstractive summarization using transformer models
* Domain-adapted legal language models
* Multi-document summarization
* Integration into full-scale legal analysis platforms

---

## 🔹 Author

**Sahithi Bathina**

---

⭐ If you found this project useful, feel free to explore the live demo and repository.
