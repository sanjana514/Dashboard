#  Big Data Analytics — Course Projects

This repository holds two mini-projects I built as part of the **Big Data Analytics** course at East West University. Both apply core big data techniques — similarity search at scale and frequent pattern mining — to problems that come up often in real-world retail and recommendation systems, wrapped in interactive dashboards rather than plain scripts.

## 📂 Projects in This Repo

### 1. 🔍 [FAISS Product Matching System](./faiss%20product%20matching%20application)

A semantic product search engine that lets you search a product catalog using natural language instead of exact keywords, powered by **FAISS** (Facebook AI Similarity Search) and sentence embeddings.

**Key points:**
- Compares **4 different FAISS indexing strategies** (FlatL2, FlatIP, IVF, HNSW) side-by-side, not just one
- Evaluation is based on real **Recall, Precision, F1, and MAP** scores measured against an exact-search ground truth — not just eyeballed results
- Includes a full **benchmark suite** testing performance across dataset sizes up to 200K records, plus IVF configuration tuning
- Built to show the practical speed-vs-accuracy trade-off that comes up whenever exact search doesn't scale

📄 Full details: [`faiss product matching application/README.md`](./faiss%20product%20matching%20application/readme.md)

### 2. 🛒 [Frequent Pattern Mining Dashboard](./frequent%20pattern%20mining%20dashboard)

An interactive dashboard for discovering frequently co-occurring items in transactional data using the **Apriori algorithm** — the same kind of analysis behind "customers who bought this also bought…" recommendations.

**Key points:**
- Lets you upload any transaction dataset and tune mining parameters (support threshold, pattern size) live
- Visualizes results through multiple charts — support distribution, pattern size breakdown, and top itemsets
- Supports both transaction-list and one-hot encoded CSV formats
- Turns a normally command-line, one-off analysis into something explorable and repeatable

📄 Full details: [`frequent pattern mining dashboard/README.md`](./frequent%20pattern%20mining%20dashboard/README.md)

##  Why These Two Together

Both projects tackle the same underlying question from different angles — *how do you find meaningful structure in large datasets efficiently?* FAISS does this for similarity in high-dimensional vector space, and Apriori does it for co-occurrence patterns in transactional data. Together they cover two of the more common big data problems: fast retrieval at scale, and pattern discovery in large transaction logs.

## 🛠️ Shared Tech Stack

Both dashboards are built with **Streamlit** for the interface, **Pandas** for data handling, and **Plotly** for visualizations — the FAISS project additionally uses **FAISS** and **Sentence-Transformers**, while the mining dashboard uses **mlxtend** for the Apriori implementation.

---

*Both projects were completed individually as part of the Big Data Analytics coursework at East West University.*
