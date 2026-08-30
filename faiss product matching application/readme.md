# 🔍 FAISS Product Matching System

An interactive semantic product search engine built with **FAISS** and **Sentence Transformers** — upload any product catalog and instantly search it using natural language, with real accuracy metrics and side-by-side comparisons across four different similarity search algorithms.

📄 See [`faiss product matching output.pdf`](./faiss%20product%20matching%20output.pdf) for a full walkthrough of the app's search results, and [`faiss index comparison.pdf`](./faiss%20index%20comparison.pdf) for the detailed FlatL2 vs IVF vs HNSW benchmark comparison.

## 📌 Overview

Traditional keyword search fails when a customer searches "warm winter jacket" but the product is listed as "insulated puffer coat." This system solves that by embedding product text into vector space using a sentence-transformer model, then using **FAISS** (Facebook AI Similarity Search) to retrieve semantically similar products in milliseconds — even across datasets scaled up to 200,000 products.

Beyond search, the app lets you benchmark and compare four different FAISS index strategies side-by-side, with real Recall, Precision, F1, and MAP metrics calculated against an exact-search ground truth — not simulated numbers.

## ✨ Features

- **Semantic Product Search** — natural language queries matched against product catalogs using sentence embeddings
- **4 FAISS Index Types** — FlatL2, FlatIP, IVF, and HNSW, selectable and comparable in real time
- **Accurate Evaluation Metrics** — Recall@K, Precision@K, F1 Score, and MAP@K, computed against an exact cosine-similarity ground truth (FlatIP)
- **Performance Comparison Charts** — build time and search time compared across any two (or all four) index types
- **Comprehensive Benchmark Suite** — automatically tests FlatL2 vs IVF across dataset sizes from the uploaded data up to 200K synthetic records, plus IVF configuration tuning (`nlist`/`nprobe`)
- **Card-Based Results UI** — matched products displayed as clean cards with a prominent similarity-percentage badge, not a raw table
- **Flexible CSV Input** — auto-detects a `name`/`product_name` column, or combines available text columns

## 🛠️ Tech Stack

- **Streamlit** — web app framework & UI
- **FAISS** (`faiss-cpu`) — vector similarity search / indexing
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — text embedding generation
- **Plotly** — interactive performance and benchmark charts
- **Pandas / NumPy** — data handling and metric computation

## 🧠 Index Types Compared

| Index | Description | Best For | Speed | Accuracy |
|-------|-------------|----------|-------|----------|
| **FlatL2** | Exact L2 distance search | Small datasets, highest accuracy | Slow | 100% |
| **FlatIP** | Exact cosine similarity (used as ground truth) | Small datasets, semantic search | Slow | 100% |
| **IVF** | Inverted File Index — clusters data for fast approximate search | Large datasets | Fast | 95–99% |
| **HNSW** | Hierarchical graph-based search | Balanced speed/accuracy | Very Fast | 97–99% |

📊 The included benchmark (see output PDF) shows IVF achieving significant speedup over exact FlatL2 search as dataset size scales toward 200K products, while maintaining recall above 95% with a tuned `nlist`/`nprobe` configuration.

## 📐 Metrics Explained

- **Recall@K**: fraction of ground-truth items successfully retrieved — `(Retrieved ∩ Ground Truth) / K`
- **Precision@K**: fraction of retrieved items that are actually relevant — `(Retrieved ∩ Ground Truth) / Retrieved`
- **F1 Score**: harmonic mean of Precision and Recall
- **MAP@K**: Mean Average Precision — rewards correct results appearing higher in the ranking

## 📂 CSV Format

```csv
product_id,name,category,price,discount
1,Wireless Headphones,Electronics,99.99,10%
2,Red Cotton Shirt,Clothing,29.99,15%
3,Gaming Laptop 16GB,Computers,1299.99,5%
```

- **Best practice:** include a `name` or `product_name` column
- **Alternative:** any text columns will be auto-combined into a searchable field
- **Performance:** tested with datasets from 1K to 200K products

## 🚀 Running Locally

**1. Clone the repository:**
```bash
git clone https://github.com/sanjana514/big-data-analytics-projects.git
cd "faiss product matching application"
```

**2. Install dependencies:**
```bash
pip install streamlit pandas numpy faiss-cpu sentence-transformers plotly
```

**3. Run the app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> 💡 On first run, the sentence-transformer model (`all-MiniLM-L6-v2`) downloads automatically — this requires an internet connection and takes a few seconds.

---

*This project was completed as part of the Big Data Analytics coursework at East West University.*