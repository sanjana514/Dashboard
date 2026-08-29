# 🛒 Elite Apriori Mining Dashboard

An interactive web dashboard for discovering frequent itemsets in transactional data using the **Apriori algorithm** — built with Streamlit for real-time, visual market basket analysis.

📄 See [`Apriori Mining Dashboard output.pdf`](./Apriori%20Mining%20Dashboard%20output.pdf) for a walkthrough of the dashboard's output and visualizations.

## 📌 Overview

This dashboard lets users upload transaction data and instantly discover frequently co-occurring item patterns (association rule mining), commonly used in retail for market basket analysis, recommendation systems, and inventory planning. Users can tune mining parameters, explore results through multiple interactive visualizations, and export findings — all without writing a single line of code.

## ✨ Features

- **Real-Time Mining** — Runs the Apriori algorithm on uploaded transaction data instantly
- **Customizable Parameters** — Adjust minimum support threshold, maximum pattern size, and number of top results shown
- **Multi-Tab Analysis View**:
  - **Pattern Table** — Ranked view of top frequent itemsets
  - **Graph** — Bar charts, pattern size distribution, support distribution histogram, and count-vs-support scatter plot
  - **Details** — Full sorted itemset table plus summary statistics (max/min support, average & most frequent pattern size)
  - **Export** — Download complete results as CSV
- **Customizable Color Themes** — Multiple Plotly color scales (viridis, plasma, inferno, magma, cividis, teal, rainbow) for visualizations
- **Flexible CSV Input** — Supports both transaction-list format and one-hot/binary encoded format

## 🛠️ Tech Stack

- **Streamlit** — web app framework & UI
- **mlxtend** — Apriori algorithm & transaction encoding
- **Pandas** — data handling
- **Plotly Express** — interactive visualizations

## 📂 Supported CSV Formats

**1. Transaction list (space-separated):**
T1 Milk Bread
T2 Bread Butter

**2. One-hot / binary encoded:**
Each column represents an item; a value of `1`/`True` indicates the item is present in that transaction.

## 🚀 Running Locally

**1. Clone the repository:**
```bash
git clone https://github.com/sanjana514/elite-apriori-app.git
cd elite-apriori-app
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> 💡 A sample dataset (`transactions.csv`) is included in the repo for quick testing.

---

*This project was completed as part of the Big Data Analytics coursework at East West University.*