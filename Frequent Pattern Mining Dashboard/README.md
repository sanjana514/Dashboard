# 🛒 Elite Apriori Mining Dashboard

Welcome to the **Elite Apriori Mining Dashboard**, a powerful and user-friendly tool built with Streamlit to perform Apriori algorithm-based frequent pattern mining on transactional data. This dashboard allows you to upload CSV files, analyze patterns, visualize results, and export findings with ease.

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
## Deploying the App

This app is deployed using Streamlit Community Cloud.
Live URL: https://2hcgapppyrzb2daifseu7mp.streamlit.app/ – Access the full dashboard here!
Use the transactions.csv dataset file to test the app.

*This project was completed as part of the Big Data Analytics coursework at East West University.*