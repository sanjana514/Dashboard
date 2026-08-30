"""
FAISS Product Matching System - Streamlit Version (v2 - Card UI)
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import time
import plotly.graph_objects as go
from datetime import datetime
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="FAISS Product Matching System", page_icon="🔍", layout="wide")

# ============================================================================
# Configuration
# ============================================================================
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
IVF_NLIST = 100
IVF_NPROBE = 10
HNSW_M = 32
HNSW_EF = 128

# ============================================================================
# Session State Initialization
# ============================================================================
for key in ['uploaded_data', 'embeddings', 'model', 'indices', 'build_times']:
    if key not in st.session_state:
        st.session_state[key] = None
if st.session_state.indices is None:
    st.session_state.indices = {}
if st.session_state.build_times is None:
    st.session_state.build_times = {}

# ============================================================================
# Global Styling — fixed contrast: dark bg -> light text everywhere
# ============================================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong {
        color: #f1f1f6 !important;
    }
    h1, h2, h3, h4, h5 { color: #ffffff !important; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px 30px; border-radius: 20px; margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4); text-align: center;
    }
    .main-header h1 { color: #ffffff !important; margin: 0; font-size: 28px; font-weight: 800; }
    .main-header p { color: rgba(255,255,255,0.95) !important; margin: 12px 0 0 0; font-size: 16px; font-weight: 500; }

    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important; border: none !important; font-weight: 700 !important;
        border-radius: 10px !important; padding: 0.6rem 1.5rem !important;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102,126,234,0.4); }

    .stTabs [data-baseweb="tab-list"] button { color: #cfcfe8 !important; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: #ffffff !important; border-bottom-color: #8b5cf6 !important; }

    .stTextInput input, .stSelectbox div, .stSlider label { color: #ffffff !important; }
    .stTextInput input { background: rgba(255,255,255,0.08) !important; }

    .stDataFrame { background: rgba(255,255,255,0.03); border-radius: 10px; }

    .stMarkdown table { color: #f1f1f6 !important; }
    .stMarkdown th { color: #ffffff !important; background: rgba(255,255,255,0.08) !important; }
    .stMarkdown td { color: #e5e5f0 !important; }

    .stMarkdown code { color: #ffd166 !important; background: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 FAISS Product Matching System</h1>
    <p>✨ Accurate Metrics • ⚡ Real FAISS • 📊 Comparison Charts • ✅ Fixed Calculations</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Core Function: Process CSV and Build Indices
# ============================================================================
def process_csv(file):
    try:
        data = pd.read_csv(file)

        if 'name' in data.columns:
            data['search_text'] = data['name'].astype(str)
        elif 'product_name' in data.columns:
            data['search_text'] = data['product_name'].astype(str)
        else:
            text_cols = data.select_dtypes(include=['object']).columns[:3]
            data['search_text'] = data[text_cols].astype(str).agg(' '.join, axis=1)

        data['search_text'] = data['search_text'].str.lower().str.strip()
        data = data.reset_index(drop=True)

        if st.session_state.model is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.model = SentenceTransformer(EMBEDDING_MODEL)

        with st.spinner(f"Generating embeddings for {len(data):,} products..."):
            embeddings = st.session_state.model.encode(
                data['search_text'].tolist(),
                show_progress_bar=False,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype('float32')

        indices = {}
        build_times = {}

        start = time.time()
        idx_flatl2 = faiss.IndexFlatL2(EMBEDDING_DIM)
        idx_flatl2.add(embeddings)
        build_times['FlatL2'] = time.time() - start
        indices['FlatL2'] = idx_flatl2

        start = time.time()
        idx_flatip = faiss.IndexFlatIP(EMBEDDING_DIM)
        idx_flatip.add(embeddings)
        build_times['FlatIP'] = time.time() - start
        indices['FlatIP'] = idx_flatip

        start = time.time()
        nlist = min(IVF_NLIST, max(1, len(embeddings) // 10))
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        idx_ivf = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist)
        idx_ivf.train(embeddings)
        idx_ivf.add(embeddings)
        idx_ivf.nprobe = min(IVF_NPROBE, nlist)
        build_times['IVF'] = time.time() - start
        indices['IVF'] = idx_ivf

        start = time.time()
        idx_hnsw = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        idx_hnsw.hnsw.efSearch = HNSW_EF
        idx_hnsw.add(embeddings)
        build_times['HNSW'] = time.time() - start
        indices['HNSW'] = idx_hnsw

        st.session_state.uploaded_data = data
        st.session_state.embeddings = embeddings
        st.session_state.indices = indices
        st.session_state.build_times = build_times

        return True, build_times

    except Exception as e:
        return False, str(e)


# ============================================================================
# Core Function: Search Products
# ============================================================================
def search_products(query, top_k, selected_index, compare_with):
    data = st.session_state.uploaded_data
    model = st.session_state.model
    indices = st.session_state.indices
    build_times = st.session_state.build_times

    logs = []
    logs.append(f"Query: '{query}' | Top K: {top_k} | Index: {selected_index}")
    logs.append(f"Dataset size: {len(data):,} products")

    query_embedding = model.encode([query], normalize_embeddings=True).astype('float32')
    logs.append(f"Query encoded to {EMBEDDING_DIM}D vector")

    ground_truth_index = indices['FlatIP']
    gt_distances, gt_indices = ground_truth_index.search(query_embedding, top_k)
    gt_set = set(gt_indices[0].tolist())

    selected_idx = indices[selected_index]
    start_time = time.time()
    distances, result_indices = selected_idx.search(query_embedding, top_k)
    search_time = time.time() - start_time

    result_indices = result_indices[0]
    result_distances = distances[0]
    retrieved_set = set(result_indices.tolist())

    logs.append(f"{selected_index} search completed in {search_time*1000:.2f}ms")

    intersection = len(gt_set.intersection(retrieved_set))
    recall = intersection / len(gt_set) if len(gt_set) > 0 else 0
    precision = intersection / len(retrieved_set) if len(retrieved_set) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    ap = 0
    hits = 0
    for i, idx in enumerate(result_indices):
        if idx in gt_set:
            hits += 1
            ap += hits / (i + 1)
    map_score = ap / len(gt_set) if len(gt_set) > 0 else 0

    logs.append(f"Metrics: Recall={recall:.4f} | Precision={precision:.4f} | F1={f1:.4f} | MAP={map_score:.4f}")
    logs.append(f"Ground Truth Overlap: {intersection}/{top_k} items matched")

    if selected_index in ['FlatIP', 'IVF', 'HNSW']:
        similarities = np.clip(result_distances, 0, 1)
    else:
        max_dist = np.max(result_distances) if np.max(result_distances) > 0 else 1
        similarities = 1 - (result_distances / (max_dist + 1e-6))
        similarities = np.clip(similarities, 0, 1)

    results_df = data.iloc[result_indices].copy()
    results_df['Rank'] = range(1, top_k + 1)
    results_df['Similarity'] = [f"{s*100:.1f}%" for s in similarities]
    results_df['Distance'] = [f"{d:.4f}" for d in result_distances]

    if compare_with == "All Indices":
        compare_indices = ['FlatL2', 'FlatIP', 'IVF', 'HNSW']
    else:
        compare_indices = [selected_index, compare_with]

    search_times = []
    for idx_name in compare_indices:
        idx = indices[idx_name]
        s = time.time()
        _, _ = idx.search(query_embedding, top_k)
        t = (time.time() - s) * 1000
        search_times.append(t)

    build_times_list = [build_times[idx] * 1000 for idx in compare_indices]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Build Time (ms)', x=compare_indices, y=build_times_list,
                          marker_color='#667eea', text=[f'{t:.1f}' for t in build_times_list], textposition='outside'))
    fig.add_trace(go.Bar(name='Search Time (ms)', x=compare_indices, y=search_times,
                          marker_color='#38ef7d', text=[f'{t:.2f}' for t in search_times], textposition='outside'))
    fig.update_layout(
        title=f'Performance Comparison: {selected_index} vs {compare_with}',
        xaxis_title='Index Type', yaxis_title='Time (milliseconds)', height=400,
        barmode='group', template='plotly_dark',
        plot_bgcolor='rgb(30,30,30)', paper_bgcolor='rgb(30,30,30)',
        font=dict(color='#f1f1f6'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    metrics = {
        'recall': recall, 'precision': precision, 'f1': f1, 'map': map_score,
        'search_time': search_time, 'build_time': build_times[selected_index]
    }

    return results_df, fig, logs, metrics


# ============================================================================
# HTML Card Renderers
# ============================================================================
def render_performance_card(index_name, build_time, search_time):
    total_time = build_time + search_time
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 22px 26px; border-radius: 14px; margin-bottom: 18px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.35);">
        <h3 style="margin: 0 0 14px 0; color: #0b1220; font-size: 19px; font-weight: 800;">
            {index_name} Performance
        </h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;">
            <div>
                <p style="margin:0; color:#0b1220; opacity:0.75; font-size:12px; font-weight:600;">BUILD TIME</p>
                <p style="margin:4px 0 0 0; color:#0b1220; font-size:24px; font-weight:800;">{build_time:.3f}s</p>
            </div>
            <div>
                <p style="margin:0; color:#0b1220; opacity:0.75; font-size:12px; font-weight:600;">SEARCH TIME</p>
                <p style="margin:4px 0 0 0; color:#0b1220; font-size:24px; font-weight:800;">{search_time*1000:.2f}ms</p>
            </div>
            <div>
                <p style="margin:0; color:#0b1220; opacity:0.75; font-size:12px; font-weight:600;">TOTAL TIME</p>
                <p style="margin:4px 0 0 0; color:#0b1220; font-size:24px; font-weight:800;">{total_time:.3f}s</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_eval_metrics(top_k, recall, precision, f1, map_score):
    boxes = [
        (f"Recall@{top_k}", recall, "#f7b733", "#fc4a1a"),
        (f"Precision@{top_k}", precision, "#fbc2eb", "#a6c1ee"),
        ("F1 Score", f1, "#89f7fe", "#66a6ff"),
        (f"MAP@{top_k}", map_score, "#f6d365", "#fda085"),
    ]
    cols = st.columns(4)
    for col, (label, value, c1, c2) in zip(cols, boxes):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {c1} 0%, {c2} 100%);
                        padding: 20px; border-radius: 14px; text-align: center;
                        box-shadow: 0 4px 14px rgba(0,0,0,0.35);">
                <p style="margin:0; color:#0b1220; opacity:0.8; font-size:13px; font-weight:700;">{label}</p>
                <p style="margin:8px 0 0 0; color:#0b1220; font-size:30px; font-weight:800;">{value:.3f}</p>
            </div>
            """, unsafe_allow_html=True)


def render_product_cards(results_df, exclude_cols=('search_text',)):
    cards_html = "<div style='display:flex; flex-wrap:wrap; gap:18px; margin-top: 10px;'>"
    for _, row in results_df.iterrows():
        details = ""
        for col, val in row.items():
            if col in exclude_cols or col in ['Rank', 'Similarity', 'Distance']:
                continue
            details += f"<p style='margin:4px 0; font-size:13px; color:#e5e5f0;'><strong style='color:#ffffff;'>{col}:</strong> {val}</p>"

        cards_html += f"""
        <div style="background: linear-gradient(135deg, #302b63 0%, #24243e 100%);
                    padding: 20px; border-radius: 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.35);
                    width: calc(50% - 9px); min-width: 300px; max-height: 300px; overflow-y: auto;
                    border: 1px solid rgba(255,255,255,0.08);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <h3 style="margin:0; color:#4facfe; font-size:18px; font-weight:800;">Rank {row['Rank']}</h3>
                <p style="margin:0; color:#38ef7d; font-size:20px; font-weight:800;">{row['Similarity']}</p>
            </div>
            <div>{details}</div>
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ============================================================================
# Core Function: Benchmark
# ============================================================================
def create_synthetic_dataset(emb, target_size):
    if target_size <= len(emb):
        return emb[:target_size]
    repeats = (target_size // len(emb)) + 1
    synthetic = np.tile(emb, (repeats, 1))[:target_size]
    noise = np.random.normal(0, 0.01, synthetic.shape).astype('float32')
    result = synthetic + noise
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return (result / norms).astype('float32')


def benchmark_index(emb, idx_type, test_queries=100):
    n, d = emb.shape
    start = time.time()
    if idx_type == 'FlatL2':
        idx = faiss.IndexFlatL2(d)
        idx.add(emb)
    elif idx_type == 'IVF':
        nlist = min(100, max(1, n // 10))
        quantizer = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quantizer, d, nlist)
        idx.train(emb)
        idx.add(emb)
        idx.nprobe = min(10, nlist)
    build_time = time.time() - start

    num_queries = min(test_queries, n)
    queries = emb[:num_queries]
    start = time.time()
    _, _ = idx.search(queries, 10)
    search_time = (time.time() - start) / num_queries * 1000
    return build_time, search_time


def run_benchmark(progress_bar, status_text):
    embeddings = st.session_state.embeddings
    sizes = [
        (len(embeddings), f'{len(embeddings)/1000:.1f}K'),
        (50000, '50K'), (100000, '100K'), (150000, '150K'), (200000, '200K')
    ]

    comparison_data = []
    for i, (size, label) in enumerate(sizes):
        status_text.text(f"Testing {label}...")
        progress_bar.progress((i + 1) / (len(sizes) + 1))

        test_emb = create_synthetic_dataset(embeddings, size) if size > len(embeddings) else embeddings[:size]
        flat_build, flat_search = benchmark_index(test_emb, 'FlatL2')
        ivf_build, ivf_search = benchmark_index(test_emb, 'IVF')
        speedup = flat_search / ivf_search if ivf_search > 0 else 0

        comparison_data.append({
            'Dataset Size': label, 'FlatL2 Build (s)': round(flat_build, 3),
            'FlatL2 Search (ms)': round(flat_search, 2), 'IVF Build (s)': round(ivf_build, 3),
            'IVF Search (ms)': round(ivf_search, 2), 'IVF Speedup': f'{speedup:.2f}x'
        })

    comparison_df = pd.DataFrame(comparison_data)

    status_text.text("Testing IVF configurations...")
    test_size = min(100000, len(embeddings) * 2)
    test_emb = create_synthetic_dataset(embeddings, test_size) if test_size > len(embeddings) else embeddings

    idx_flat = faiss.IndexFlatIP(EMBEDDING_DIM)
    idx_flat.add(test_emb)
    num_test_queries = min(100, len(test_emb))
    test_queries = test_emb[:num_test_queries]
    _, gt_results = idx_flat.search(test_queries, 10)

    ivf_configs = []
    for nlist, nprobe in [(100, 10), (200, 10), (100, 20), (200, 20)]:
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        idx_ivf = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist)
        start = time.time()
        idx_ivf.train(test_emb)
        idx_ivf.add(test_emb)
        build_time = time.time() - start
        idx_ivf.nprobe = nprobe

        start = time.time()
        _, ivf_results = idx_ivf.search(test_queries, 10)
        search_time = (time.time() - start) / num_test_queries * 1000

        recalls, precisions = [], []
        for i in range(num_test_queries):
            gt_set = set(gt_results[i])
            ivf_set = set(ivf_results[i])
            intersection = len(gt_set.intersection(ivf_set))
            recalls.append(intersection / len(gt_set) if len(gt_set) > 0 else 0)
            precisions.append(intersection / len(ivf_set) if len(ivf_set) > 0 else 0)

        ivf_configs.append({
            'nlist': nlist, 'nprobe': nprobe, 'Build Time (s)': round(build_time, 3),
            'Search Time (ms)': round(search_time, 2),
            'Recall@10': round(np.mean(recalls), 4), 'Precision@10': round(np.mean(precisions), 4)
        })

    ivf_config_df = pd.DataFrame(ivf_configs)
    progress_bar.progress(1.0)
    status_text.text("Benchmark completed!")

    fig_build = go.Figure()
    fig_build.add_trace(go.Bar(name='FlatL2', x=comparison_df['Dataset Size'], y=comparison_df['FlatL2 Build (s)'], marker_color='#667eea'))
    fig_build.add_trace(go.Bar(name='IVF', x=comparison_df['Dataset Size'], y=comparison_df['IVF Build (s)'], marker_color='#38ef7d'))
    fig_build.update_layout(title='Index Build Time: FlatL2 vs IVF', xaxis_title='Dataset Size',
                             yaxis_title='Build Time (s)', barmode='group', height=420, template='plotly_dark',
                             plot_bgcolor='rgb(30,30,30)', paper_bgcolor='rgb(30,30,30)', font=dict(color='#f1f1f6'))

    fig_search = go.Figure()
    fig_search.add_trace(go.Bar(name='FlatL2', x=comparison_df['Dataset Size'], y=comparison_df['FlatL2 Search (ms)'], marker_color='#f093fb'))
    fig_search.add_trace(go.Bar(name='IVF', x=comparison_df['Dataset Size'], y=comparison_df['IVF Search (ms)'], marker_color='#4facfe'))
    fig_search.update_layout(title='Search Time per Query: FlatL2 vs IVF', xaxis_title='Dataset Size',
                              yaxis_title='Search Time (ms)', barmode='group', height=420, template='plotly_dark',
                              plot_bgcolor='rgb(30,30,30)', paper_bgcolor='rgb(30,30,30)', font=dict(color='#f1f1f6'))

    max_speedup = max([float(row['IVF Speedup'].replace('x', '')) for row in comparison_data])
    best_config = ivf_config_df.loc[ivf_config_df['Recall@10'].idxmax()]

    summary = {
        'max_speedup': max_speedup,
        'best_nlist': int(best_config['nlist']),
        'best_nprobe': int(best_config['nprobe']),
        'best_recall': ivf_config_df['Recall@10'].max()
    }

    return summary, fig_build, fig_search, comparison_df, ivf_config_df


# ============================================================================
# UI: Tabs
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🔍 Search Products", "📊 Benchmark Analysis", "📖 Documentation"])

with tab1:
    csv_file = st.file_uploader("📁 Upload CSV Dataset", type=['csv'])

    if csv_file is not None and st.session_state.uploaded_data is None:
        success, result = process_csv(csv_file)
        if success:
            st.success(f"✅ Dataset loaded! {len(st.session_state.uploaded_data):,} products | "
                       f"Build times — FlatL2: {result['FlatL2']:.2f}s | FlatIP: {result['FlatIP']:.2f}s | "
                       f"IVF: {result['IVF']:.2f}s | HNSW: {result['HNSW']:.2f}s")
        else:
            st.error(f"❌ Error: {result}")

    if st.session_state.uploaded_data is not None:
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(st.session_state.uploaded_data.head(10), use_container_width=True)

        st.markdown("""
        <div style="background: rgba(255,255,255,0.08); padding: 14px 18px; border-radius: 10px; margin: 14px 0;">
            <p style="margin:0; color:#f1f1f6; font-size:14px;">
                💡 <strong>Tips:</strong> Enter a search query → select index type → choose comparison → click Search
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("🔎 Search Query", placeholder="e.g., 'wireless headphones', 'red shirt'...")
        with col2:
            top_k = st.slider("Top K Results", 3, 20, 5)

        col3, col4 = st.columns(2)
        with col3:
            index_selector = st.selectbox("🔵 Primary Index Type", ['FlatL2', 'FlatIP', 'IVF', 'HNSW'], index=2)
        with col4:
            compare_selector = st.selectbox("🟢 Compare With", ['FlatL2', 'FlatIP', 'IVF', 'HNSW', 'All Indices'], index=0)

        search_clicked = st.button("🔍 Search Products", type="primary")

        if search_clicked:
            if not query.strip():
                st.warning("⚠️ Please enter a search query")
            else:
                with st.spinner("Searching..."):
                    results_df, fig, logs, metrics = search_products(query, top_k, index_selector, compare_selector)

                st.success(f"✅ Found {top_k} products | Recall: {metrics['recall']:.3f} | Precision: {metrics['precision']:.3f}")

                render_performance_card(index_selector, metrics['build_time'], metrics['search_time'])
                render_eval_metrics(top_k, metrics['recall'], metrics['precision'], metrics['f1'], metrics['map'])

                st.markdown("<h3 style='margin-top:24px;'>📦 Matched Products</h3>", unsafe_allow_html=True)
                render_product_cards(results_df)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📝 Execution Logs"):
                    for log in logs:
                        st.markdown(f"<p style='font-family:monospace; font-size:13px; color:#cfcfe8;'>[{datetime.now().strftime('%H:%M:%S')}] {log}</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload a CSV file to get started")

with tab2:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
        <h3 style="color:#ffffff; margin:0 0 8px 0;">🧪 Comprehensive Benchmark Testing</h3>
        <p style="color:#e5e5f0; margin:0; font-size:14px;">
            This will test FlatL2 vs IVF performance across multiple dataset sizes (4.5K → 200K products)
            and evaluate different IVF configurations with accurate Recall and Precision metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.embeddings is None:
        st.warning("⚠️ Please upload a dataset first (in the Search Products tab)")
    else:
        if st.button("🚀 Run Full Benchmark", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            summary, fig_build, fig_search, comparison_df, ivf_config_df = run_benchmark(progress_bar, status_text)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 24px 28px; border-radius: 14px; margin: 14px 0;
                        box-shadow: 0 8px 22px rgba(102,126,234,0.4);">
                <h3 style="margin:0 0 10px 0; color:#ffffff;">🎉 Benchmark Complete — Real FAISS Results</h3>
                <p style="margin:6px 0; color:#f1f1f6; font-size:14px;"><strong>Max IVF Speedup:</strong> {summary['max_speedup']:.2f}x faster than FlatL2</p>
                <p style="margin:6px 0; color:#f1f1f6; font-size:14px;"><strong>Best IVF Configuration:</strong> nlist={summary['best_nlist']}, nprobe={summary['best_nprobe']}</p>
                <p style="margin:6px 0; color:#f1f1f6; font-size:14px;"><strong>Best Recall@10:</strong> {summary['best_recall']:.4f} ({summary['best_recall']*100:.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_build, use_container_width=True)
            with col2:
                st.plotly_chart(fig_search, use_container_width=True)

            st.markdown("<h3>📊 Performance Comparison: FlatL2 vs IVF</h3>", unsafe_allow_html=True)
            st.dataframe(comparison_df, use_container_width=True)

            st.markdown("<h3>⚙️ IVF Configuration Analysis</h3>", unsafe_allow_html=True)
            st.dataframe(ivf_config_df, use_container_width=True)

with tab3:
    st.markdown("""
    <div style="color:#f1f1f6;">

    ## 🔍 FAISS Product Matching System

    ### ✨ Key Features
    | Feature | Description |
    |---------|-------------|
    | CSV Upload | Fast upload and processing with optimized batch encoding |
    | Real FAISS Search | Actual FAISS indices (FlatL2, FlatIP, IVF, HNSW) |
    | Accurate Metrics | Correct Recall@K, Precision@K, F1 Score, MAP calculations |
    | Visual Comparison | Interactive charts comparing selected indices |
    | Performance Analysis | Build time and search time comparisons |
    | IVF Tuning | Test different nlist/nprobe configurations |

    ---

    ### 🚀 How to Use
    **Step 1: Upload Dataset** — Go to Search Products tab, upload your CSV, wait for processing.

    **Step 2: Search Products** — Enter a search query, adjust Top K, select index type & comparison, click Search.

    **Step 3: Analyze Results** — Review Recall, Precision, F1, MAP, and the performance comparison chart.

    **Step 4: Run Benchmark** (optional) — Go to Benchmark Analysis tab and test across dataset sizes.

    ---

    ### 📋 CSV Format Requirements
    ```csv
    product_id,name,category,price,discount
    1,Wireless Headphones,Electronics,99.99,10%
    2,Red Cotton Shirt,Clothing,29.99,15%
    3,Gaming Laptop 16GB,Computers,1299.99,5%
    ```
    - **Best practice:** include a `name` or `product_name` column
    - **Alternative:** any text columns will be combined
    - **Performance:** works best with 1K–200K products

    ---

    ### 🧠 Index Types Explained
    | Index | Description | Best For | Speed | Accuracy |
    |-------|-------------|----------|-------|----------|
    | **FlatL2** | Exact L2 distance search | Small datasets, highest accuracy | Slow | 100% |
    | **FlatIP** | Exact cosine similarity | Small datasets, semantic search | Slow | 100% |
    | **IVF** | Inverted File Index | Large datasets, fast approximate | Fast | 95–99% |
    | **HNSW** | Hierarchical graph | Balanced speed/accuracy | Very Fast | 97–99% |

    ---

    ### 📐 Metrics Explained
    - **Recall@K**: fraction of ground-truth items retrieved — `(Retrieved ∩ Ground Truth) / K`
    - **Precision@K**: fraction of retrieved items that are correct — `(Retrieved ∩ Ground Truth) / Retrieved`
    - **F1 Score**: harmonic mean of Precision and Recall
    - **MAP@K**: Mean Average Precision — considers ranking order of results

    ---

    ### 💡 Performance Tips
    - **Small datasets** (< 10K): use FlatL2 or FlatIP
    - **Medium datasets** (10K–100K): use IVF or HNSW
    - **Large datasets** (> 100K): use IVF with optimized config
    - **Best accuracy**: FlatIP (exact cosine similarity)
    - **Best speed**: HNSW (very fast approximate)
    - **Balanced**: IVF with nlist=200, nprobe=20

    </div>
    """, unsafe_allow_html=True)