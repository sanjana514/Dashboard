"""
ELITE APRIORI - 100% PERFECT + FIXED
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="Apriori Mining Dashboard", page_icon="✨", layout="wide")

# CSS + FOOTER + BOXES FIXED
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #f8f9fa; }
/* Warning block override */
    [data-testid="stWarningBlock"] {
    background-color: #ffcccc !important;
    color: white !important;
    border: 1px solid #cc0000;
    border-radius: 5px;
    padding: 10px;
}

/* More specific override */
[data-testid="stWarningBlock"] > div {
    background-color: #ffcccc !important;
    color: white !important;
}

    /* হেডার */
    .main-header {
        background:linear-gradient(135deg, #C8A2C8, #87CEEB);
        padding: 1.3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem auto 1.5rem;
        max-width: 90%;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.4);
    }
    .main-header h1 { 
        color: white; 
        font-size: 2.3rem; 
        margin: 0; 
        font-weight: 800; 
    }

    /* সাইডবার */
    [data-testid="stSidebar"] {
        background: #1e293b !important;
    }
    [data-testid="stSidebar"] * { color: #f1f5f9 !important; }

 /* Slider track (the line) */
.stSlider [data-testid="stSliderTrack"] {
    background: linear-gradient(90deg, #a855f7, #ec4899) !important;  /* <== track colors */
    height: 5px !important;
    border-radius: 4px !important;
}
/* Slider thumb (the circle) */
.stSlider [data-testid="stThumbValue"] + div > div {
    background: #8b5cf6 !important;      /* <== thumb color */
    border: 3px solid #7c3aed !important;/* <== thumb border color */
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(139, 92, 246, 0.5) !important; /* <== glow color */
}
    /* Value নিচে নামানো */
    .stSlider [data-baseweb="slider"] + div {
        margin-top: 50px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* বাটন */
    .stButton > button {
        background: linear-gradient(135deg, #14b8a6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(139, 92, 246, 0.5) !important;
    }

    /* ৪টা বক্স — লেখা কালো + বড় */
    @keyframes float {
        0% { transform: translateY(0px); }
        100% { transform: translateY(-8px); }
    }
    .info-box {
        animation: float 3s ease-in-out infinite alternate;
        background: white;
        padding: 1.3rem;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 190px;
        border-left: 6px solid;
        text-align: left;
        transition: all 0.3s;
    }
    .info-box:hover {
        animation-play-state: paused;
        transform: translateY(-12px) !important;
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.3) !important;
    }
    .info-box h3 {
        font-size: 1.15rem !important;
        margin: 0 0 0.7rem 0 !important;
        color: #1e293b !important;
        font-weight: 800 !important;
    }
    .info-box p {
        font-size: 0.88rem !important;
        line-height: 1.55 !important;
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* নতুন মেট্রিক বক্স - font size বড় */
   /* নতুন মেট্রিক বক্স - box ছোট এবং value কেন্দ্রে */
.metric-box {
    background:#BFA2DB;
    padding: 0.3rem 0.5rem;
    border-radius: 8px;
    text-align: center;
    color: white;
    box-shadow: 0 3px 10px rgba(139, 92, 246, 0.3);
    display: flex;
    flex-direction: column;
    justify-content: center;   /* ✅ vertical center */
    align-items: center;       /* ✅ horizontal center */
    height: 60px;
    width: 80%;
    line-height: 1;            /* ✅ line spacing ঠিক করা */
}

.metric-box h4 {
    margin: 0;                 /* ✅ extra margin বাদ দাও */
    padding: 0;                /* ✅ padding বাদ দাও */
    font-size: 0.8rem;
    opacity: 0.9;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-box p {
    margin: 0;                 /* ✅ extra margin বাদ দাও */
    padding: 0;                /* ✅ padding বাদ দাও */
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
}

    /* ফুটার — white background */
    .elite-footer {
        margin-top: 5rem;
        padding: 3rem 1rem;
        background: white;
        border-top: 5px solid #14b8a6;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
    }
    .elite-footer::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 50% 0%, rgba(20, 184, 166, 0.05), transparent 70%);
        animation: pulse 6s infinite;
    }
    .footer-content h3 {
        color: #14b8a6;
        font-size: 1.9rem;
        margin: 0 0 0.5rem;
        font-weight: 800;
        text-shadow: 0 0 12px rgba(20, 184, 166, 0.3);
    }
    .footer-content p {
        color: #475569;
        font-size: 0.95rem;
        margin: 0.4rem 0;
        font-weight: 500;
    }
    .copyright {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 1rem;
        font-weight: 600;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    
    /* Graph Box */
    .graph-box {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .graph-box h4 {
        font-size: 0.9rem;
        color: #1e293b;
        margin: 0 0 0.5rem 0;
        font-weight: 700;
    }
    
    /* Tab buttons স্টাইল */
    .stTabs [data-baseweb="tab-list"] button {
        background: linear-gradient(135deg, #14b8a6, #8b5cf6);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #0d9488, #7c3aed);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }
    
    /* Info banner - dark background */
    .info-banner {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-left: 4px solid #8b5cf6;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #f1f5f9;
        font-weight: 600;
    }
    
    /* Section heading - dark color */
    .section-heading {
        color: #1e293b !important;
        font-size: 1.3rem !important;
        font-weight: 800 !important;
        margin: 0.5rem 0 1rem 0 !important;
    }
    
 .stat-box {
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    width: 150px;
    height: 80px;
    border-radius: 10px;
    color: white;
    margin-bottom: 0.5rem;
    box-shadow: 0 3px 8px rgba(236, 72, 153, 0.3);

    display: flex;
    flex-direction: column;
    justify-content: center; /* vertical center */
    align-items: center;     /* horizontal center */
    padding: 0;              /* padding বাদ দিয়ে compact */
    gap: 0;                  /* h5-p gap একদম 0 */
}

.stat-box h5 {
    font-size: 0.9rem;
    margin: 0;               /* কোন extra space নেই */
    line-height: 1;          /* compact line spacing */
}

.stat-box p {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0;               /* কোন extra space নেই */
    line-height: 1;          /* compact line spacing */
}



</style>
""", unsafe_allow_html=True)

# Session
for k in ['transactions', 'freq', 'min_support', 'max_len', 'te']:
    if k not in st.session_state: st.session_state[k] = None

# হেডার
st.markdown("""
<div class="main-header">
    <h1>Elite Apriori Mining Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# সাইডবার
with st.sidebar:
    st.markdown("## Control Panel")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    st.markdown("### Parameters")
    min_support = st.slider("Min Support", 0.001, 0.5, 0.02, 0.001)
    top_n = st.slider("Top N", 5, 50, 15, 5)
    max_len = st.number_input("Max Pattern Size", 1, 10, 3)
    
    st.markdown("### Graph Color Theme")
    color_theme = st.selectbox("Choose", 
        ["viridis", "plasma", "inferno", "magma", "cividis", "teal", "purple", "rainbow"])
    
    run = st.button("RUN ANALYSIS", use_container_width=True)

# ওয়েলকাম স্ক্রিন
if not uploaded_file:
    st.markdown("## Welcome to Apriori Elite!")
    c1, c2, c3, c4 = st.columns(4)
    boxes = [
        ("Getting Started", "1. Upload CSV<br>2. Set parameters<br>3. Click RUN<br>4. Explore!", "#14b8a6"),
        ("Features", "• Real-time mining<br>• Color themes<br>• Export CSV<br>• Mobile ready", "#8b5cf6"),
        ("CSV Format", "T1 Milk Bread<br>T2 Bread Butter<br>Space separated", "#06b6d4"),
        ("Best Settings", "Support: 0.02<br>Max Size: 3<br>Top N: 15", "#8b5cf6")
    ]
    for col, (t, c, colr) in zip([c1,c2,c3,c4], boxes):
        with col:
            st.markdown(f"""
            <div class="info-box" style="border-left-color:{colr}">
                <h3>{t}</h3>
                <p>{c}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ফুটার হোমপেজে
    st.markdown("""
    <div class="elite-footer">
        <div class="footer-content">
            <h3>Elite Apriori Dashboard</h3>
            <p>Powered by <strong>Streamlit</strong> • Built with <strong>Love</strong></p>
            <p class="copyright">© 2025 • All Rights Reserved • Made in Bangladesh</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# লোড CSV
try:
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    transactions = []
    for l in lines:
        p = l.split()
        if len(p) > 1:
            transactions.append(p[1:] if p[0].startswith("T") else p)
    if not transactions:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        for _, r in df.iterrows():
            trans = [str(c) for c, v in r.items() if v in [1, True]]
            if trans: transactions.append(trans)
    st.session_state.transactions = transactions
    st.success(f"✅ Loaded {len(transactions)} transactions")
except Exception as e:
    st.error(f"❌ Invalid CSV format: {str(e)}")
    st.stop()

# রান - parameters update করলেই re-run
if run or (st.session_state.freq is not None and 
          (st.session_state.min_support != min_support or st.session_state.max_len != max_len)):
    with st.spinner(" Mining patterns..."):
        try:
            te = TransactionEncoder()
            X = te.fit(transactions).transform(transactions)
            df_bin = pd.DataFrame(X, columns=te.columns_)
            freq = apriori(df_bin, min_support=min_support, use_colnames=True, max_len=max_len)
            
            if freq.empty:
              st.markdown(
              """
              <div style="background-color: #ff9999; color: white; border: 0px solid #cc0000; border-radius: 5px; padding: 10px; text-align: center;">
               ⚠️ No frequent patterns found! Try lowering min support.
              </div>
              """,
              unsafe_allow_html=True
           )
              st.stop()
            
            freq['length'] = freq['itemsets'].apply(len)
            freq['count'] = (freq['support'] * len(transactions)).astype(int)
            freq['support_pct'] = (freq['support'] * 100).round(2)
            freq['items'] = freq['itemsets'].apply(lambda x: ', '.join(sorted(x)))
            
            st.session_state.freq = freq
            st.session_state.min_support = min_support
            st.session_state.max_len = max_len
            st.session_state.te = te
            if run:
                st.balloons()
        except Exception as e:
            st.error(f"❌ Mining failed: {str(e)}")
            st.stop()

# রেজাল্ট
if st.session_state.freq is not None:
    df = st.session_state.freq
    total = len(st.session_state.transactions)
    te = st.session_state.te

    # মেট্রিক - font size বড়
    m1, m2, m3, m4 = st.columns(4)
    for col, (l, v) in zip([m1, m2, m3, m4], [
      ("Patterns", len(df)),
      ("Avg Support", f"{df['support_pct'].mean():.1f}%"),
      (" Items ", len(te.columns_)),
      ("Transactions", total)
   ]):
      with col:
        st.markdown(f"""
        <div class="metric-box">
            <h4>{l}</h4>
            <p>{v}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ট্যাব
    t1, t2, t3, t4 = st.tabs([" Pattern Table", " Graph", " Details", " Export"])

    with t1:
        # Info banner - dark background
        st.markdown(f"""
        <div class="info-banner">
             <strong>Current Parameters:</strong> Max Pattern Size = {st.session_state.max_len} | Min Support = {st.session_state.min_support}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h3 class="section-heading">Frequent Pattern Itemsets</h3>', unsafe_allow_html=True)
        top = df.nlargest(top_n, 'support_pct')[['items', 'support_pct', 'count', 'length']]
        top.columns = ['Pattern', 'Support %', 'Count', 'Size']
        top.index = range(1, len(top)+1)
        st.dataframe(top, use_container_width=True, height=350)

    with t2:
        st.markdown('<h3 class="section-heading"> Pattern Visualizations</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="graph-box"><h4> Top 10 Patterns by Support</h4></div>', unsafe_allow_html=True)
            fig1 = px.bar(df.nlargest(10, 'support_pct'), 
                         x='support_pct', y='items',
                         orientation='h', 
                         color='support_pct',
                         color_continuous_scale=color_theme,
                         labels={'support_pct': 'Support %', 'items': 'Pattern'})
            fig1.update_layout(height=320, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown('<div class="graph-box"><h4> Pattern Size Distribution</h4></div>', unsafe_allow_html=True)
            size_dist = df['length'].value_counts().sort_index().reset_index()
            size_dist.columns = ['Size', 'Count']
            fig2 = px.bar(size_dist, x='Size', y='Count', 
                         color='Count',
                         color_continuous_scale=color_theme)
            fig2.update_layout(height=320, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown('<div class="graph-box"><h4> Support Distribution</h4></div>', unsafe_allow_html=True)
            fig3 = px.histogram(df, x='support_pct', nbins=20,
                               color_discrete_sequence=['#8b5cf6'])
            fig3.update_layout(height=320, showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                              xaxis_title="Support %", yaxis_title="Frequency")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            st.markdown('<div class="graph-box"><h4> Transaction Count vs Support</h4></div>', unsafe_allow_html=True)
            fig4 = px.scatter(df.nlargest(20, 'support_pct'), 
                             x='support_pct', y='count',
                             size='length', color='length',
                             color_continuous_scale=color_theme,
                             labels={'support_pct': 'Support %', 'count': 'Count'})
            fig4.update_layout(height=320, showlegend=False, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig4, use_container_width=True)

    with t3:
        st.markdown('<h3 class="section-heading"> Complete Details Analysis</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-banner">
            <strong>Mining Summary:</strong> {len(df)} Patterns Found | Min Support = {st.session_state.min_support} | Max Size = {st.session_state.max_len}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<h4 class="section-heading" style="font-size:1.1rem !important;"> All Frequent Itemsets</h4>', unsafe_allow_html=True)
            details_df = df[['items', 'support_pct', 'count', 'length']].sort_values('support_pct', ascending=False)
            details_df.columns = ['Pattern', 'Support %', 'Count', 'Size']
            details_df.index = range(1, len(details_df)+1)
            st.dataframe(details_df, use_container_width=True, height=350)
        
        with col2:
            st.markdown('<h4 class="section-heading" style="font-size:1.2rem !important;"> Statistics</h4>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-box">
                <h5>Max Support</h5>
                <p>{df['support_pct'].max():.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(135deg, #8b5cf6, #06b6d4);">
                <h5>Min Support</h5>
                <p>{df['support_pct'].min():.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(135deg, #ec4899, #f59e0b);">
                <h5>Avg Pattern Size</h5>
                <p>{df['length'].mean():.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-box" style="background: linear-gradient(135deg, #8b5cf6, #ec4899);">
                <h5>Most Frequent Size</h5>
                <p>{int(df['length'].mode()[0])}</p>
            </div>
            """, unsafe_allow_html=True)

    with t4:
        st.markdown('<h3 class="section-heading"> Export Results</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-banner">
             <strong>Download your mining results</strong> - Total {len(df)} patterns ready to export
        </div>
        """, unsafe_allow_html=True)
        
        export_df = df[['items', 'support_pct', 'count', 'length']].sort_values('support_pct', ascending=False)
        export_df.columns = ['Pattern', 'Support %', 'Count', 'Size']
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="apriori_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown('<h4 class="section-heading" style="font-size:1.1rem !important;"> Preview (Top 10)</h4>', unsafe_allow_html=True)
        st.dataframe(export_df.head(10), use_container_width=True, height=300)

    # ফুটার রেজাল্ট পেজে
    st.markdown("""
    <div class="elite-footer">
        <div class="footer-content">
            <h3>Elite Apriori Dashboard</h3>
            <p>Powered by <strong>Streamlit</strong> • Built with <strong>Love</strong></p>
            <p class="copyright">© 2025 • All Rights Reserved • Made in Bangladesh</p>
        </div>
    </div>
    """, unsafe_allow_html=True)