import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Enterprise Neural Matrix Node", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# High-Fidelity Cyber Ops Custom Theme Stylesheet
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #0a0f24 0%, #030712 100%); color: #f1f5f9; }
    div[data-testid="stMetricValue"] { color: #00f2fe !important; font-family: 'Courier New', monospace; font-weight: 800; font-size: 2.2rem !important; text-shadow: 0 0 10px rgba(0, 242, 254, 0.4); }
    div[data-testid="stMetricLabel"] { color: #64748b !important; letter-spacing: 1.5px; text-transform: uppercase; font-size: 0.75rem !important; font-weight: 600; }
    .stTextInput>div>div>input { background-color: #020617; color: #38bdf8; border: 1px solid #1e293b; border-radius: 8px; font-family: monospace; padding: 12px; }
    .stTextInput>div>div>input:focus { border-color: #00f2fe !important; box-shadow: 0 0 12px rgba(0, 242, 254, 0.2); }
    .sources-box { background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); padding: 16px; border-radius: 8px; border-left: 4px solid #38bdf8; box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-top: 15px; }
    .stButton>button { background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); color: #020617; font-weight: bold; border: none; border-radius: 8px; padding: 10px 24px; transition: all 0.3s ease; }
    .stButton>button:hover { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4); color: #020617; }
    div[data-testid="stExpander"] { background-color: #090d1a; border: 1px solid #1e293b; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Distributed Neural-RAG Cluster Engine")
st.caption("Core Node Configuration Pipeline: Multi-Stage Hybrid Retrieval & Production Telemetry Diagnostics Framework")

API_URL = "http://127.0.0.1:8000"

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "node_user_alpha"

# Sidebar Control Hub
with st.sidebar:
    st.markdown("### 🏢 Cluster Operations Control")
    st.markdown("---")
    
    st.markdown("#### 📥 Knowledge Base Pipeline")
    uploaded_files = st.file_uploader("Upload Cluster Documentation Nodes:", accept_multiple_files=True, type=['pdf'])
    if uploaded_files:
        for file in uploaded_files:
            files = {"file": (file.name, file.getvalue(), "application/pdf")}
            requests.post(f"{API_URL}/api/v5/upload", files=files)
        st.success("IO Buffer Sync Finalized.")

    if st.button("🔥 Align Matrix Graph Clusters", use_container_width=True):
        with st.spinner("Compiling structural vector spaces..."):
            res = requests.post(f"{API_URL}/api/v5/ingest").json()
            st.success(f"✓ Sync Confirmed. Active Elements: {res.get('total_chunks')}")
            
    st.markdown("---")
    st.markdown("#### ⚙️ Runtime Metrics")
    st.text("Node Cluster: Node-01_A_Amity")
    st.text("Security Policy: Real-time Guardrails")
    st.text("State Persistence: SQLite Framework")

# Core Screen Layout
workspace_panel, analytics_panel = st.columns([3, 2])

with workspace_panel:
    st.markdown("### 🔍 Execution Terminal Input")
    user_query = st.text_input("Inject target query string directly into distributed vector graphs:", placeholder="e.g., Explain self-attention layer dimensions...")

    if user_query:
        with st.spinner("Streaming data transaction from vector matrix pools..."):
            try:
                res = requests.post(f"{API_URL}/api/v5/query", json={"question": user_query, "session_id": st.session_state["session_id"]})
                if res.status_code == 200:
                    data = res.json()
                    
                    st.markdown("#### 💡 Synthesized Context Response Frame")
                    st.write(data["answer"])
                    
                    if data.get("sources"):
                        st.markdown('<div class="sources-box"><strong>🛡️ Audit Trail Proof (Verified Pages):</strong><br>' + ', '.join(data['sources']) + '</div>', unsafe_allow_html=True)
                    
                    if "⚠️ TRANSACTION TERMINATED" not in data["answer"]:
                        f_col1, f_col2, _ = st.columns([1, 1, 8])
                        with f_col1:
                            if st.button("👍", key="up"):
                                requests.post(f"{API_URL}/api/v5/feedback", json={"query": user_query, "answer": data["answer"], "rating": "POSITIVE"})
                                st.toast("Alignment parameter recorded.")
                        with f_col2:
                            if st.button("👎", key="down"):
                                requests.post(f"{API_URL}/api/v5/feedback", json={"query": user_query, "answer": data["answer"], "rating": "NEGATIVE"})
                                st.toast("Anomalous error vector recorded.")

                    st.session_state["latest_metrics"] = data.get("metrics", {})
                    st.session_state["chart_data"] = data.get("chart_data", [])
                else:
                    st.error(res.json()['detail'])
            except Exception:
                st.error("Cluster Gateway Timed Out. Verify connection string state.")

with analytics_panel:
    st.markdown("### 📊 Live Operations Telemetry")
    
    if "latest_metrics" in st.session_state and st.session_state["latest_metrics"]:
        metrics = st.session_state["latest_metrics"]
        
        # Grid Telemetry Frame Dashboard Blocks
        m_col1, m_col2 = st.columns(2)
        with m_col1: st.metric(label="⏱️ Node Latency", value=f"{metrics.get('latency_sec')} s")
        with m_col2: st.metric(label="🎯 Max Neural Match", value=metrics.get('confidence_score'))
        
        m_col3, m_col4 = st.columns(2)
        with m_col3: st.metric(label="📊 Precision Accuracy", value=f"{int(metrics.get('faithfulness', 0)*100)} %")
        with m_col4: st.metric(label="🛡️ Firewall State", value=metrics.get('security_state'))
        
        st.markdown("---")
        st.markdown("#### 📈 Neural Reranking Space Distribution")
        chart_data = st.session_state.get("chart_data", [])
        if chart_data:
            df = pd.DataFrame(chart_data)
            
            # 🚀 ULTIMATE GRAPH CONFIGURATION: Enterprise Area Gradient Plot with Cyber-Blue styling
            fig = px.area(
                df, x="Rank", y="Neural_Weight", text="Source_Context",
                title="Cross-Encoder Mathematical Decay Curves",
                labels={"Neural_Weight": "Confidence Distribution Scale", "Rank": "Sorted Reranker Execution"},
                template="plotly_dark"
            )
            
            # Adding industry level glow effects to indicators and area vectors
            fig.update_traces(
                line_color="#00f2fe", 
                mode="lines+markers+text", 
                textposition="top center",
                fill="tozeroy",
                fillcolor="rgba(0, 242, 254, 0.15)"
            )
            
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", 
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#1e293b", tickmode="linear"),
                yaxis=dict(showgrid=True, gridcolor="#1e293b")
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Telemetry buffers are idle. Awaiting operational cluster loop transaction metrics.")
