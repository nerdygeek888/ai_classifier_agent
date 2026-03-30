"""
RPA + AI Hybrid Automation — Customer Query Processor
Author: Sushma Nadkarni
Description: Simulates an enterprise automation pipeline that:
             1. Reads customer queries from CSV (RPA-style ingestion)
             2. Classifies them using a HuggingFace NLP model (AI layer)
             3. Routes to appropriate handlers (automation logic)
             4. Generates a summary report (RPA-style output)
             This hybrid pattern mirrors real-world AA + NLP integrations.
"""

import streamlit as st
import pandas as pd
from processor import QueryProcessor
from report_generator import generate_html_report
import tempfile, os

st.set_page_config(page_title="RPA + AI Query Processor", page_icon="⚙️", layout="wide")

st.title("⚙️ RPA + AI Hybrid Customer Query Processor")
st.markdown(
    "Simulates an enterprise **RPA + AI** automation pipeline: "
    "ingest queries → classify with NLP → route to handlers → generate report. "
    "Mirrors patterns used in **Automation Anywhere + IQ Bot / NLP integrations**."
)

# ── Session state ─────────────────────────────────────────────────────────────
if "processor" not in st.session_state:
    with st.spinner("Loading NLP classification model..."):
        st.session_state.processor = QueryProcessor()
if "results_df" not in st.session_state:
    st.session_state.results_df = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    st.markdown("**Pipeline Steps:**")
    st.markdown("1. 📥 Ingest queries from CSV")
    st.markdown("2. 🤖 NLP Classification (HuggingFace)")
    st.markdown("3. 🔀 Rule-based routing")
    st.markdown("4. ✅ Handler processing")
    st.markdown("5. 📊 Report generation")
    st.divider()
    st.markdown("**Classification Categories:**")
    st.markdown("- 💳 `billing` — Billing/payment queries")
    st.markdown("- 🔧 `technical_support` — Tech issues")
    st.markdown("- 📦 `order_status` — Order enquiries")
    st.markdown("- 🔄 `refund` — Refund requests")
    st.markdown("- ℹ️ `general_enquiry` — Other")
    st.divider()
    confidence_threshold = st.slider("Confidence threshold for auto-processing", 0.5, 0.95, 0.75)
    st.caption("Queries below threshold are flagged for human review")

# ── Main ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📥 Input", "⚙️ Processing", "📊 Report"])

with tab1:
    st.subheader("Input: Load Customer Queries")
    input_mode = st.radio("Choose input method:", ["Use sample data", "Upload CSV"])

    if input_mode == "Use sample data":
        sample_df = st.session_state.processor.get_sample_queries()
        st.dataframe(sample_df, use_container_width=True)
        input_df = sample_df

    else:
        uploaded = st.file_uploader("Upload CSV with columns: query_id, customer_name, query_text", type=["csv"])
        if uploaded:
            input_df = pd.read_csv(uploaded)
            st.dataframe(input_df, use_container_width=True)
        else:
            st.info("Upload a CSV file or switch to sample data.")
            input_df = None

    if st.button("▶️ Run Pipeline", type="primary"):
        if input_df is not None:
            st.session_state.run_input = input_df
            st.session_state.run_threshold = confidence_threshold
            st.success("Pipeline queued! Switch to the ⚙️ Processing tab.")
        else:
            st.warning("Please load data first.")

with tab2:
    st.subheader("Pipeline Execution")
    if hasattr(st.session_state, "run_input") and st.session_state.results_df is None:
        progress = st.progress(0, text="Starting pipeline...")
        status_box = st.empty()
        log_box = st.empty()
        logs = []

        def update_log(msg):
            logs.append(msg)
            log_box.code("\n".join(logs[-10:]))

        with st.spinner("Processing..."):
            update_log("🔄 Step 1: Ingesting queries...")
            progress.progress(10, "Step 1: Ingesting queries")

            update_log(f"📊 Loaded {len(st.session_state.run_input)} queries")
            progress.progress(30, "Step 2: Running NLP classification")
            update_log("🤖 Step 2: Classifying with HuggingFace NLP model...")

            results = st.session_state.processor.process(
                st.session_state.run_input,
                confidence_threshold=st.session_state.run_threshold
            )
            progress.progress(70, "Step 3: Routing and processing")
            update_log("🔀 Step 3: Routing to handlers...")

            st.session_state.results_df = results
            progress.progress(100, "✅ Pipeline complete")
            update_log("✅ Pipeline complete! Switch to 📊 Report tab.")
            status_box.success("Pipeline completed successfully!")

    elif st.session_state.results_df is not None:
        st.success("✅ Pipeline already run. View results in the 📊 Report tab.")
        if st.button("🔄 Re-run pipeline"):
            st.session_state.results_df = None
            if hasattr(st.session_state, "run_input"):
                del st.session_state.run_input
            st.rerun()
    else:
        st.info("👈 Load data in the Input tab and click Run Pipeline.")

with tab3:
    st.subheader("Processing Report")
    if st.session_state.results_df is not None:
        df = st.session_state.results_df

        # KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", len(df))
        with col2:
            auto = len(df[df["status"] == "processed"])
            st.metric("Auto-Processed", auto)
        with col3:
            flagged = len(df[df["status"] == "flagged_for_review"])
            st.metric("Flagged for Review", flagged)
        with col4:
            avg_conf = df["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.0%}")

        # Category breakdown
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Classification Breakdown**")
            cat_counts = df.groupby("category")["query_id"].count().reset_index()
            cat_counts.columns = ["Category", "Count"]
            st.dataframe(cat_counts, use_container_width=True)
        with col2:
            st.markdown("**Status Breakdown**")
            status_counts = df.groupby("status")["query_id"].count().reset_index()
            status_counts.columns = ["Status", "Count"]
            st.dataframe(status_counts, use_container_width=True)

        # Full results
        st.divider()
        st.markdown("**Full Processing Results**")
        st.dataframe(
            df[["query_id", "customer_name", "query_text", "category", "confidence", "status", "handler_response"]],
            use_container_width=True
        )

        # Download
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download Results CSV", csv, "processing_results.csv", "text/csv")
    else:
        st.info("Run the pipeline first to see the report.")
