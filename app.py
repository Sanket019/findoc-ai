import os
from io import BytesIO
import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import pdfplumber
import pikepdf
import plotly.express as px
from dotenv import load_dotenv

from ingest import (
    process_and_store_text,
    parse_csv_statement,
    extract_transactions,
    safe_print,
)
from rag_chain import get_answer, answer_analytical_question

load_dotenv()

if not os.getenv("GROQ_API_KEY") and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="FinDoc AI — Bank Statement Analyser",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = "none"
if "transactions_df" not in st.session_state:
    st.session_state["transactions_df"] = None
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "active_dataset_label" not in st.session_state:
    st.session_state["active_dataset_label"] = ""
if "processed_file_key" not in st.session_state:
    st.session_state["processed_file_key"] = None
if "validation_summary" not in st.session_state:
    st.session_state["validation_summary"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown("""
<style>

:root {
    --accent: #00D4AA;
    --accent-dark: #00a888;
    --bg-dark: #0f1117;
    --card-bg: #1a1f2e;
    --card-border: #2d3748;
}

div[data-testid="metric-container"] {
    background: #1a1f2e !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #00D4AA !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #2d3748 !important;
}

div[data-testid="stButton"] > button {
    border-radius: 6px !important;
    border: 1px solid #2d3748 !important;
    background: #1a1f2e !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #00D4AA !important;
    color: #00D4AA !important;
    background: #142824 !important;
}

h1, h2, h3, h4 {
    color: #f8fafc !important;
    font-weight: 600 !important;
}
p, span, label {
    color: #cbd5e1;
}

div[data-testid="stRadio"] > div {
    gap: 12px;
}
</style>
""", unsafe_allow_html=True)

def fmt_inr(amount: float) -> str:

    return f"Rs.{amount:,.2f}"

def clear_active_dataset():

    st.session_state["data_mode"] = "none"
    st.session_state["transactions_df"] = None
    st.session_state["vectorstore"] = None
    st.session_state["active_dataset_label"] = ""
    st.session_state["processed_file_key"] = None
    st.session_state["validation_summary"] = None
    st.session_state["messages"] = []
    st.session_state.pop("statement_uploader", None)

def load_sample_dataset():

    demo_csv_path = "data/demo_bank_statement.csv"
    if not os.path.exists(demo_csv_path):
        demo_csv_path = "sample_data/sample_statement.csv"

    df, summary = parse_csv_statement(demo_csv_path)

    with open(demo_csv_path, "r", encoding="utf-8") as f:
        demo_raw_text = f.read()

    rag_context_header = (
        "BANK STATEMENT SUMMARY:\n"
        "Account Holder: Sanket Sharma\n"
        "Account Number: XXXX-XXXX-4521\n"
        "Bank: State Bank of India (Koramangala Branch)\n"
        "IFSC Code: SBIN0012345\n"
        "Statement Period: May 1, 2026 to July 31, 2026\n"
        "Opening Balance: Rs.15,000.00\n"
        "Closing Balance: Rs.103,758.10\n\n"
    )
    vectorstore = process_and_store_text(
        rag_context_header + demo_raw_text,
        source_name="demo_bank_statement.csv",
        session_id="sample_demo_session",
    )

    st.session_state["data_mode"] = "sample"
    st.session_state["transactions_df"] = df
    st.session_state["vectorstore"] = vectorstore
    st.session_state["validation_summary"] = summary
    st.session_state["active_dataset_label"] = "Sample Bank Statement (May – July 2026)"
    st.session_state["processed_file_key"] = "demo_bank_statement"

def extract_text_from_pdf(file) -> str:

    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return text

with st.sidebar:
    st.markdown("## 💰 FinDoc AI")
    st.caption("Bank Statement Analyser")
    st.divider()

    mode = st.session_state.get("data_mode", "none")
    if mode != "none" and st.session_state.get("transactions_df") is not None:
        st.markdown(f"**Active Data:** {st.session_state.get('active_dataset_label', 'Active')}")
        st.caption(f"Loaded {len(st.session_state['transactions_df'])} transactions")
        if st.button("🔄 Reset / Clear Dataset", use_container_width=True):
            clear_active_dataset()
            st.rerun()

st.markdown("# FinDoc AI")
st.markdown("### *Your bank statements, easier to understand.*")
st.write(
    "A smart financial assistant that combines **exact Pandas analytics** with **AI document intelligence** "
    "to help you explore spending patterns, track cash flow, and ask questions about your bank statements."
)
st.divider()

col_mode1, col_mode2 = st.columns([1, 1])

with col_mode1:
    sample_active = st.session_state.get("data_mode") == "sample"
    st.markdown("#### 🚀 Option A: Try with Sample Data")
    st.caption("Test the dashboard and AI immediately with our fictional 3-month dataset. No upload needed.")
    if st.button("📊 Load Sample Statement", type="primary" if not sample_active else "secondary", use_container_width=True):
        load_sample_dataset()
        st.rerun()

with col_mode2:
    upload_active = st.session_state.get("data_mode") == "uploaded"
    st.markdown("#### 📂 Option B: Upload Your Own Statement")
    st.caption("Upload your own CSV or digital PDF bank statement (supports SBI, HDFC, ICICI, etc.).")

st.markdown("")

show_uploader = (st.session_state.get("data_mode") != "sample") or st.checkbox("Show file upload box to switch to a personal statement", value=False)

if show_uploader:
    uploaded_file = st.file_uploader(
        "Drop your bank statement file here",
        type=["pdf", "csv"],
        help="Upload a digital PDF bank statement or CSV export (up to 15MB).",
        key="statement_uploader"
    )

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("processed_file_key") != file_key:
            clear_active_dataset()

            with st.status("Processing your statement...", expanded=True) as status_box:
                st.write("📄 **Stage 1:** Reading and verifying document format...")

                file_to_process = uploaded_file
                if file_type == "pdf":
                    try:
                        with pikepdf.open(uploaded_file) as _:
                            pass
                    except pikepdf.PasswordError:
                        pwd = st.text_input("🔐 Statement is password-protected. Enter password:", type="password")
                        if not pwd:
                            st.warning("Enter the password to decrypt the statement.")
                            st.stop()
                        try:
                            uploaded_file.seek(0)
                            with pikepdf.open(uploaded_file, password=pwd) as pdf:
                                buf = BytesIO()
                                pdf.save(buf)
                                buf.seek(0)
                                file_to_process = buf
                        except Exception as e:
                            st.error(f"Failed to decrypt PDF: {e}")
                            st.stop()
                    except Exception as e:
                        st.error(f"Cannot read PDF: {e}")
                        st.stop()

                st.write("⚡ **Stage 2:** Extracting financial transactions...")
                parsed_df = None
                val_summary = None

                if file_type == "csv":
                    parsed_df, val_summary = parse_csv_statement(uploaded_file)
                    uploaded_file.seek(0)
                    csv_raw_text = uploaded_file.read().decode("utf-8", errors="replace")
                    vectorstore = process_and_store_text(csv_raw_text, uploaded_file.name)
                else:

                    raw_pdf_text = extract_text_from_pdf(file_to_process)
                    if not raw_pdf_text.strip():
                        st.error("⚠️ No text could be extracted from this PDF. Scanned image documents require OCR.")
                        st.stop()

                    vectorstore = process_and_store_text(raw_pdf_text, uploaded_file.name)
                    parsed_df, debug_info = extract_transactions(raw_pdf_text)
                    val_summary = {
                        "total_rows_read": len(parsed_df),
                        "valid_transactions": len(parsed_df),
                        "invalid_rows_dropped": 0,
                        "duplicates_removed": 0,
                        "metadata": {},
                        "warnings": []
                    }

                st.write("🔍 **Stage 3:** Validating records and balance reconciliation...")
                if parsed_df is None or parsed_df.empty:
                    st.error("No valid transactions could be extracted. Please check the document format.")
                    st.stop()

                st.write("📊 **Stage 4:** Preparing financial overview...")
                st.session_state["data_mode"] = "uploaded"
                st.session_state["transactions_df"] = parsed_df
                st.session_state["vectorstore"] = vectorstore
                st.session_state["validation_summary"] = val_summary
                st.session_state["active_dataset_label"] = uploaded_file.name
                st.session_state["processed_file_key"] = file_key

                status_box.update(
                    label=f"Statement processed successfully! ({len(parsed_df)} transactions loaded)",
                    state="complete",
                    expanded=False
                )
            st.rerun()

transactions_df = st.session_state.get("transactions_df")

if transactions_df is not None and not transactions_df.empty:
    st.divider()

    active_name = st.session_state.get("active_dataset_label", "Statement")
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(f"### 📊 Financial Overview: `{active_name}`")
    with col_hdr2:
        if st.session_state.get("data_mode") == "sample":
            st.success("Mode: Sample Dataset")
        else:
            st.info("Mode: Uploaded Statement")

    val_summary = st.session_state.get("validation_summary")
    if val_summary:
        reconcile = val_summary.get("reconciliation", {})
        if reconcile.get("status") == "reconciled":
            st.success(f"✅ **Reconciled:** {reconcile.get('message')}")
        elif reconcile.get("status") == "discrepancy":
            st.warning(f"⚠️ **Reconciliation Notice:** {reconcile.get('message')}")

    debits_df  = transactions_df[transactions_df["type"] == "debit"]
    credits_df = transactions_df[transactions_df["type"] == "credit"]

    total_count    = len(transactions_df)
    total_spent    = debits_df["amount"].sum()
    total_received = credits_df["amount"].sum()
    net_flow       = total_received - total_spent
    avg_debit      = debits_df["amount"].mean() if not debits_df.empty else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🧾 Transactions", f"{total_count:,}")
    k2.metric("💸 Total Spent", fmt_inr(total_spent))
    k3.metric("💰 Total Received", fmt_inr(total_received))
    k4.metric("📈 Net Cash Flow", fmt_inr(net_flow))
    k5.metric("📉 Avg. Debit", fmt_inr(avg_debit))

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:

        if not debits_df.empty and "category" in debits_df.columns:
            cat_data = (
                debits_df.groupby("category")["amount"]
                .sum()
                .reset_index()
                .sort_values("amount", ascending=True)
            )
            fig_cat = px.bar(
                cat_data,
                x="amount",
                y="category",
                orientation="h",
                title="💳 Spending by Category",
                labels={"amount": "Amount (Rs.)", "category": ""},
                color="amount",
                color_continuous_scale=["#162a26", "#00D4AA"],
            )
            fig_cat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                coloraxis_showscale=False,
                margin=dict(l=0, r=20, t=35, b=0),
                height=300,
            )
            fig_cat.update_xaxes(showgrid=True, gridcolor="#2d3748")
            fig_cat.update_yaxes(showgrid=False)
            st.plotly_chart(fig_cat, use_container_width=True)

    with chart_c2:

        if "date" in transactions_df.columns:
            df_dated = transactions_df.dropna(subset=["date"]).copy()
            df_dated["month"] = pd.to_datetime(df_dated["date"]).dt.to_period("M").astype(str)
            if df_dated["month"].nunique() > 1:
                monthly_flow = (
                    df_dated.groupby(["month", "type"])["amount"]
                    .sum()
                    .reset_index()
                )
                fig_m = px.bar(
                    monthly_flow,
                    x="month",
                    y="amount",
                    color="type",
                    barmode="group",
                    title="📅 Monthly Income vs. Spending",
                    labels={"amount": "Amount (Rs.)", "month": "Month", "type": "Type"},
                    color_discrete_map={"debit": "#f87171", "credit": "#00D4AA"},
                )
                fig_m.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    margin=dict(l=0, r=20, t=35, b=0),
                    height=300,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_m.update_xaxes(showgrid=False)
                fig_m.update_yaxes(showgrid=True, gridcolor="#2d3748")
                st.plotly_chart(fig_m, use_container_width=True)
            else:

                top_pay = debits_df.groupby("description")["amount"].sum().reset_index().nlargest(6, "amount").sort_values("amount", ascending=True)
                fig_p = px.bar(
                    top_pay,
                    x="amount",
                    y="description",
                    orientation="h",
                    title="🏢 Top Payees & Merchants",
                    labels={"amount": "Amount (Rs.)", "description": ""},
                    color="amount",
                    color_continuous_scale=["#1a2a3a", "#38bdf8"],
                )
                fig_p.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=20, t=35, b=0),
                    height=300,
                )
                st.plotly_chart(fig_p, use_container_width=True)

    with st.expander("📋 Search & Inspect All Transactions", expanded=False):
        col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
        with col_s1:
            search_term = st.text_input("Search merchant, narration, or keyword", placeholder="e.g. Swiggy, Rent, TechCorp")
        with col_s2:
            cat_list = ["All Categories"] + sorted([str(c) for c in transactions_df["category"].dropna().unique()])
            filter_cat = st.selectbox("Category", cat_list)
        with col_s3:
            filter_type = st.selectbox("Type", ["All Types", "debit", "credit"])

        view_df = transactions_df.copy()
        if search_term.strip():
            view_df = view_df[view_df["description"].str.contains(search_term.strip(), case=False, na=False)]
        if filter_cat != "All Categories":
            view_df = view_df[view_df["category"] == filter_cat]
        if filter_type != "All Types":
            view_df = view_df[view_df["type"] == filter_type]

        cols_to_show = [c for c in ["date", "description", "amount", "type", "category", "payment_method", "balance"] if c in view_df.columns]
        st.dataframe(
            view_df[cols_to_show].sort_values("date", ascending=False, na_position="last"),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Showing {len(view_df)} of {len(transactions_df)} transactions.")

else:

    st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: #141824; border-radius: 8px; border: 1px dashed #2d3748; margin: 20px 0;">
    <div style="font-size: 40px; margin-bottom: 8px;">📊</div>
    <h4 style="color: #cbd5e1; margin-bottom: 6px;">No Bank Statement Loaded</h4>
    <p style="color: #64748b; font-size: 14px; max-width: 480px; margin: 0 auto;">
        Click <strong>"Load Sample Statement"</strong> above to test the app immediately,
        or drop your own bank statement PDF / CSV.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()
st.markdown("### 💬 Ask your statement")
st.caption("Ask natural language questions. Arithmetic questions use exact Pandas calculations; document facts use ChromaDB RAG.")

suggested_questions = [
    "How much did I spend in total?",
    "Which merchant received the most money?",
    "Show my largest transactions",
    "Compare spending across months",
]

st.markdown("**Suggested questions:**")
q_grid_c1, q_grid_c2 = st.columns(2)
clicked_preset = None

with q_grid_c1:
    if st.button(f"🔍 {suggested_questions[0]}", key="chip_0", use_container_width=True):
        clicked_preset = suggested_questions[0]
    if st.button(f"🔍 {suggested_questions[1]}", key="chip_1", use_container_width=True):
        clicked_preset = suggested_questions[1]

with q_grid_c2:
    if st.button(f"🔍 {suggested_questions[2]}", key="chip_2", use_container_width=True):
        clicked_preset = suggested_questions[2]
    if st.button(f"🔍 {suggested_questions[3]}", key="chip_3", use_container_width=True):
        clicked_preset = suggested_questions[3]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "route" in msg and msg["route"]:
            st.caption(f"Engine: **{msg['route']}**")

ANALYTICAL_KEYWORDS = [
    "how much", "total", "most", "top", "spent", "spend", "spending",
    "highest", "lowest", "average", "mean", "sum", "breakdown",
    "category", "categories", "whom", "who did i pay", "payee", "merchant",
    "largest", "biggest", "expensive", "pay", "paid", "receive", "received",
    "credit", "debit", "upi", "neft", "imps", "cash", "card", "pos",
    "month", "monthly", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
    "net cash", "cash flow", "savings", "surplus", "transactions"
]

input_query = st.chat_input("e.g. How much did I spend in July? Who did I pay the most?")
active_question = clicked_preset or input_query

if active_question:

    with st.chat_message("user"):
        st.markdown(active_question)
    st.session_state["messages"].append({"role": "user", "content": active_question})

    current_df = st.session_state.get("transactions_df")

    with st.chat_message("assistant"):
        if current_df is None or current_df.empty:
            help_msg = (
                "⚠️ **No active bank statement found.**\n\n"
                "Please click **'Load Sample Statement'** above or upload your statement to begin asking questions!"
            )
            st.markdown(help_msg)
            st.session_state["messages"].append({"role": "assistant", "content": help_msg, "route": ""})
        else:
            with st.spinner("Analyzing statement..."):
                try:
                    q_lower = active_question.lower()
                    is_analytical = any(kw in q_lower for kw in ANALYTICAL_KEYWORDS)

                    if is_analytical:
                        ans = answer_analytical_question(active_question, current_df)
                        route_name = "Deterministic Analytics (Pandas)"
                    else:
                        ans = get_answer(active_question, st.session_state.get("vectorstore"))
                        route_name = "Document Lookup (ChromaDB RAG)"

                    st.markdown(ans)
                    st.caption(f"Engine: **{route_name}**")
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": ans,
                        "route": route_name
                    })
                except Exception as e:
                    err_msg = f"Sorry, hit an issue generating answer: {e}"
                    st.error(err_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": err_msg, "route": "Error"})

if st.session_state["messages"]:
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()
