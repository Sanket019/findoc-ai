"""
app.py — Financial Document Assistant
A RAG-powered chatbot for bank statement analysis built with:
  LangChain · ChromaDB · Groq LLM · HuggingFace Embeddings · Streamlit · Pandas

Architecture:
  - Lookup questions  → ChromaDB similarity search (RAG)
  - Analytical queries → Pandas groupby/filter → LLM for phrasing
"""

import os
from io import BytesIO

import streamlit as st
import pandas as pd
import pdfplumber
import pikepdf
import plotly.express as px
from dotenv import load_dotenv

from ingest import process_and_store_text, extract_transactions
from rag_chain import get_answer, answer_analytical_question

load_dotenv()

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="FinDoc AI — Bank Statement Analyser",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# CUSTOM CSS  (fintech teal/dark theme)
# ===========================================================================
st.markdown("""
<style>
/* ---- Accent / brand colour ---- */
:root { --accent: #00D4AA; --accent-dark: #00a888; }

/* ---- Metric cards ---- */
div[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label {
    color: #a0aec0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #00D4AA !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ---- Sidebar styling ---- */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #2d3748;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ---- Sample question chips ---- */
div[data-testid="stButton"] > button {
    border-radius: 20px !important;
    border: 1px solid #2d3748 !important;
    background: #1a1f2e !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    border-color: #00D4AA !important;
    color: #00D4AA !important;
    background: #0d1f1c !important;
}

/* ---- Section headers ---- */
h2, h3 { color: #e2e8f0 !important; }

/* ---- Info/success boxes ---- */
div[data-testid="stAlert"] { border-radius: 10px; }

/* ---- Upload area ---- */
div[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #2d3748 !important;
    border-radius: 12px !important;
    background: #1a1f2e !important;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("## 💰 FinDoc AI")
    st.markdown("**Bank Statement Analyser**")
    st.divider()

    # Re-extract button (clears cached extraction, keeps vectorstore)
    if st.button("🔄 Re-extract Transactions", use_container_width=True):
        for key in ["transactions_df", "processed_file_key", "extraction_debug"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown("")
    st.caption("Built with LangChain · ChromaDB · Groq")


# ===========================================================================
# HELPERS
# ===========================================================================
def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file object using pdfplumber (layout-aware)."""
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


def extract_text_from_csv(file) -> str:
    """Parse a CSV bank export and return it as a formatted text block."""
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc, on_bad_lines="skip")
            return df.to_string(index=False)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"CSV read error: {e}")
            return ""
    st.error("Could not decode the CSV file.")
    return ""


def fmt_inr(amount: float) -> str:
    """Format a number as Indian Rupees (e.g. Rs.12,450)."""
    return f"Rs.{amount:,.0f}"


# ===========================================================================
# MAIN AREA — HEADER
# ===========================================================================
st.markdown("# 💰 Financial Document Assistant")
st.markdown("Upload a bank statement and ask questions in plain English — "
            "powered by **RAG**, **LangChain**, and **Groq LLM**.")
st.divider()

# ===========================================================================
# FILE UPLOAD
# ===========================================================================
uploaded_file = st.file_uploader(
    "📂 Upload your bank statement",
    type=["pdf", "csv"],
    help="Supports PDF (text-based, not scanned) and CSV exports from your bank.",
)
st.caption("🔒 Password-protected PDFs are supported — you'll be prompted if needed.")

# ===========================================================================
# DOCUMENT PROCESSING
# Cached per (filename + size) so processing only runs ONCE per upload,
# not on every Streamlit rerun (which happens on every button click / message).
# ===========================================================================
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    file_key  = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("processed_file_key") != file_key:
        # ---- Step 0: Decrypt password-protected PDFs ----
        file_is_copy  = False   # True if file_to_process is a new BytesIO copy
        file_to_process = uploaded_file

        if file_type == "pdf":
            try:
                with pikepdf.open(uploaded_file) as _:
                    pass   # Not encrypted
            except pikepdf.PasswordError:
                pwd = st.text_input("🔐 This PDF is encrypted — enter password:", type="password")
                if not pwd:
                    st.warning("Enter the PDF password to continue.")
                    st.stop()
                try:
                    uploaded_file.seek(0)
                    with pikepdf.open(uploaded_file, password=pwd) as pdf:
                        buf = BytesIO()
                        pdf.save(buf)
                        buf.seek(0)
                        file_to_process = buf
                        file_is_copy    = True
                except pikepdf.PasswordError:
                    st.error("❌ Incorrect password. Please try again.")
                    st.stop()
                except Exception as e:
                    st.error(f"PDF unlock error: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"PDF open error: {e}")
                st.stop()

            if not file_is_copy:
                uploaded_file.seek(0)

        # ---- Step 1: Extract raw text ----
        if file_type == "pdf":
            extracted_text = extract_text_from_pdf(file_to_process)
        else:
            extracted_text = extract_text_from_csv(uploaded_file)

        if not extracted_text.strip():
            st.error("⚠️ No text could be extracted. If this is a scanned PDF, OCR is required.")
            st.stop()

        # ---- Step 2: Build RAG vector store ----
        with st.spinner("Processing document..."):
            try:
                vectorstore = process_and_store_text(extracted_text, uploaded_file.name)
                st.session_state["vectorstore"] = vectorstore
            except Exception as e:
                st.error(f"Indexing error: {e}")

        # ---- Step 3: Extract structured transactions ----
        try:
            transactions_df, debug = extract_transactions(extracted_text)
            st.session_state["transactions_df"] = transactions_df
            st.session_state["extraction_debug"] = debug
        except Exception as e:
            st.session_state["transactions_df"]  = pd.DataFrame()
            st.session_state["extraction_debug"] = {"error": str(e), "raw_output": "", "clean_json": ""}

        # Mark this file as processed
        st.session_state["processed_file_key"] = file_key

    # ===========================================================================
    # DASHBOARD — always render if we have transactions
    # ===========================================================================
    transactions_df = st.session_state.get("transactions_df")

    if transactions_df is not None and not transactions_df.empty:
        st.divider()
        st.markdown("### 📊 Spending Dashboard")

        # Metrics row
        total_tx       = len(transactions_df)
        total_spent    = transactions_df[transactions_df["type"] == "debit"]["amount"].sum()
        total_received = transactions_df[transactions_df["type"] == "credit"]["amount"].sum()
        avg_tx         = transactions_df[transactions_df["type"] == "debit"]["amount"].mean() if total_spent > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧾 Total Transactions", total_tx)
        c2.metric("💸 Total Spent",    fmt_inr(total_spent))
        c3.metric("💰 Total Received", fmt_inr(total_received))
        c4.metric("📉 Avg. Debit",     fmt_inr(avg_tx))

        # Spending by category chart
        debits_df = transactions_df[transactions_df["type"] == "debit"]
        if not debits_df.empty and "category" in debits_df.columns:
            cat_spend = (
                debits_df.groupby("category")["amount"]
                .sum()
                .reset_index()
                .sort_values("amount", ascending=True)
            )
            if not cat_spend.empty:
                fig = px.bar(
                    cat_spend,
                    x="amount",
                    y="category",
                    orientation="h",
                    title="💳 Spending by Category",
                    labels={"amount": "Amount (Rs.)", "category": ""},
                    color="amount",
                    color_continuous_scale=["#1a3a33", "#00D4AA"],
                )
                fig.update_layout(
                    plot_bgcolor  = "rgba(0,0,0,0)",
                    paper_bgcolor = "rgba(0,0,0,0)",
                    font_color    = "#e2e8f0",
                    coloraxis_showscale = False,
                    margin = dict(l=0, r=20, t=50, b=0),
                    title_font_size = 16,
                )
                fig.update_xaxes(showgrid=True, gridcolor="#2d3748")
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

        # Collapsible transactions table
        with st.expander("📋 View all extracted transactions"):
            cols = [c for c in ["date", "description", "amount", "type", "category", "payment_method"]
                    if c in transactions_df.columns]
            st.dataframe(
                transactions_df[cols].sort_values("date", ascending=False, na_position="last"),
                use_container_width=True,
                hide_index=True,
            )

else:
    # Nothing uploaded yet — show a clean welcome state
    st.markdown("""
<div style="text-align:center; padding: 60px 20px; color: #4a5568;">
    <div style="font-size: 64px;">📄</div>
    <h3 style="color: #718096;">Upload a bank statement to get started</h3>
    <p>Supports SBI, HDFC, ICICI, Axis, and most other Indian bank PDFs</p>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# CHAT INTERFACE
# ===========================================================================
if "vectorstore" in st.session_state:
    st.divider()
    st.markdown("### 💬 Ask About Your Statement")

    # ---- Sample question chips ----
    sample_questions = [
        "What is the closing balance?",
        "How much did I spend on food?",
        "How much via UPI?",
        "Who did I pay the most?",
        "What is my total spending?",
        "What is the account holder name?",
    ]

    st.markdown("**Quick questions:**")
    chip_cols = st.columns(len(sample_questions))
    preset_q  = None
    for col, q in zip(chip_cols, sample_questions):
        if col.button(q, key=f"chip_{q}"):
            preset_q = q

    st.markdown("")

    # ---- Chat history ----
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Routing keywords ----
    ANALYTICAL_KW = [
        "how much", "total", "most", "top", "spent", "spend",
        "highest", "lowest", "average", "sum", "breakdown",
        "category", "categories", "whom", "who did i pay",
        "largest", "biggest", "expensive",
        "pay", "paid", "receive", "received", "credit", "debit",
        "upi", "neft", "imps", "cash", "card", "pos",
    ]

    # ---- Process question (either from chip or chat input) ----
    user_question = preset_q or st.chat_input(
        "e.g. What is the closing balance? How much did I spend on food?"
    )

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state["messages"].append({"role": "user", "content": user_question})

        q_lower       = user_question.lower()
        is_analytical = any(kw in q_lower for kw in ANALYTICAL_KW)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if is_analytical and "transactions_df" in st.session_state:
                        # ANALYTICAL PATH: pandas → LLM phrasing
                        answer = answer_analytical_question(
                            question=user_question,
                            transactions_df=st.session_state["transactions_df"],
                        )
                    else:
                        # RAG PATH: ChromaDB similarity search → LLM
                        answer = get_answer(
                            question=user_question,
                            vectorstore=st.session_state["vectorstore"],
                        )

                    # Route label for transparency
                    route = "📊 Analytical (Pandas)" if is_analytical else "🔍 RAG (ChromaDB)"
                    st.markdown(answer)
                    st.caption(f"Answered via: **{route}**")
                    st.session_state["messages"].append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    # Clear chat button
    if st.session_state.get("messages"):
        st.markdown("")
        if st.button("🗑️ Clear chat history"):
            st.session_state["messages"] = []
            st.rerun()
