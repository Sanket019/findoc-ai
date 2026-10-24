"""
ingest.py — Document ingestion and structured transaction extraction.

Pipeline:
  1. process_and_store_text() — Chunks text → HuggingFace embeddings → ChromaDB (for RAG)
  2. extract_transactions()   — Groq LLM parses raw text → structured DataFrame
     Uses a 3-model fallback chain so rate limits on one model don't break the app.
"""

import sys
import os
import re
import json
import time

import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()


# ---------------------------------------------------------------------------
# Utility: Windows-safe print
# ---------------------------------------------------------------------------
def safe_print(msg: str):
    """
    Print to Windows cmd.exe without crashing on special Unicode chars.
    Strips: U+202F (narrow no-break space pandas uses in number formatting)
            U+20B9 (Indian Rupee sign, not in cp1252)
    """
    msg = str(msg).replace("\u202f", "").replace("\xa0", " ")
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# FUNCTION 1 — process_and_store_text (builds the RAG vector store)
# ---------------------------------------------------------------------------
def process_and_store_text(text: str, source_name: str) -> Chroma:
    """
    Splits raw text into overlapping chunks, embeds them with a local
    HuggingFace model (all-MiniLM-L6-v2), and stores them in ChromaDB.

    Returns the Chroma vectorstore object used later for similarity search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks produced — document may be empty.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    metadatas  = [{"source": source_name} for _ in chunks]

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="./chroma_db",
    )
    return vectorstore


# ---------------------------------------------------------------------------
# FUNCTION 2 — extract_transactions (structured data extraction)
# ---------------------------------------------------------------------------

# Fallback chain: (model_name, max_tokens).
# max_tokens is capped to stay under Groq free-tier OTPM limits.
EXTRACTION_MODELS = [
    ("openai/gpt-oss-120b", 1000),
    ("qwen/qwen3.8-27b",     800),
    ("openai/gpt-oss-20b",  1000),
]

EMPTY_DF = pd.DataFrame(
    columns=["date", "description", "amount", "type", "category", "payment_method"]
)


def _make_prompt(text: str) -> str:
    """
    Build the LLM extraction prompt.
    We keep the statement text to the first 5000 chars so the total prompt
    stays under model context limits and OTPM rate limits.
    """
    return f"""You are a precise financial data parser.

Your ONLY job: extract every transaction from the bank statement below
and return them as a valid JSON array.

STRICT OUTPUT RULES — violating these will break the application:
- Output MUST start with [ and end with ]
- NO markdown, NO code fences, NO explanation, NO comments
- Just the raw JSON array, nothing else

Each transaction object MUST have these exact keys:
  "date"           : "YYYY-MM-DD" string, or null
  "description"    : narration/merchant text (string)
  "amount"         : positive number (e.g. 1500.00)
  "type"           : "debit" or "credit"
  "category"       : one of:
      "Food"            (Swiggy, Zomato, Blinkit, restaurant)
      "Transport"       (Uber, Ola, Rapido, metro, bus, fuel)
      "Shopping"        (Amazon, Flipkart, Meesho, mall)
      "Entertainment"   (Netflix, Spotify, Prime, Zee, hotstar)
      "Health"          (hospital, pharmacy, Apollo, medical)
      "Education"       (school, college, Coursera, udemy)
      "Housing"         (rent, society, maintenance, electricity)
      "Cash Withdrawal" (ATM, cash)
      "Income"          (salary, credited by company, interest)
      "Other"           (everything else, UPI person-to-person)
  "payment_method" : one of "UPI", "NEFT", "IMPS", "Card", "Cash", "Other"
      Infer from description prefix: UPI/ → UPI, NEFT/ → NEFT, etc.

EXAMPLE (your entire response must look exactly like this):
[{{"date":"2026-07-01","description":"UPI/Ashish Kumar","amount":500.0,"type":"debit","category":"Other","payment_method":"UPI"}},{{"date":"2026-07-02","description":"Salary NEFT ABCCorp","amount":50000.0,"type":"credit","category":"Income","payment_method":"NEFT"}}]

BANK STATEMENT TEXT:
{text[:5000]}
"""


def extract_transactions(text: str, progress_callback=None):
    """
    Extract ALL transactions from raw bank statement text using Groq LLM.

    Tries each model in EXTRACTION_MODELS in sequence. On failure (rate limit,
    empty response, or bad JSON) it waits briefly and tries the next model.

    Args:
        text: Raw extracted text from the bank statement.
        progress_callback: Optional callable(str) for status updates (used by
                           Streamlit to show which model is being tried).

    Returns:
        tuple: (pd.DataFrame, dict)
            DataFrame columns: date, description, amount, type, category, payment_method
            dict keys: raw_output, clean_json, error, model_used
    """
    debug = {"raw_output": "", "clean_json": "", "error": "", "model_used": ""}
    prompt = _make_prompt(text)

    def _log(msg: str):
        safe_print(msg)
        if progress_callback:
            progress_callback(msg)

    last_error = "No models attempted"

    for model_name, max_tok in EXTRACTION_MODELS:
        _log(f"[EXTRACT] Trying {model_name} (max_tokens={max_tok})...")
        try:
            llm = ChatGroq(model=model_name, temperature=0, max_tokens=max_tok)
            response   = llm.invoke(prompt)
            raw_output = (response.content or "").strip()

            finish = response.response_metadata.get("finish_reason", "unknown")
            _log(f"[EXTRACT] finish_reason={finish}, content_length={len(raw_output)}")
            debug["raw_output"] = raw_output

            # Reasoning models sometimes return empty content (used all tokens thinking)
            if not raw_output:
                last_error = f"Empty response from {model_name} (finish_reason={finish})"
                _log(f"[EXTRACT] ⚠️  Empty response — {last_error}")
                time.sleep(2)
                continue

            # Find the outermost JSON array in the response
            json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if not json_match:
                last_error = f"No JSON array found in output from {model_name}"
                _log(f"[EXTRACT] ⚠️  {last_error}")
                _log(f"[EXTRACT] Preview: {raw_output[:300]}")
                time.sleep(2)
                continue

            clean_json = json_match.group(0)
            debug["clean_json"] = clean_json[:1000]

            try:
                transactions = json.loads(clean_json)
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error from {model_name}: {e}"
                _log(f"[EXTRACT] ⚠️  {last_error}")
                time.sleep(2)
                continue

            if not isinstance(transactions, list) or len(transactions) == 0:
                last_error = f"{model_name} returned empty transaction list"
                _log(f"[EXTRACT] ⚠️  {last_error}")
                time.sleep(2)
                continue

            # ✅ Success — build DataFrame
            df = pd.DataFrame(transactions)
            df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)
            df["date"]   = pd.to_datetime(df.get("date"),    errors="coerce")

            # Ensure required columns exist with sensible defaults
            for col, default in [("type", "debit"), ("category", "Other"),
                                  ("payment_method", "Other"), ("description", "")]:
                if col not in df.columns:
                    df[col] = default

            debug["error"]      = ""
            debug["model_used"] = model_name
            _log(f"[EXTRACT] ✅ Success with {model_name}: {len(df)} transactions")
            return df, debug

        except Exception as e:
            last_error = f"{type(e).__name__} from {model_name}: {str(e)[:200]}"
            _log(f"[EXTRACT] ❌ Exception: {last_error}")
            time.sleep(3)
            continue

    # All models failed
    debug["error"] = f"All models failed. Last error: {last_error}"
    _log(f"[EXTRACT] ❌ All models failed: {last_error}")
    return EMPTY_DF.copy(), debug
