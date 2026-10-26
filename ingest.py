import sys
import os
import re
import json
import time
import io
import hashlib
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

def safe_print(msg: str):

    msg = str(msg).replace("\u202f", "").replace("\xa0", " ")
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

CATEGORY_RULES = [
    (r"\b(salary|interest credit|dividend|consulting|freelance|stipend|income)\b", "Income"),
    (r"\b(rent|society|maintenance|bescom|electricity|water|broadband|airtel|wifi|gas|utility)\b", "Housing"),
    (r"\b(swiggy|zomato|blinkit|zepto|restaurant|cafe|dining|food|lunch|dinner|grocer|nature basket|reliance fresh|starbucks)\b", "Food"),
    (r"\b(uber|ola|rapido|metro|petrol|fuel|shell|hp pump|indian oil|transport|commute|flight|train|irctc|cab)\b", "Transport"),
    (r"\b(amazon|flipkart|meesho|myntra|zara|shopping|electronics|mall|croma)\b", "Shopping"),
    (r"\b(netflix|spotify|prime|hotstar|bookmyshow|movie|theatre|youtube|entertainment)\b", "Entertainment"),
    (r"\b(pharmacy|apollo|medplus|hospital|clinic|doctor|health|cultfit|gym|medicine|diagnostic)\b", "Health"),
    (r"\b(school|college|university|coursera|udemy|fees|education|tuition)\b", "Education"),
    (r"\b(atm|cash withdrawal|cash wdl|atm-cash)\b", "Cash Withdrawal"),
]

def infer_category(desc: str) -> str:

    desc_l = str(desc).lower()
    for pattern, cat in CATEGORY_RULES:
        if re.search(pattern, desc_l):
            return cat
    return "Other"

def infer_payment_method(desc: str) -> str:

    desc_l = str(desc).lower()
    if re.search(r"\bupi\b|^upi/", desc_l):
        return "UPI"
    if re.search(r"\bneft\b|^neft/", desc_l):
        return "NEFT"
    if re.search(r"\bimps\b|^imps/", desc_l):
        return "IMPS"
    if re.search(r"\b(atm|cash)\b", desc_l):
        return "Cash"
    if re.search(r"\b(pos|card|debit card|credit card)\b", desc_l):
        return "Card"
    return "Other"

def process_and_store_text(text: str, source_name: str, session_id: Optional[str] = None) -> Chroma:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks produced — document may be empty.")

    raw_ident = f"{source_name}_{session_id or 'default'}"
    ident_hash = hashlib.md5(raw_ident.encode("utf-8")).hexdigest()[:12]
    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", source_name)[:18].strip("_") or "doc"
    collection_name = f"fd_{clean_name}_{ident_hash}"

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    metadatas  = [{"source": source_name, "chunk_id": i} for i in range(len(chunks))]

    persist_dir = "./chroma_db"
    os.makedirs(persist_dir, exist_ok=True)

    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
    except Exception as e:
        safe_print(f"[VECTORSTORE] Using direct Chroma.from_texts ({e})")
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
    return vectorstore

def _clean_num(val) -> Optional[float]:

    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    s = re.sub(r"[₹$€£,]|(?:Rs\.?|INR)", "", s, flags=re.IGNORECASE).strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def parse_csv_statement(content_or_path) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    lines: List[str] = []
    if isinstance(content_or_path, str) and os.path.exists(content_or_path):
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                with open(content_or_path, "r", encoding=enc, errors="replace") as f:
                    lines = f.readlines()
                break
            except Exception:
                continue
    elif isinstance(content_or_path, bytes):
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                lines = content_or_path.decode(enc).splitlines()
                break
            except Exception:
                continue
    elif hasattr(content_or_path, "read"):
        if hasattr(content_or_path, "seek"):
            content_or_path.seek(0)
        data = content_or_path.read()
        if isinstance(data, bytes):
            for enc in ["utf-8", "latin1", "cp1252"]:
                try:
                    lines = data.decode(enc).splitlines()
                    break
                except Exception:
                    continue
        else:
            lines = str(data).splitlines()
    else:
        lines = str(content_or_path).splitlines()

    metadata: Dict[str, str] = {}
    warnings: List[str] = []
    clean_lines = [l.strip() for l in lines if l.strip()]

    if not clean_lines:
        return EMPTY_DF.copy(), {
            "total_rows_read": 0,
            "valid_transactions": 0,
            "invalid_rows_dropped": 0,
            "duplicates_removed": 0,
            "metadata": {},
            "warnings": ["Empty CSV file provided."]
        }

    header_idx = -1
    for i, line in enumerate(clean_lines):
        line_l = line.lower()
        if "date" in line_l and any(k in line_l for k in ["desc", "narration", "particular", "amount", "debit", "credit", "type"]):
            header_idx = i
            break

    if header_idx == -1:
        header_idx = 0

    for line in clean_lines[:header_idx]:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            metadata[parts[0]] = parts[1]

    csv_text = "\n".join(clean_lines[header_idx:])
    try:
        raw_df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")
    except Exception as e:
        return EMPTY_DF.copy(), {
            "total_rows_read": 0,
            "valid_transactions": 0,
            "invalid_rows_dropped": 0,
            "duplicates_removed": 0,
            "metadata": metadata,
            "warnings": [f"CSV parsing failed: {e}"]
        }

    col_map = {}
    for col in raw_df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_")
        if c_clean in ["date", "txn_date", "transaction_date", "value_date", "tx_date"]:
            col_map[col] = "date"
        elif c_clean in ["description", "narration", "particulars", "remarks", "details", "merchant", "details"]:
            col_map[col] = "description"
        elif c_clean in ["amount", "txn_amount", "transaction_amount"]:
            col_map[col] = "amount"
        elif c_clean in ["type", "dr/cr", "cr/dr", "txn_type", "transaction_type"]:
            col_map[col] = "type"
        elif c_clean in ["debit", "dr", "withdrawal", "withdrawal_amount", "debit_amount"]:
            col_map[col] = "debit"
        elif c_clean in ["credit", "cr", "deposit", "deposit_amount", "credit_amount"]:
            col_map[col] = "credit"
        elif c_clean in ["category", "cat"]:
            col_map[col] = "category"
        elif c_clean in ["payment_method", "mode", "channel", "payment_mode"]:
            col_map[col] = "payment_method"
        elif c_clean in ["balance", "closing_balance", "running_balance", "bal"]:
            col_map[col] = "balance"

    df = raw_df.rename(columns=col_map)
    parsed_rows = []
    invalid_rows = 0

    meta_keywords = [
        "account holder", "account number", "ifsc code", "branch",
        "statement period", "opening balance", "closing balance", "total"
    ]

    for idx, row in df.iterrows():
        raw_date = row.get("date")
        raw_desc = str(row.get("description", "") or "").strip()

        if pd.isna(raw_date) or str(raw_date).strip() == "":
            invalid_rows += 1
            continue

        raw_date_str = str(raw_date).strip()

        if any(mk in raw_date_str.lower() for mk in meta_keywords):
            metadata[raw_date_str] = raw_desc or str(row.get("amount", ""))
            continue

        parsed_date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed_date):
            warnings.append(f"Row {idx+1}: Unparseable date '{raw_date}' dropped.")
            invalid_rows += 1
            continue

        amt = None
        txn_type = None

        if "debit" in df.columns or "credit" in df.columns:
            deb_val = _clean_num(row.get("debit"))
            cred_val = _clean_num(row.get("credit"))

            if deb_val is not None and deb_val > 0:
                amt = float(deb_val)
                txn_type = "debit"
            elif cred_val is not None and cred_val > 0:
                amt = float(cred_val)
                txn_type = "credit"

        if amt is None and "amount" in df.columns:
            val = _clean_num(row.get("amount"))
            if val is not None and val != 0:
                amt = abs(float(val))
                raw_t = str(row.get("type", "")).lower().strip()
                if "cr" in raw_t or (val < 0 and "dr" not in raw_t):
                    txn_type = "credit"
                elif "dr" in raw_t:
                    txn_type = "debit"
                elif any(w in raw_desc.lower() for w in ["credit", "salary", "dividend", "interest"]):
                    txn_type = "credit"
                else:
                    txn_type = "debit"

        if amt is None or amt <= 0 or txn_type is None:
            invalid_rows += 1
            continue

        cat = row.get("category")
        if pd.isna(cat) or str(cat).strip() in ["", "Other", "nan", "None"]:
            cat = infer_category(raw_desc)
        else:
            cat = str(cat).strip()

        pm = row.get("payment_method")
        if pd.isna(pm) or str(pm).strip() in ["", "Other", "nan", "None"]:
            pm = infer_payment_method(raw_desc)
        else:
            pm = str(pm).strip()

        bal = row.get("balance")
        bal_num = _clean_num(bal)
        bal_val = round(float(bal_num), 2) if bal_num is not None else None

        parsed_rows.append({
            "date": parsed_date,
            "description": raw_desc or "Transaction",
            "amount": round(amt, 2),
            "type": txn_type,
            "category": cat,
            "payment_method": pm,
            "balance": bal_val,
        })

    if not parsed_rows:
        return EMPTY_DF.copy(), {
            "total_rows_read": len(raw_df),
            "valid_transactions": 0,
            "invalid_rows_dropped": invalid_rows,
            "duplicates_removed": 0,
            "metadata": metadata,
            "warnings": warnings or ["No valid transaction records found in CSV."]
        }

    out_df = pd.DataFrame(parsed_rows)
    initial_len = len(out_df)
    out_df = out_df.drop_duplicates(subset=["date", "description", "amount", "type"]).reset_index(drop=True)
    dupes_removed = initial_len - len(out_df)
    if dupes_removed > 0:
        warnings.append(f"Removed {dupes_removed} duplicate transaction records.")

    reconciliation = {"status": "unverified", "message": "Opening/closing balance not specified"}
    op_bal = metadata.get("Opening Balance")
    cl_bal = metadata.get("Closing Balance")
    if op_bal and cl_bal:
        try:
            op_f = float(str(op_bal).replace(",", "").replace("Rs.", "").replace("₹", ""))
            cl_f = float(str(cl_bal).replace(",", "").replace("Rs.", "").replace("₹", ""))
            tot_cr = out_df[out_df["type"] == "credit"]["amount"].sum()
            tot_dr = out_df[out_df["type"] == "debit"]["amount"].sum()
            expected_cl = round(op_f + tot_cr - tot_dr, 2)
            diff = round(cl_f - expected_cl, 2)
            if abs(diff) < 0.05:
                reconciliation = {
                    "status": "reconciled",
                    "opening_balance": op_f,
                    "closing_balance": cl_f,
                    "expected_closing": expected_cl,
                    "message": "Closing balance fully reconciled (Opening + Credits - Debits = Closing)."
                }
            else:
                reconciliation = {
                    "status": "discrepancy",
                    "opening_balance": op_f,
                    "closing_balance": cl_f,
                    "expected_closing": expected_cl,
                    "discrepancy": diff,
                    "message": f"Statement balance discrepancy of Rs.{abs(diff):,.2f}."
                }
        except Exception:
            pass

    summary = {
        "total_rows_read": len(raw_df),
        "valid_transactions": len(out_df),
        "invalid_rows_dropped": invalid_rows,
        "duplicates_removed": dupes_removed,
        "metadata": metadata,
        "warnings": warnings,
        "reconciliation": reconciliation,
    }
    return out_df, summary

EXTRACTION_MODELS = [
    ("openai/gpt-oss-120b", 1000),
    ("qwen/qwen3.8-27b",     800),
    ("openai/gpt-oss-20b",  1000),
]

EMPTY_DF = pd.DataFrame(
    columns=["date", "description", "amount", "type", "category", "payment_method"]
)

def _make_prompt(text: str) -> str:

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
  "category"       : one of "Food", "Transport", "Shopping", "Entertainment", "Health", "Education", "Housing", "Cash Withdrawal", "Income", "Other"
  "payment_method" : one of "UPI", "NEFT", "IMPS", "Card", "Cash", "Other"

EXAMPLE (your entire response must look exactly like this):
[{{"date":"2026-07-01","description":"UPI/Ashish Kumar","amount":500.0,"type":"debit","category":"Other","payment_method":"UPI"}},{{"date":"2026-07-02","description":"Salary NEFT ABCCorp","amount":50000.0,"type":"credit","category":"Income","payment_method":"NEFT"}}]

BANK STATEMENT TEXT:
{text}
"""

def extract_transactions_regex_fallback(text: str) -> pd.DataFrame:

    rows = []

    pattern = re.compile(
        r"(?P<date>\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\s+"
        r"(?P<desc>[A-Za-z0-9/_\-\.\s]{3,40}?)\s+"
        r"(?P<amount>(?:Rs\.?|₹)?\s*[\d,]+\.\d{2})\s*"
        r"(?P<type>dr|cr|debit|credit)?",
        re.IGNORECASE
    )

    for line in text.splitlines():
        line = line.strip()
        m = pattern.search(line)
        if m:
            raw_d = m.group("date")
            raw_desc = m.group("desc").strip()
            raw_a = m.group("amount").replace(",", "").replace("Rs.", "").replace("₹", "").strip()
            raw_t = (m.group("type") or "").lower()

            amt = pd.to_numeric(raw_a, errors="coerce")
            parsed_date = pd.to_datetime(raw_d, errors="coerce")

            if pd.notna(amt) and amt > 0 and pd.notna(parsed_date):
                if "cr" in raw_t or any(k in raw_desc.lower() for k in ["salary", "credit", "interest"]):
                    txn_type = "credit"
                else:
                    txn_type = "debit"

                rows.append({
                    "date": parsed_date,
                    "description": raw_desc,
                    "amount": round(float(amt), 2),
                    "type": txn_type,
                    "category": infer_category(raw_desc),
                    "payment_method": infer_payment_method(raw_desc),
                })

    if not rows:
        return EMPTY_DF.copy()

    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "description", "amount", "type"])
    return df

def _extract_single_chunk(chunk_text: str, progress_callback=None) -> Tuple[pd.DataFrame, Dict[str, str]]:

    debug = {"raw_output": "", "clean_json": "", "error": "", "model_used": ""}
    prompt = _make_prompt(chunk_text)

    last_error = "No models attempted"

    for model_name, max_tok in EXTRACTION_MODELS:
        safe_print(f"[EXTRACT] Trying {model_name} (max_tokens={max_tok})...")
        if progress_callback:
            progress_callback("Parsing transactions with financial extraction model...")
        try:
            llm = ChatGroq(model=model_name, temperature=0, max_tokens=max_tok)
            response   = llm.invoke(prompt)
            raw_output = (response.content or "").strip()

            finish = response.response_metadata.get("finish_reason", "unknown")
            safe_print(f"[EXTRACT] finish_reason={finish}, content_length={len(raw_output)}")
            debug["raw_output"] = raw_output

            if not raw_output:
                last_error = f"Empty response from {model_name} (finish_reason={finish})"
                time.sleep(1)
                continue

            json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
            if not json_match:
                last_error = f"No JSON array found in output from {model_name}"
                time.sleep(1)
                continue

            clean_json = json_match.group(0)
            debug["clean_json"] = clean_json[:1000]

            try:
                transactions = json.loads(clean_json)
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error from {model_name}: {e}"
                time.sleep(1)
                continue

            if not isinstance(transactions, list) or len(transactions) == 0:
                last_error = f"{model_name} returned empty transaction list"
                time.sleep(1)
                continue

            df = pd.DataFrame(transactions)
            df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0)
            df["date"]   = pd.to_datetime(df.get("date"), errors="coerce")

            for col, default in [("type", "debit"), ("category", "Other"),
                                  ("payment_method", "Other"), ("description", "")]:
                if col not in df.columns:
                    df[col] = default

            df["category"] = df.apply(
                lambda r: infer_category(r["description"]) if r["category"] in ["", "Other", None] else r["category"],
                axis=1
            )
            df["payment_method"] = df.apply(
                lambda r: infer_payment_method(r["description"]) if r["payment_method"] in ["", "Other", None] else r["payment_method"],
                axis=1
            )

            debug["error"]      = ""
            debug["model_used"] = model_name
            safe_print(f"[EXTRACT] Success with {model_name}: {len(df)} transactions")
            return df, debug

        except Exception as e:
            last_error = f"{type(e).__name__} from {model_name}: {str(e)[:200]}"
            safe_print(f"[EXTRACT] Exception: {last_error}")
            time.sleep(2)
            continue

    debug["error"] = f"All models failed. Last error: {last_error}"
    return EMPTY_DF.copy(), debug

def extract_transactions(text: str, progress_callback=None) -> Tuple[pd.DataFrame, Dict[str, str]]:

    if not text or not text.strip():
        return EMPTY_DF.copy(), {"error": "Empty text provided", "model_used": "none", "raw_output": "", "clean_json": ""}

    MAX_CHUNK_CHARS = 4000
    if len(text) <= MAX_CHUNK_CHARS:
        if progress_callback:
            progress_callback("Analyzing statement transactions...")
        df, debug = _extract_single_chunk(text, progress_callback=progress_callback)
        if df.empty and not os.getenv("GROQ_API_KEY"):

            safe_print("[EXTRACT] No GROQ_API_KEY set; attempting regex fallback.")
            df_regex = extract_transactions_regex_fallback(text)
            if not df_regex.empty:
                debug["model_used"] = "regex_fallback"
                debug["error"] = ""
                return df_regex, debug
        return df, debug

    safe_print(f"[EXTRACT] Long document ({len(text)} chars) — splitting into batches.")
    lines = text.splitlines()
    chunks = []
    current_chunk: List[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > MAX_CHUNK_CHARS and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_len = line_len
        else:
            current_chunk.append(line)
            current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    safe_print(f"[EXTRACT] Processing {len(chunks)} statement batches.")
    collected_dfs = []
    last_debug = {"model_used": "", "error": "", "raw_output": "", "clean_json": ""}

    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Extracting transactions: section {idx+1} of {len(chunks)}...")
        df_chunk, debug = _extract_single_chunk(chunk, progress_callback=None)
        last_debug = debug
        if not df_chunk.empty:
            collected_dfs.append(df_chunk)

    if collected_dfs:
        combined_df = pd.concat(collected_dfs, ignore_index=True)

        combined_df = combined_df.drop_duplicates(
            subset=["date", "description", "amount", "type"]
        ).reset_index(drop=True)
        last_debug["error"] = ""
        safe_print(f"[EXTRACT] Combined total: {len(combined_df)} unique transactions.")
        return combined_df, last_debug

    safe_print("[EXTRACT] LLM chunks returned no records; trying regex fallback.")
    fallback_df = extract_transactions_regex_fallback(text)
    if not fallback_df.empty:
        last_debug["model_used"] = "regex_fallback"
        last_debug["error"] = ""
        return fallback_df, last_debug

    return EMPTY_DF.copy(), last_debug
