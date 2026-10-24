"""
rag_chain.py — LLM-powered Q&A for the Financial Document Assistant.

Two functions:
  get_answer()                — RAG lookup: retrieves relevant chunks from
                                ChromaDB and asks the LLM a grounded question.
  answer_analytical_question()— Pandas-first analytics: all math is done by
                                pandas; the LLM only phrases the final sentence.
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Single model constant — change here to update both functions at once
GROQ_MODEL      = "openai/gpt-oss-120b"
GROQ_MAX_TOKENS = 400   # Generous enough for answers, safe under OTPM limits


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def safe_print(msg: str):
    """Print safely on Windows (strips U+202F narrow spaces and handles cp1252)."""
    msg = str(msg).replace("\u202f", "").replace("\xa0", " ")
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# FUNCTION 1 — get_answer  (RAG / lookup questions)
# ---------------------------------------------------------------------------
def get_answer(question: str, vectorstore) -> str:
    """
    Retrieves the top-4 most relevant text chunks from ChromaDB via cosine
    similarity, then asks the LLM to answer using ONLY that context.

    Why RAG?  The full document is too large for the LLM's context window.
    ChromaDB lets us retrieve only the paragraphs that are most likely to
    contain the answer (e.g. account holder name, IFSC code, closing balance).
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt_template = """You are a helpful financial document assistant.
Answer ONLY using the context below. If the answer is not in the context,
say: "I don't have enough information in this document to answer that."
Do not guess or use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, max_tokens=GROQ_MAX_TOKENS)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    try:
        return chain.invoke(question)
    except Exception as e:
        safe_print(f"[get_answer] Error: {e}")
        return f"Sorry, I hit an error looking that up: {e}"


# ---------------------------------------------------------------------------
# FUNCTION 2 — answer_analytical_question  (aggregation / math questions)
# ---------------------------------------------------------------------------
def answer_analytical_question(question: str, transactions_df: pd.DataFrame) -> str:
    """
    Pandas-first analytics engine.

    Step 1 — Classify the question and run the correct pandas operation.
              All arithmetic happens here — never in the LLM.
    Step 2 — Pass the exact numbers to the LLM as plain text, asking it
              to write one clear sentence.

    Why not just ask the LLM?  LLMs are unreliable at arithmetic across
    large datasets. Pandas is deterministic and correct every time.
    """
    if transactions_df is None or transactions_df.empty:
        return (
            "I wasn't able to extract structured transactions from your document.\n\n"
            "Try clicking **Re-extract Transactions** in the sidebar, or upload a "
            "different file."
        )

    q  = question.lower()
    df = transactions_df.copy()

    # Normalise columns for case-insensitive matching
    df["cat_lower"]  = df.get("category",       pd.Series(dtype=str)).str.lower().fillna("")
    df["desc_lower"] = df.get("description",     pd.Series(dtype=str)).str.lower().fillna("")
    df["pay_lower"]  = df.get("payment_method",  pd.Series(dtype=str)).str.lower().fillna("")

    debits  = df[df["type"] == "debit"]
    credits = df[df["type"] == "credit"]

    facts = []   # plain-text fact strings handed to the LLM for phrasing

    # -----------------------------------------------------------------------
    # Routing map 1: CATEGORY keywords
    # -----------------------------------------------------------------------
    CATEGORY_MAP = {
        "food": "food",         "swiggy": "food",     "zomato": "food",
        "blinkit": "food",      "transport": "transport", "uber": "transport",
        "ola": "transport",     "rapido": "transport", "shopping": "shopping",
        "amazon": "shopping",   "flipkart": "shopping", "entertainment": "entertainment",
        "netflix": "entertainment", "spotify": "entertainment",
        "health": "health",     "hospital": "health",  "pharmacy": "health",
        "education": "education", "housing": "housing", "rent": "housing",
        "cash": "cash withdrawal", "atm": "cash withdrawal",
        "salary": "income",
    }

    # Routing map 2: PAYMENT METHOD keywords
    PAYMENT_MAP = {
        "upi":         "upi",
        "neft":        "neft",
        "imps":        "imps",
        "card":        "card",
        "pos":         "card",
        "debit card":  "card",
        "credit card": "card",
    }

    matched_cat = None
    for kw, cat in CATEGORY_MAP.items():
        if kw in q:
            matched_cat = cat
            break

    matched_pm = None
    for kw, pm in PAYMENT_MAP.items():
        if kw in q:
            matched_pm = pm
            break

    # --- Payment method filter ---
    if matched_pm:
        pm_debits = debits[debits["pay_lower"] == matched_pm]
        if pm_debits.empty:
            pm_debits = debits[debits["desc_lower"].str.contains(matched_pm, na=False)]

        pm_credits = credits[credits["pay_lower"] == matched_pm]
        if pm_credits.empty:
            pm_credits = credits[credits["desc_lower"].str.contains(matched_pm, na=False)]

        d_total = pm_debits["amount"].sum()
        c_total = pm_credits["amount"].sum()
        safe_print(f"\n[PANDAS] Payment={matched_pm.upper()} debits={d_total:.2f} credits={c_total:.2f}")

        facts.append(f"Total PAID via {matched_pm.upper()}: Rs.{d_total:,.2f} ({len(pm_debits)} transactions)")
        if c_total > 0:
            facts.append(f"Total RECEIVED via {matched_pm.upper()}: Rs.{c_total:,.2f} ({len(pm_credits)} transactions)")
        if not pm_debits.empty:
            facts.append("Individual payments:\n" + "\n".join(
                f"  {r['description']}: Rs.{r['amount']:,.2f}"
                for _, r in pm_debits.head(10).iterrows()
            ))

    # --- Category filter ---
    elif matched_cat:
        cat_df = debits[debits["cat_lower"] == matched_cat]
        if cat_df.empty:
            cat_df = debits[debits["desc_lower"].str.contains(matched_cat.split()[0], na=False)]

        total = cat_df["amount"].sum()
        safe_print(f"\n[PANDAS] Category={matched_cat} total={total:.2f} rows={len(cat_df)}")
        facts.append(f"Total spent on {matched_cat}: Rs.{total:,.2f} ({len(cat_df)} transactions)")
        if not cat_df.empty:
            facts.append("Transactions:\n" + "\n".join(
                f"  {r['description']}: Rs.{r['amount']:,.2f}"
                for _, r in cat_df.head(10).iterrows()
            ))

    # --- Name-based search ---
    if not matched_cat and not matched_pm:
        STOP = {
            "how", "much", "did", "i", "pay", "to", "total", "spent", "spend",
            "on", "for", "the", "a", "an", "is", "was", "what", "where", "who",
            "whom", "my", "in", "of", "and", "or", "through", "via", "using",
            "about", "highest", "lowest", "sum", "breakdown", "category",
            "largest", "biggest", "top", "all", "overall", "give", "show", "me",
            "list", "any", "some", "average", "mean",
        }
        words = set(re.findall(r"\b[a-z]{3,}\b", q)) - STOP - set(CATEGORY_MAP) - set(PAYMENT_MAP)
        best_df, best_name = pd.DataFrame(), ""
        for w in words:
            hit = debits[debits["desc_lower"].str.contains(w, na=False)]
            if not hit.empty and len(hit) > len(best_df):
                best_df, best_name = hit, w

        if not best_df.empty:
            total = best_df["amount"].sum()
            safe_print(f"\n[PANDAS] Name='{best_name}' total={total:.2f} rows={len(best_df)}")
            facts.append(f"Total paid to/for '{best_name}': Rs.{total:,.2f} ({len(best_df)} transactions)")
            facts.append("Matching transactions:\n" + "\n".join(
                f"  {r['description']}: Rs.{r['amount']:,.2f}"
                for _, r in best_df.head(10).iterrows()
            ))

    # --- Top payees ---
    if any(kw in q for kw in ["whom", "who", "payee", "paid most", "pay most", "highest payee"]):
        top = debits.groupby("description")["amount"].sum().sort_values(ascending=False).head(5).round(2)
        safe_print(f"\n[PANDAS] Top payees:\n{top.to_string()}")
        facts.append("Top 5 payees:\n" + "\n".join(f"  {d}: Rs.{a:,.2f}" for d, a in top.items()))

    # --- Total spend / received ---
    if any(kw in q for kw in ["total", "overall", "all", "everything"]) and not matched_cat:
        ts = debits["amount"].sum()
        tr = credits["amount"].sum()
        safe_print(f"\n[PANDAS] Overall debits={ts:.2f} credits={tr:.2f}")
        facts.append(f"Total spent (all debits):    Rs.{ts:,.2f}")
        facts.append(f"Total received (all credits): Rs.{tr:,.2f}")

    # --- Biggest transactions ---
    if any(kw in q for kw in ["biggest", "largest", "highest", "most expensive", "top transaction"]):
        top5 = debits.nlargest(5, "amount")[["date", "description", "amount"]]
        safe_print(f"\n[PANDAS] Top 5 largest:\n{top5.to_string()}")
        facts.append("5 largest transactions:\n" + "\n".join(
            f"  {r['description']}: Rs.{r['amount']:,.2f}" for _, r in top5.iterrows()
        ))

    # --- Category breakdown ---
    if any(kw in q for kw in ["breakdown", "categor", "split", "by category"]):
        cb = debits.groupby("category")["amount"].sum().sort_values(ascending=False).round(2)
        safe_print(f"\n[PANDAS] Category breakdown:\n{cb.to_string()}")
        facts.append("Spending by category:\n" + "\n".join(f"  {c}: Rs.{a:,.2f}" for c, a in cb.items()))

    # --- Fallback: general summary ---
    if not facts:
        ts = debits["amount"].sum()
        cb = debits.groupby("category")["amount"].sum().sort_values(ascending=False).round(2)
        safe_print(f"\n[PANDAS] Fallback total={ts:.2f}")
        facts.append(f"Total spent overall: Rs.{ts:,.2f}")
        facts.append("By category:\n" + "\n".join(f"  {c}: Rs.{a:,.2f}" for c, a in cb.items()))

    facts_text = "\n".join(facts)
    safe_print(f"\n[PANDAS] Facts → LLM:\n{facts_text}\n")

    # Hand pre-calculated numbers to LLM — phrasing only, zero arithmetic
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, max_tokens=GROQ_MAX_TOKENS)
    prompt = PromptTemplate(
        input_variables=["facts", "question"],
        template="""You are a helpful financial assistant.
The numbers below were computed by pandas — they are 100% accurate.
Write a concise, friendly answer to the user's question using only these facts.
Do NOT recalculate anything.

Pre-calculated data:
{facts}

User question: {question}

Answer:""",
    )
    try:
        return (prompt | llm | StrOutputParser()).invoke({"facts": facts_text, "question": question})
    except Exception as e:
        safe_print(f"[answer_analytical_question] LLM phrasing error: {e}")
        return f"Here is what I found:\n\n{facts_text}"
