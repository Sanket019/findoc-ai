import os
import re
from typing import Optional, List, Dict, Any
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_MODEL      = "openai/gpt-oss-120b"
GROQ_MAX_TOKENS = 400

def safe_print(msg: str):

    msg = str(msg).replace("\u202f", "").replace("\xa0", " ")
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))

def fmt_currency(amount: float) -> str:

    return f"Rs.{amount:,.2f}"

def get_answer(question: str, vectorstore) -> str:

    if vectorstore is None:
        return "No document is currently indexed. Please load sample data or upload a bank statement first."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "🔒 **Groq API key required for RAG document lookup.**\n\n"
            "Please provide a free `GROQ_API_KEY` in your `.env` file or Streamlit sidebar to use semantic Q&A.\n\n"
            "💡 *Tip: Numerical calculations (e.g., 'What is my total spending?', 'Who did I pay the most?') "
            "use our deterministic Pandas engine and work immediately!*"
        )

    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        docs = retriever.invoke(question)
        if not docs:
            return "I don't have enough information in this document to answer that."

        context_text = "\n\n".join(d.page_content for d in docs)

        prompt_template = """You are a helpful financial document assistant.
Answer ONLY using the bank statement context provided below.
If the answer cannot be determined from the context, respond strictly with:
"I don't have enough information in this document to answer that."
Do not guess or assume outside details.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, max_tokens=GROQ_MAX_TOKENS)

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context_text, "question": question})
        return answer.strip()

    except Exception as e:
        safe_print(f"[get_answer] Error: {e}")
        return f"Unable to retrieve an answer at this time: {e}"

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}

CATEGORY_MAP = {
    "food": "Food", "swiggy": "Food", "zomato": "Food", "blinkit": "Food", "zepto": "Food",
    "transport": "Transport", "uber": "Transport", "ola": "Transport", "rapido": "Transport",
    "fuel": "Transport", "petrol": "Transport",
    "shopping": "Shopping", "amazon": "Shopping", "flipkart": "Shopping", "meesho": "Shopping",
    "entertainment": "Entertainment", "netflix": "Entertainment", "spotify": "Entertainment", "prime": "Entertainment",
    "health": "Health", "hospital": "Health", "pharmacy": "Health", "apollo": "Health", "medplus": "Health", "gym": "Health",
    "education": "Education", "school": "Education", "college": "Education", "course": "Education",
    "housing": "Housing", "rent": "Housing", "electricity": "Housing", "maintenance": "Housing", "bescom": "Housing",
    "cash": "Cash Withdrawal", "atm": "Cash Withdrawal", "withdrawal": "Cash Withdrawal",
    "income": "Income", "salary": "Income", "interest": "Income", "freelance": "Income",
}

PAYMENT_MAP = {
    "upi": "UPI",
    "neft": "NEFT",
    "imps": "IMPS",
    "card": "Card",
    "pos": "Card",
    "debit card": "Card",
    "credit card": "Card",
    "cash": "Cash",
}

def answer_analytical_question(question: str, transactions_df: pd.DataFrame) -> str:

    if transactions_df is None or transactions_df.empty:
        return (
            "No structured transactions are currently available from this statement.\n\n"
            "Try re-extracting or uploading a clearer statement file."
        )

    q = question.lower()
    df = transactions_df.copy()

    df["cat_lower"]  = df.get("category", pd.Series(dtype=str)).astype(str).str.lower().fillna("")
    df["desc_lower"] = df.get("description", pd.Series(dtype=str)).astype(str).str.lower().fillna("")
    df["pay_lower"]  = df.get("payment_method", pd.Series(dtype=str)).astype(str).str.lower().fillna("")
    if "date" in df.columns:
        df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["dt"] = pd.NaT

    debits  = df[df["type"] == "debit"]
    credits = df[df["type"] == "credit"]

    facts: List[str] = []

    matched_month_num = None
    matched_month_name = None
    for m_name, m_num in MONTH_MAP.items():

        if re.search(rf"\b{m_name}\b", q):
            matched_month_num = m_num
            matched_month_name = m_name.capitalize()
            break

    if matched_month_num and not df["dt"].isna().all():
        m_debits = debits[debits["dt"].dt.month == matched_month_num]
        m_credits = credits[credits["dt"].dt.month == matched_month_num]
        m_spent = m_debits["amount"].sum()
        m_rec = m_credits["amount"].sum()

        facts.append(f"Spending in {matched_month_name}: {fmt_currency(m_spent)} ({len(m_debits)} debit transactions)")
        if m_rec > 0:
            facts.append(f"Credits received in {matched_month_name}: {fmt_currency(m_rec)} ({len(m_credits)} credit transactions)")
        if not m_debits.empty:
            cat_m = m_debits.groupby("category")["amount"].sum().sort_values(ascending=False)
            facts.append(f"Top categories in {matched_month_name}:\n" + "\n".join(
                f"  - {c}: {fmt_currency(a)}" for c, a in cat_m.head(5).items()
            ))

    elif any(kw in q for kw in ["across months", "monthly", "by month", "each month", "compare month"]):
        if not df["dt"].isna().all():
            df["month_yr"] = df["dt"].dt.to_period("M").astype(str)
            m_agg = df.groupby(["month_yr", "type"])["amount"].sum().unstack(fill_value=0)
            breakdown_lines = []
            for m_str, row in m_agg.iterrows():
                dr_val = row.get("debit", 0.0)
                cr_val = row.get("credit", 0.0)
                breakdown_lines.append(f"  - {m_str}: Spent {fmt_currency(dr_val)}, Received {fmt_currency(cr_val)}")
            facts.append("Monthly summary:\n" + "\n".join(breakdown_lines))

    matched_cat = None
    for kw, cat in CATEGORY_MAP.items():
        if re.search(rf"\b{kw}\b", q):
            matched_cat = cat
            break

    if matched_cat and not matched_month_num:
        cat_df = debits[debits["cat_lower"] == matched_cat.lower()]
        if cat_df.empty:
            cat_df = debits[debits["desc_lower"].str.contains(matched_cat.lower().split()[0], na=False)]

        total_cat = cat_df["amount"].sum()
        facts.append(f"Total spent on {matched_cat}: {fmt_currency(total_cat)} across {len(cat_df)} transactions.")
        if not cat_df.empty:
            facts.append("Notable transactions:\n" + "\n".join(
                f"  - {r['description']}: {fmt_currency(r['amount'])}"
                for _, r in cat_df.head(6).iterrows()
            ))

    matched_pm = None
    for kw, pm in PAYMENT_MAP.items():
        if re.search(rf"\b{kw}\b", q):
            matched_pm = pm
            break

    if matched_pm and not matched_cat and not matched_month_num:
        pm_debits = debits[debits["pay_lower"] == matched_pm.lower()]
        if pm_debits.empty:
            pm_debits = debits[debits["desc_lower"].str.contains(matched_pm.lower(), na=False)]
        pm_credits = credits[credits["pay_lower"] == matched_pm.lower()]
        if pm_credits.empty:
            pm_credits = credits[credits["desc_lower"].str.contains(matched_pm.lower(), na=False)]

        d_tot = pm_debits["amount"].sum()
        c_tot = pm_credits["amount"].sum()
        facts.append(f"Total paid via {matched_pm}: {fmt_currency(d_tot)} ({len(pm_debits)} debits)")
        if c_tot > 0:
            facts.append(f"Total received via {matched_pm}: {fmt_currency(c_tot)} ({len(pm_credits)} credits)")

    if any(kw in q for kw in ["whom", "who", "payee", "paid most", "pay most", "merchant", "highest payee"]):
        top_payees = debits.groupby("description")["amount"].sum().sort_values(ascending=False).head(5)
        facts.append("Top 5 merchants / payees by spend:\n" + "\n".join(
            f"  - {desc}: {fmt_currency(amt)}" for desc, amt in top_payees.items()
        ))

    if any(kw in q for kw in ["largest", "biggest", "highest transaction", "most expensive", "top transaction"]):
        top5 = debits.nlargest(5, "amount")[["date", "description", "amount"]]
        facts.append("Top 5 largest debit transactions:\n" + "\n".join(
            f"  - {r['description']}: {fmt_currency(r['amount'])}" for _, r in top5.iterrows()
        ))

    if any(kw in q for kw in ["net cash flow", "net flow", "savings", "saved", "net balance", "surplus"]):
        ts = debits["amount"].sum()
        tr = credits["amount"].sum()
        net = tr - ts
        facts.append(f"Net Cash Flow: {fmt_currency(net)} (Total Received: {fmt_currency(tr)} minus Total Spent: {fmt_currency(ts)})")

    if any(kw in q for kw in ["average", "mean", "avg"]):
        avg_d = debits["amount"].mean() if not debits.empty else 0.0
        facts.append(f"Average debit transaction: {fmt_currency(avg_d)} (across {len(debits)} debits)")

    if any(kw in q for kw in ["total spending", "total spend", "how much did i spend", "overall spend", "total debits"]):
        ts = debits["amount"].sum()
        facts.append(f"Total spending (all debits): {fmt_currency(ts)} across {len(debits)} transactions.")

    if any(kw in q for kw in ["total received", "total credit", "how much did i receive", "income", "salary received"]):
        tr = credits["amount"].sum()
        facts.append(f"Total received (all credits): {fmt_currency(tr)} across {len(credits)} transactions.")

    if not facts:
        STOP_WORDS = {
            "how", "much", "did", "i", "pay", "to", "total", "spent", "spend",
            "on", "for", "the", "a", "an", "is", "was", "what", "where", "who",
            "whom", "my", "in", "of", "and", "or", "through", "via", "using",
            "about", "highest", "lowest", "sum", "breakdown", "category",
            "largest", "biggest", "top", "all", "overall", "give", "show", "me",
            "list", "any", "some", "average", "mean", "tell"
        }
        words = set(re.findall(r"\b[a-z]{3,}\b", q)) - STOP_WORDS
        best_df, best_word = pd.DataFrame(), ""
        for w in words:
            hit = df[df["desc_lower"].str.contains(w, na=False)]
            if not hit.empty and len(hit) > len(best_df):
                best_df, best_word = hit, w

        if not best_df.empty:
            b_deb = best_df[best_df["type"] == "debit"]["amount"].sum()
            b_cred = best_df[best_df["type"] == "credit"]["amount"].sum()
            facts.append(f"Found {len(best_df)} transactions matching '{best_word}':")
            if b_deb > 0:
                facts.append(f"  - Total debited: {fmt_currency(b_deb)}")
            if b_cred > 0:
                facts.append(f"  - Total credited: {fmt_currency(b_cred)}")
            facts.append("Transactions:\n" + "\n".join(
                f"  - {r['description']}: {fmt_currency(r['amount'])} ({r['type']})"
                for _, r in best_df.head(6).iterrows()
            ))

    if not facts:
        ts = debits["amount"].sum()
        tr = credits["amount"].sum()
        cb = debits.groupby("category")["amount"].sum().sort_values(ascending=False).head(4)
        facts.append(f"Overall Total Spent: {fmt_currency(ts)} ({len(debits)} debits)")
        facts.append(f"Overall Total Received: {fmt_currency(tr)} ({len(credits)} credits)")
        facts.append("Top categories:\n" + "\n".join(f"  - {c}: {fmt_currency(a)}" for c, a in cb.items()))

    facts_text = "\n".join(facts)

    if not os.getenv("GROQ_API_KEY"):
        return f"{facts_text}"

    try:
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.2, max_tokens=GROQ_MAX_TOKENS)
        prompt = PromptTemplate(
            input_variables=["facts", "question"],
            template="""You are an accurate financial assistant.
The facts below were computed deterministically by Pandas and are 100% verified.
Write a clear, concise answer addressing the user's question using ONLY these facts.
Do not recalculate or estimate anything.

Verified Facts:
{facts}

User Question: {question}

Answer:""",
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"facts": facts_text, "question": question})
        return answer.strip()
    except Exception as e:
        safe_print(f"[answer_analytical_question] LLM phrasing fallback ({e})")
        return f"{facts_text}"
