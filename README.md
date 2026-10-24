# 💰 FinDoc AI — Financial Document Assistant

A **RAG-powered chatbot** that lets you ask natural language questions about your bank statements. Upload a PDF or CSV and start chatting immediately.

> **Built to demonstrate:** LangChain · ChromaDB · Groq LLM · HuggingFace Embeddings · Pandas · Streamlit

---

## 🎥 Demo

| Dashboard | Chat Interface |
|-----------|----------------|
| Upload a bank statement → instant metrics, spending chart, and transaction table | Ask anything in plain English — lookup facts or aggregate totals |

**Sample questions you can ask:**
- *"What is my closing balance?"*
- *"How much did I spend on food?"*
- *"Who did I pay the most?"*
- *"How much did I receive via UPI?"*
- *"What is the account holder name?"*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Question                          │
└───────────────────┬─────────────────────────────────────────┘
                    │
          ┌─────────▼─────────┐
          │  Question Router  │  (keyword detection)
          └──────┬──────┬─────┘
                 │      │
    ┌────────────▼─┐  ┌─▼──────────────────┐
    │  RAG Path    │  │  Analytical Path    │
    │              │  │                    │
    │  ChromaDB    │  │  Pandas groupby /  │
    │  similarity  │  │  filter / sum      │
    │  search      │  │  (exact math)      │
    │              │  │        │           │
    │  Top-4 chunks│  │  Pre-calculated    │
    │      │       │  │  facts             │
    └──────┬───────┘  └────────┬───────────┘
           │                   │
           └─────────┬─────────┘
                     │
            ┌────────▼────────┐
            │   Groq LLM      │  (phrases the answer)
            └────────┬────────┘
                     │
              Natural language answer
```

### Why two paths?

| Path | Used for | Why not just LLM? |
|------|----------|-------------------|
| **RAG (ChromaDB)** | Lookup facts: account name, IFSC, balance | Retrieves only relevant chunks from large docs |
| **Pandas** | Aggregations: totals, groupby, top-N | LLMs are unreliable at arithmetic across 100s of rows |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Groq (`openai/gpt-oss-120b`) | Fast inference, structured extraction, answer phrasing |
| **Orchestration** | LangChain | RAG chain construction, prompt templates |
| **Vector Database** | ChromaDB | Local persistent vector store for embeddings |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Free, local, 384-dim sentence embeddings |
| **Analytics** | Pandas | Deterministic groupby/filter/sum on transactions |
| **UI** | Streamlit | Rapid Python web app with chat interface |
| **PDF Parsing** | pdfplumber + pikepdf | Layout-aware text extraction + password unlock |
| **Visualization** | Plotly Express | Interactive spending-by-category bar chart |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/findoc-ai.git
cd findoc-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
findoc-ai/
│
├── app.py              # Streamlit UI, routing logic, dashboard
├── ingest.py           # PDF/CSV text extraction, ChromaDB indexing,
│                       # structured transaction extraction (LLM)
├── rag_chain.py        # RAG lookup chain + pandas analytics engine
│
├── requirements.txt    # All Python dependencies
├── .env                # API keys (not committed to git)
├── .gitignore
│
├── sample_data/
│   └── sample_statement.csv   # Demo bank statement for testing
│
└── chroma_db/          # Auto-created: local vector store (gitignored)
```

---

## 🔑 Key Concepts Demonstrated

### 1. Retrieval-Augmented Generation (RAG)
The document is too large for the LLM's context window. Instead:
1. Text is split into 500-char overlapping chunks
2. Each chunk is embedded as a 384-dim vector using `all-MiniLM-L6-v2`
3. At query time, the top-4 most semantically similar chunks are retrieved
4. Only those chunks are sent to the LLM as context

### 2. Structured Data Extraction
Beyond RAG, the LLM is prompted to parse the raw bank statement into a structured JSON array — extracting `date`, `description`, `amount`, `type`, `category`, and `payment_method` for every transaction. This JSON is loaded into a Pandas DataFrame.

### 3. Pandas-First Analytics
For aggregation questions ("how much did I spend on food?"), **Pandas does all the math** — the LLM never calculates. This is critical for correctness: LLMs fail on arithmetic across large datasets.

### 4. Smart Question Routing
The app detects whether a question is:
- **Analytical** (contains "how much", "total", "UPI", "spent", etc.) → Pandas path
- **Lookup** (factual) → RAG path

---

## ⚙️ Environment & Rate Limits

This app uses **Groq's free tier**. The `openai/gpt-oss-120b` model is used with `max_tokens` capped to stay under OTPM limits. A 3-model fallback chain (`120b → qwen3.8-27b → gpt-oss-20b`) ensures extraction succeeds even when one model is rate-limited.

---

## 🔒 Privacy Note

Your bank statement is processed **entirely locally** — no data is sent anywhere except to Groq's API for LLM inference. ChromaDB stores embeddings on your machine in the `./chroma_db` folder.

---

## 📝 License

MIT
