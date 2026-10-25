# 💰 FinDoc AI — RAG-Powered Bank Statement Analyser

**FinDoc AI** is a financial intelligence and bank statement analysis application built with **Streamlit**, **LangChain**, **ChromaDB**, **Groq LLM**, and **Pandas**.

It features a **dual-engine architecture** that combines **deterministic Pandas analytics** (ensuring 100% mathematical accuracy for calculations, totals, and monthly cash flow) with **Retrieval-Augmented Generation (RAG)** for grounded semantic questions (account numbers, IFSC codes, transaction policies, and document facts).

---

## 🌟 Key Features

- **🚀 One-Click Sample Statement**: Recruiters and evaluators can test the complete system with a single click using our realistic, verified 3-month demo bank statement — no file uploads or API keys required to view the dashboard and query financial metrics.
- **📊 Deterministic Financial Analytics**: Numerical totals, net cash flow, monthly comparisons, category breakdowns, and top payees are calculated directly in Python/Pandas to eliminate LLM arithmetic hallucinations.
- **🔍 Grounded Document Q&A (RAG)**: Semantic and account-level questions are answered using ChromaDB vector similarity search with local HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **🛡️ Intelligent Data Validation & Reconciliation**: Automatically parses statement metadata, flags corrupt rows, removes exact duplicates, and verifies balance reconciliation:
  $$\text{Closing Balance} = \text{Opening Balance} + \text{Total Credits} - \text{Total Debits}$$
- **📂 Multi-Format Ingestion**: Supports CSV bank exports (SBI, HDFC, ICICI, Axis, standard international schemas) and digital PDF statements with password-unlock support via `pikepdf`.
- **🔒 Session & Vector Isolation**: In-memory and session-keyed ChromaDB collections ensure that switching documents never leaks data across user sessions or files.

---

## 🏗️ Architecture & Data Flow

```
                              ┌─────────────────────────────┐
                              │     User Query / Prompt     │
                              └──────────────┬──────────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │  Question Router  │
                                   └────┬───────────┬──┘
                                        │           │
                 ┌──────────────────────┘           └──────────────────────┐
                 │ [Analytical / Numerical]                [Semantic / Fact Lookup]
                 ▼                                                         ▼
    ┌─────────────────────────┐                               ┌─────────────────────────┐
    │   Pandas Analytics      │                               │   ChromaDB Vector Store │
    │   - Sums / Averages     │                               │   - Local MiniLM Embeds │
    │   - Monthly Breakdowns  │                               │   - Similarity Search   │
    │   - Top Payees / Spends │                               │   - Top-4 Context Chunks│
    └────────────┬────────────┘                               └────────────┬────────────┘
                 │ (Deterministic Facts)                                   │ (Document Chunks)
                 └──────────────────────┐           ┌──────────────────────┘
                                        │           │
                                   ┌────▼───────────▼──┐
                                   │     Groq LLM      │  (Phrases response or
                                   │  (openai/gpt-oss) │   offline facts fallback)
                                   └─────────┬─────────┘
                                             │
                                             ▼
                                  Grounded, Verified Answer
```

### Why a Dual-Engine Design?

| Engine | Query Type | Why this approach? |
| :--- | :--- | :--- |
| **Pandas (Deterministic)** | "What is my total spending?", "How much in July?", "Net cash flow" | LLMs frequently make arithmetic errors over large tabular datasets. Pandas guarantees 100% calculation accuracy. |
| **ChromaDB (RAG)** | "What is the account holder name?", "What is the IFSC code?", "Branch location" | Chunks and embeds the raw document locally; retrieves only the exact paragraphs containing account details. |

---

## 🛠️ Technology Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) | Clean dark-navy financial dashboard, interactive filters, chat interface |
| **Data Engine** | [Pandas](https://pandas.pydata.org/) | Deterministic aggregations, CSV parsing, data validation, reconciliation |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Persistent & session-isolated local vector store |
| **Embeddings** | [HuggingFace](https://huggingface.co/) (`all-MiniLM-L6-v2`) | Local 384-dimensional sentence transformer embeddings (free & offline) |
| **LLM Inference** | [Groq](https://groq.com/) (`openai/gpt-oss-120b`) | High-speed LLM inference for natural language phrasing and PDF parsing |
| **PDF Extraction** | [pdfplumber](https://github.com/jsvine/pdfplumber) + [pikepdf](https://github.com/pikepdf/pikepdf) | Layout-aware PDF text extraction and encrypted document unlocking |
| **Visualizations** | [Plotly Express](https://plotly.com/python/) | Interactive spending-by-category and monthly cash flow charts |

---

## 📊 Using the Application

### Option A: Try with Sample Data (Instant Recruiter Demo)

1. Open the application in your browser.
2. Under **"🚀 Option A: Try with Sample Data"**, click **"Load Sample Bank Statement"**.
3. The dashboard instantly renders:
   - 45 synthetic transactions spanning **May 1 to July 31, 2026**.
   - Verified totals: Total Debits (Rs.91,054.00), Total Credits (Rs.179,812.10), Net Cash Flow (+Rs.88,758.10).
   - Balance reconciliation verification against opening balance (Rs.15,000.00) and closing balance (Rs.103,758.10).
   - Interactive Plotly charts: Spending by Category and Monthly Cash Flow Comparison.
   - Filterable, searchable transaction explorer.
   - Instant suggested question chips.

### Option B: Upload Your Own Statement

1. Select **"📂 Option B: Upload Your Own Statement"**.
2. Upload a **CSV** or digital **PDF** statement (up to 15MB).
3. If your PDF statement is password-protected, enter the document password when prompted.
4. The system automatically processes the file and updates all metrics.

---

## 📋 Supported CSV Schemas

The deterministic CSV parser automatically normalizes column names and removes currency symbols (`$`, `₹`, `Rs.`, `,`). Supported column layouts include:

### Layout 1: Separate Debit & Credit Columns (Standard Bank Export)
```csv
Date,Description,Debit,Credit,Balance
2026-05-01,Salary TechCorp,,55000.00,70000.00
2026-05-03,Rent Landlord,15000.00,,55000.00
2026-05-04,Swiggy Order,420.00,,54580.00
```

### Layout 2: Combined Amount & Type Column
```csv
Date,Description,Amount,Type
2026-07-02,Salary Credit NEFT,52000.00,Credit
2026-07-03,UPI/Ashish Kumar/Rent,12000.00,Debit
2026-07-04,UPI/Swiggy/Order#4821,349.00,Debit
```

---

## 💬 Example Questions

### Analytical Questions (Deterministic Pandas)
- *"What is my total spending?"*
- *"How much did I receive?"*
- *"What is my net cash flow?"*
- *"How much did I spend in July?"*
- *"Who did I pay the most?"*
- *"Show my largest transactions."*
- *"How much did I spend on food?"*
- *"How much did I pay via UPI?"*
- *"What is my average debit transaction?"*

### Semantic Lookup Questions (ChromaDB RAG)
- *"What is the account holder name?"*
- *"What is the IFSC code?"*
- *"Which bank branch issued this statement?"*
- *"What is the statement period?"*

---

## 🧪 Automated Testing

FinDoc AI includes a unit test suite built with Python's built-in `unittest` framework. The tests run completely offline without requiring internet access or a Groq API key:

```bash
python -m unittest discover tests
```

### What the test suite verifies:
- `test_demo_bank_statement_integrity_and_totals`: Verifies 45 transactions, exact credits (Rs.179,812.10), debits (Rs.91,054.00), and net cash flow (Rs.88,758.10).
- `test_sample_statement_parsing_and_metadata`: Verifies header/footer metadata extraction and transaction normalization.
- `test_empty_and_corrupt_csv`: Ensures invalid dates and malformed amounts are dropped safely with validation warnings.
- `test_duplicate_row_removal`: Verifies that duplicate transactions are dropped and reported.
- `test_reconciliation_exact_and_discrepancy`: Tests closing balance reconciliation logic.
- `test_currency_symbol_and_formatting_cleanup`: Verifies normalization of strings with `$`, `₹`, `Rs.`, commas, and whitespace.
- `test_category_inference`: Tests keyword rules across food, rent, shopping, health, entertainment, transport, and utilities.
- `test_deterministic_analytics`: Tests math answers for total spending, net cash flow, monthly spending, top payees, and merchant filters.
- `test_session_state_isolation_concept`: Verifies dataset separation between documents.

---

## 📁 Repository Structure

```
findoc-ai/
│
├── app.py                     # Streamlit web application & visual dashboard
├── ingest.py                  # Deterministic CSV parser, vector store indexing, LLM extractor
├── rag_chain.py               # Deterministic Pandas analytics engine & ChromaDB RAG chain
│
├── requirements.txt           # Python package dependencies
├── .env.example               # Example configuration file
├── .gitignore                 # Git ignore rules
│
├── data/
│   └── demo_bank_statement.csv # 3-month verified synthetic demo statement
│
├── sample_data/
│   └── sample_statement.csv   # Legacy sample statement
│
├── tests/
│   └── test_findoc.py         # Automated unit & integration test suite
│
└── chroma_db/                 # Auto-generated local vector database (gitignored)
```

---

## 🔒 Security & Privacy Disclosures

- **Local Storage**: All vector embeddings are generated and stored locally on your machine in the `./chroma_db` directory.
- **Data Isolation**: Each document session generates an isolated vector collection, preventing cross-statement vector mixing.
- **External API Calls**: When `GROQ_API_KEY` is configured, only relevant query text and pre-calculated summary facts (or statement chunks for RAG) are transmitted to Groq's API for phrasing. Numerical arithmetic is never outsourced to the external LLM.
- **Credentials**: API keys and personal files are excluded from version control via `.gitignore`.

---

## ⚠️ Known Limitations

1. **Scanned Documents (OCR)**: Scanned image-only PDFs without an embedded text layer require OCR preprocessing before ingestion.
2. **Groq Free-Tier Rate Limits**: For very long unstructured PDF statements, Groq free-tier rate limits may apply. The system includes an automatic fallback chain and heuristic regex recovery to prevent application failure.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for details.
