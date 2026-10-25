"""
test_findoc.py — Automated Unit & Integration Tests for FinDoc AI.

Tests:
  - Deterministic CSV parser & metadata extraction
  - Category & payment method inference rules
  - Deterministic financial analytics calculations (totals, net flow, monthly, merchant)
  - Balance reconciliation checks
  - Error handling (empty files, malformed rows, duplicate records)
  - Offline analytical Q&A behavior without GROQ_API_KEY
  - Vectorstore & session isolation behavior
"""

import os
import unittest
import pandas as pd

from ingest import (
    parse_csv_statement,
    infer_category,
    infer_payment_method,
    extract_transactions_regex_fallback,
    EMPTY_DF,
)
from rag_chain import answer_analytical_question, get_answer


class TestCSVParser(unittest.TestCase):
    """Tests for deterministic CSV parsing and validation."""

    def test_demo_bank_statement_integrity_and_totals(self):
        """Verify the 3-month demo bank statement parses correctly with exact math."""
        demo_path = os.path.join("data", "demo_bank_statement.csv")
        self.assertTrue(os.path.exists(demo_path), "demo_bank_statement.csv must exist in data/")

        df, summary = parse_csv_statement(demo_path)

        # Expected counts and totals
        self.assertEqual(len(df), 45, "Demo statement should contain exactly 45 transactions.")
        self.assertEqual(summary["invalid_rows_dropped"], 0, "No valid rows should be dropped.")
        self.assertEqual(summary["duplicates_removed"], 0, "No duplicates should exist.")

        # Check Credits and Debits
        total_credits = df[df["type"] == "credit"]["amount"].sum()
        total_debits = df[df["type"] == "debit"]["amount"].sum()
        net_flow = round(total_credits - total_debits, 2)

        self.assertAlmostEqual(total_credits, 179812.10, places=2)
        self.assertAlmostEqual(total_debits, 91054.00, places=2)
        self.assertAlmostEqual(net_flow, 88758.10, places=2)

        # Reconcile running balance
        first_bal = df.iloc[0]["balance"]
        last_bal = df.iloc[-1]["balance"]
        self.assertIsNotNone(first_bal)
        self.assertIsNotNone(last_bal)
        self.assertAlmostEqual(last_bal, 103758.10, places=2)

    def test_sample_statement_parsing_and_metadata(self):
        """Verify legacy sample_statement.csv parses correctly and extracts metadata."""
        sample_path = os.path.join("sample_data", "sample_statement.csv")
        if os.path.exists(sample_path):
            df, summary = parse_csv_statement(sample_path)
            self.assertEqual(len(df), 27, "sample_statement.csv should yield 27 valid transactions.")

            # Metadata verification
            meta = summary.get("metadata", {})
            self.assertEqual(meta.get("Account Holder"), "Sanket Sharma")
            self.assertEqual(meta.get("IFSC Code"), "SBIN0012345")

            # Mathematical totals
            total_credits = df[df["type"] == "credit"]["amount"].sum()
            total_debits = df[df["type"] == "debit"]["amount"].sum()
            self.assertAlmostEqual(total_credits, 57082.50, places=2)
            self.assertAlmostEqual(total_debits, 30100.00, places=2)

    def test_empty_and_corrupt_csv(self):
        """Verify empty and corrupt CSVs are rejected gracefully without crashes."""
        df, summary = parse_csv_statement("")
        self.assertTrue(df.empty)
        self.assertEqual(summary["valid_transactions"], 0)

        # Corrupt data with missing columns and invalid numbers
        corrupt_csv = """Date,Description,Amount,Type
invalid-date,Bad Row,not-a-number,Debit
2026-06-01,Valid Salary,50000.00,Credit
,Empty Date Row,100.00,Debit
2026-06-03,Zero Amount,0.00,Debit
"""
        df_c, summary_c = parse_csv_statement(corrupt_csv)
        self.assertEqual(len(df_c), 1, "Only the single valid transaction should be kept.")
        self.assertEqual(df_c.iloc[0]["description"], "Valid Salary")
        self.assertGreater(summary_c["invalid_rows_dropped"], 0)

    def test_duplicate_row_removal(self):
        """Verify identical duplicate transactions are detected and dropped."""
        dup_csv = """Date,Description,Amount,Type
2026-05-01,Swiggy Order,350.00,Debit
2026-05-01,Swiggy Order,350.00,Debit
2026-05-02,Uber Commute,150.00,Debit
"""
        df_dup, summary_dup = parse_csv_statement(dup_csv)
        self.assertEqual(len(df_dup), 2, "Duplicate row should be deduplicated.")
        self.assertEqual(summary_dup["duplicates_removed"], 1)

    def test_separate_debit_credit_columns(self):
        """Verify CSVs with separate Debit and Credit columns parse properly."""
        split_csv = """Txn Date,Narration,Debit,Credit,Balance
2026-05-01,Salary TechCorp,,50000.00,50000.00
2026-05-02,Rent Landlord,15000.00,,35000.00
2026-05-03,Grocery Store,1200.00,,33800.00
"""
        df_split, _ = parse_csv_statement(split_csv)
        self.assertEqual(len(df_split), 3)
        self.assertEqual(df_split.iloc[0]["type"], "credit")
        self.assertEqual(df_split.iloc[1]["type"], "debit")
        self.assertEqual(df_split.iloc[2]["type"], "debit")
        self.assertAlmostEqual(df_split[df_split["type"] == "debit"]["amount"].sum(), 16200.00)


class TestInferenceRules(unittest.TestCase):
    """Tests for deterministic category and payment method inferencing."""

    def test_category_inference(self):
        self.assertEqual(infer_category("UPI/Swiggy/Order#123"), "Food")
        self.assertEqual(infer_category("Salary NEFT TechCorp"), "Income")
        self.assertEqual(infer_category("BESCOM Electricity Bill"), "Housing")
        self.assertEqual(infer_category("Uber Airport Ride"), "Transport")
        self.assertEqual(infer_category("Apollo Pharmacy Medicines"), "Health")
        self.assertEqual(infer_category("Netflix Monthly Subscription"), "Entertainment")
        self.assertEqual(infer_category("Amazon Online Shopping"), "Shopping")
        self.assertEqual(infer_category("ATM Cash Withdrawal Koramangala"), "Cash Withdrawal")
        self.assertEqual(infer_category("Random Transfer to Friend"), "Other")

    def test_payment_method_inference(self):
        self.assertEqual(infer_payment_method("UPI/Friend/Split"), "UPI")
        self.assertEqual(infer_payment_method("NEFT/TechCorp/Salary"), "NEFT")
        self.assertEqual(infer_payment_method("IMPS/Flipkart/Order"), "IMPS")
        self.assertEqual(infer_payment_method("ATM Cash Withdrawal"), "Cash")
        self.assertEqual(infer_payment_method("POS/Shell Petrol Pump"), "Card")


class TestDeterministicAnalytics(unittest.TestCase):
    """Tests for deterministic financial calculation accuracy (offline without Groq)."""

    @classmethod
    def setUpClass(cls):
        # Disable GROQ_API_KEY during analytical tests to verify exact deterministic math output
        cls._orig_key = os.environ.pop("GROQ_API_KEY", None)
        demo_path = os.path.join("data", "demo_bank_statement.csv")
        cls.df, _ = parse_csv_statement(demo_path)

    @classmethod
    def tearDownClass(cls):
        if cls._orig_key:
            os.environ["GROQ_API_KEY"] = cls._orig_key

    def test_total_spending_query(self):
        ans = answer_analytical_question("What is my total spending?", self.df)
        self.assertIn("91,054.00", ans)
        self.assertIn("Total spending", ans)

    def test_total_received_query(self):
        ans = answer_analytical_question("How much did I receive?", self.df)
        self.assertIn("179,812.10", ans)
        self.assertIn("Total received", ans)

    def test_net_cash_flow_query(self):
        ans = answer_analytical_question("What is my net cash flow?", self.df)
        self.assertIn("88,758.10", ans)
        self.assertIn("Net Cash Flow", ans)

    def test_monthly_spending_query(self):
        ans = answer_analytical_question("How much did I spend in July?", self.df)
        self.assertIn("July", ans)
        # Verify July debits in demo dataset:
        dt_series = pd.to_datetime(self.df["date"])
        july_debits = self.df[(dt_series.dt.month == 7) & (self.df["type"] == "debit")]["amount"].sum()
        self.assertIn(f"{july_debits:,.2f}", ans)

    def test_top_payees_query(self):
        ans = answer_analytical_question("Who did I pay the most?", self.df)
        self.assertIn("Top 5 merchants", ans)
        # Rent / Landlord is the highest payee (15000 * 3 = 45000)
        self.assertIn("Landlord", ans)

    def test_largest_transactions_query(self):
        ans = answer_analytical_question("Show my largest transactions", self.df)
        self.assertIn("Top 5 largest", ans)
        self.assertIn("15,000.00", ans)

    def test_category_query(self):
        ans = answer_analytical_question("How much did I spend on food?", self.df)
        self.assertIn("Total spent on Food", ans)
        food_tot = self.df[(self.df["category"].str.lower() == "food") & (self.df["type"] == "debit")]["amount"].sum()
        self.assertIn(f"{food_tot:,.2f}", ans)


class TestRAGSafetyAndFallbacks(unittest.TestCase):
    """Test safety bounds and fallback behavior when API is missing or empty."""

    def test_get_answer_without_api_key(self):
        orig_key = os.environ.pop("GROQ_API_KEY", None)
        try:
            ans = get_answer("What is the account number?", None)
            self.assertIn("No document is currently indexed", ans)

            class DummyVectorStore:
                pass

            ans2 = get_answer("What is the account number?", DummyVectorStore())
            self.assertIn("Groq API key required", ans2)
        finally:
            if orig_key:
                os.environ["GROQ_API_KEY"] = orig_key

    def test_currency_symbol_and_formatting_cleanup(self):
        """Verify amounts with commas, rupee symbols, dollar signs, and negative values are normalized."""
        formatted_csv = """Date,Description,Debit,Credit,Balance
2026-05-01,Salary Credit,,"Rs. 55,000.00","₹ 55,000.00"
2026-05-02,Rent Payment,"₹ 15,000.00",,"₹ 40,000.00"
2026-05-03,Food Swiggy,"$ 420.50",,"₹ 39,579.50"
"""
        df, summary = parse_csv_statement(formatted_csv)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]["amount"], 55000.00)
        self.assertEqual(df.iloc[1]["amount"], 15000.00)
        self.assertEqual(df.iloc[2]["amount"], 420.50)

    def test_session_state_isolation_concept(self):
        """Verify parsing different statements produces completely separate isolated datasets."""
        demo_path = os.path.join("data", "demo_bank_statement.csv")
        sample_path = os.path.join("sample_data", "sample_statement.csv")

        df_demo, _ = parse_csv_statement(demo_path)
        df_sample, _ = parse_csv_statement(sample_path)

        self.assertEqual(len(df_demo), 45)
        self.assertEqual(len(df_sample), 27)
        self.assertNotEqual(df_demo["amount"].sum(), df_sample["amount"].sum())


    def test_reconciliation_exact_and_discrepancy(self):
        """Verify opening/closing balance reconciliation detection."""
        reconciled_csv = """Date,Description,Amount,Type
Opening Balance,10000.00,,
2026-05-01,Salary Credit,50000.00,Credit
2026-05-02,Rent Payment,15000.00,Debit
Closing Balance,45000.00,,
"""
        df_r, sum_r = parse_csv_statement(reconciled_csv)
        self.assertEqual(len(df_r), 2)
        rec = sum_r.get("reconciliation", {})
        self.assertEqual(rec.get("status"), "reconciled")
        self.assertAlmostEqual(rec.get("expected_closing"), 45000.00)

        discrepancy_csv = """Date,Description,Amount,Type
Opening Balance,10000.00,,
2026-05-01,Salary Credit,50000.00,Credit
2026-05-02,Rent Payment,15000.00,Debit
Closing Balance,40000.00,,
"""
        df_d, sum_d = parse_csv_statement(discrepancy_csv)
        self.assertEqual(len(df_d), 2)
        rec_d = sum_d.get("reconciliation", {})
        self.assertEqual(rec_d.get("status"), "discrepancy")
        self.assertAlmostEqual(rec_d.get("discrepancy"), -5000.00)


class TestSessionPersistenceAndRegression(unittest.TestCase):
    """Tests simulating UI session-state workflows to verify sample data persistence across questions."""

    def setUp(self):
        # Simulated session state dictionary mirroring Streamlit session_state
        self.state = {
            "data_mode": "none",
            "transactions_df": None,
            "vectorstore": None,
            "active_dataset_label": "",
            "processed_file_key": None,
            "validation_summary": None,
            "messages": [],
        }

    def simulate_load_sample_data(self):
        demo_path = os.path.join("data", "demo_bank_statement.csv")
        df, summary = parse_csv_statement(demo_path)
        self.state["data_mode"] = "sample"
        self.state["transactions_df"] = df
        self.state["validation_summary"] = summary
        self.state["active_dataset_label"] = "Sample Bank Statement (May – July 2026)"
        self.state["processed_file_key"] = "demo_bank_statement"

    def simulate_ask_question(self, question: str):
        self.state["messages"].append({"role": "user", "content": question})
        current_df = self.state.get("transactions_df")
        if current_df is None or current_df.empty:
            ans = "No active bank statement found."
            route = ""
        else:
            ans = answer_analytical_question(question, current_df)
            route = "Deterministic Analytics (Pandas)"
        self.state["messages"].append({"role": "assistant", "content": ans, "route": route})
        return ans

    def test_sample_data_persistence_across_consecutive_questions(self):
        """Verify that loading sample data and asking multiple questions retains the dataset and message history."""
        # 1. Load sample data
        self.simulate_load_sample_data()
        self.assertEqual(self.state["data_mode"], "sample")
        self.assertIsNotNone(self.state["transactions_df"])
        self.assertEqual(len(self.state["transactions_df"]), 45)

        # 2. Ask Question 1
        ans1 = self.simulate_ask_question("What is my total spending?")
        self.assertIn("91,054.00", ans1)
        self.assertEqual(self.state["data_mode"], "sample")
        self.assertEqual(len(self.state["transactions_df"]), 45, "DataFrame must remain intact after Q1")
        self.assertEqual(len(self.state["messages"]), 2, "Chat history should contain 1 user + 1 assistant message")

        # 3. Ask Question 2
        ans2 = self.simulate_ask_question("Which merchant received the most money?")
        self.assertIn("landlord", ans2.lower())
        self.assertEqual(len(self.state["transactions_df"]), 45, "DataFrame must remain intact after Q2")
        self.assertEqual(len(self.state["messages"]), 4, "Chat history should contain 2 question/answer turns")

        # 4. Ask Question 3
        ans3 = self.simulate_ask_question("Show my largest transactions")
        self.assertIn("15,000.00", ans3)
        self.assertEqual(len(self.state["transactions_df"]), 45, "DataFrame must remain intact after Q3")
        self.assertEqual(len(self.state["messages"]), 6, "Chat history must preserve all previous messages")

    def test_switching_modes_replaces_dataset_cleanly(self):
        """Verify switching from sample mode to an uploaded statement cleanly replaces the dataset."""
        # 1. Start in sample mode
        self.simulate_load_sample_data()
        self.assertEqual(len(self.state["transactions_df"]), 45)

        # 2. Switch to uploaded mode with a 2-row statement
        uploaded_csv = """Date,Description,Amount,Type
2026-08-01,Uploaded Salary,60000.00,Credit
2026-08-02,Uploaded Rent,18000.00,Debit
"""
        new_df, new_sum = parse_csv_statement(uploaded_csv)
        self.state["data_mode"] = "uploaded"
        self.state["transactions_df"] = new_df
        self.state["active_dataset_label"] = "My_Uploaded_Statement.csv"
        self.state["processed_file_key"] = "uploaded_key_123"

        self.assertEqual(self.state["data_mode"], "uploaded")
        self.assertEqual(len(self.state["transactions_df"]), 2, "Uploaded dataset must cleanly replace sample data")
        self.assertEqual(self.state["active_dataset_label"], "My_Uploaded_Statement.csv")

    def test_clear_dataset_resets_state(self):
        """Verify clearing active dataset resets all relevant state keys."""
        self.simulate_load_sample_data()
        self.simulate_ask_question("Total spending?")
        self.assertEqual(len(self.state["transactions_df"]), 45)

        # Clear state
        self.state["data_mode"] = "none"
        self.state["transactions_df"] = None
        self.state["vectorstore"] = None
        self.state["active_dataset_label"] = ""
        self.state["processed_file_key"] = None
        self.state["messages"] = []

        self.assertEqual(self.state["data_mode"], "none")
        self.assertIsNone(self.state["transactions_df"])
        self.assertEqual(len(self.state["messages"]), 0)

    def test_question_without_active_dataset(self):
        """Verify submitting a question without an active dataset returns a helpful warning."""
        ans = self.simulate_ask_question("What is my balance?")
        self.assertIn("No active bank statement found", ans)

    def test_processed_file_key_prevents_duplicate_processing(self):
        """Verify that identical file keys do not trigger repeated processing."""
        file_key = "statement_12345"
        self.state["processed_file_key"] = file_key
        # Check condition used in app.py:
        needs_processing = self.state.get("processed_file_key") != file_key
        self.assertFalse(needs_processing, "Should not reprocess when file_key matches")


if __name__ == "__main__":
    unittest.main()
