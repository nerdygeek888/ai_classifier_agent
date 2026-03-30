"""
RPA + AI Hybrid Processor — 100% HuggingFace (no paid APIs)
Author: Sushma Nadkarni

Pipeline:
  Step 1: Ingest queries from CSV or sample data (RPA-style ingestion)
  Step 2: Classify each query using HuggingFace zero-shot NLP (AI layer)
  Step 3: Route based on confidence threshold (automation logic)
  Step 4: Each category gets a handler (simulates bot actions)
  Step 5: Return results with full audit trail

No API key needed — NLP model runs 100% on your local machine.
First run downloads the model (~1.6GB). After that it uses local cache.
"""

import os
import pandas as pd
from typing import Dict, List
from transformers import pipeline as hf_pipeline


# ── Sample data ───────────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    {"query_id": "QRY-001", "customer_name": "Alice Johnson",  "query_text": "I was charged twice for my last order. Please refund the extra amount."},
    {"query_id": "QRY-002", "customer_name": "Bob Smith",      "query_text": "My internet connection keeps dropping every few hours. How do I fix this?"},
    {"query_id": "QRY-003", "customer_name": "Carol White",    "query_text": "Where is my order? It was supposed to arrive 3 days ago."},
    {"query_id": "QRY-004", "customer_name": "David Lee",      "query_text": "I want to cancel my subscription and get a refund for this month."},
    {"query_id": "QRY-005", "customer_name": "Emma Brown",     "query_text": "What are your business hours and contact details?"},
    {"query_id": "QRY-006", "customer_name": "Frank Chen",     "query_text": "My invoice shows the wrong amount. The discount was not applied."},
    {"query_id": "QRY-007", "customer_name": "Grace Kim",      "query_text": "The app crashes every time I try to upload a file. Running Android 14."},
    {"query_id": "QRY-008", "customer_name": "Henry Davis",    "query_text": "I placed order ORD-8823 last week. Can you tell me the delivery status?"},
    {"query_id": "QRY-009", "customer_name": "Iris Patel",     "query_text": "You debited my account but I never received a confirmation email."},
    {"query_id": "QRY-010", "customer_name": "Jack Wilson",    "query_text": "How do I upgrade my plan to the premium tier?"},
]

# ── Categories the NLP model will classify into ───────────────────────────────
CATEGORIES = ["billing", "technical_support", "order_status", "refund", "general_enquiry"]

# ── Handlers — simulate what an RPA bot would do for each category ────────────
# In a real AA bot, these would trigger actual automations:
# billing        → create ticket in billing system
# tech_support   → raise JIRA/ServiceNow ticket
# order_status   → query order management system
# refund         → trigger refund workflow in ERP
# general        → send FAQ response via email bot
HANDLERS = {
    "billing":          lambda q: f"Billing ticket raised for {q['customer_name']}. Account flagged. ETA: 2 business days. Ref: BIL-{q['query_id'][-3:]}",
    "technical_support":lambda q: f"Tech support ticket created for {q['customer_name']}. Auto-diagnostic triggered. Ticket: TKT-{q['query_id'][-3:]}",
    "order_status":     lambda q: f"Order tracking queried for {q['customer_name']}. SMS with live status sent automatically.",
    "refund":           lambda q: f"Refund workflow initiated for {q['customer_name']}. Finance team notified. Ref: REF-{q['query_id'][-3:]}",
    "general_enquiry":  lambda q: f"FAQ auto-response sent to {q['customer_name']}. If unresolved, escalated to customer service queue.",
}


class QueryProcessor:
    """
    The core pipeline class.
    Loads the NLP model once, then processes any number of queries.
    """

    def __init__(self):
        # Load HuggingFace zero-shot classification model
        # facebook/bart-large-mnli is the standard model for this task
        # zero-shot means it can classify into ANY categories — no training needed
        print("Loading NLP model: facebook/bart-large-mnli (first run downloads ~1.6GB)...")
        self.classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # -1 = CPU. Change to 0 if you have a GPU
        )
        print("NLP model loaded and ready.")

    def get_sample_queries(self) -> pd.DataFrame:
        """Return sample data as a DataFrame for display."""
        return pd.DataFrame(SAMPLE_QUERIES)

    def classify(self, text: str) -> Dict:
        """
        Classify a single query text into one of the categories.
        Returns the top category and its confidence score (0 to 1).
        """
        result = self.classifier(text, candidate_labels=CATEGORIES)
        return {
            "category":   result["labels"][0],               # top predicted category
            "confidence": round(result["scores"][0], 4),     # confidence 0-1
            "all_scores": dict(zip(                          # all category scores
                result["labels"],
                [round(s, 4) for s in result["scores"]]
            ))
        }

    def process(self, df: pd.DataFrame, confidence_threshold: float = 0.75) -> pd.DataFrame:
        """
        Run the full pipeline on a DataFrame of queries.

        For each query:
          1. Classify with NLP
          2. If confidence >= threshold → auto-process via handler
          3. If confidence < threshold → flag for human review (Human-in-the-Loop)

        Returns a new DataFrame with all results appended.
        """
        results = []

        for _, row in df.iterrows():
            query = row.to_dict()

            # Step 2: AI layer — classify the query
            classification = self.classify(query["query_text"])

            # Step 3: Routing logic
            if classification["confidence"] >= confidence_threshold:
                # High confidence → automate it
                handler  = HANDLERS.get(classification["category"], HANDLERS["general_enquiry"])
                response = handler(query)
                status   = "auto_processed"
            else:
                # Low confidence → human review (Human-in-the-Loop)
                response = (
                    f"Low confidence ({classification['confidence']:.0%}) — "
                    f"flagged for human review. "
                    f"Predicted category: {classification['category']}"
                )
                status = "flagged_for_review"

            results.append({
                **query,
                "category":        classification["category"],
                "confidence":      classification["confidence"],
                "all_scores":      str(classification["all_scores"]),
                "status":          status,
                "handler_response": response
            })

        return pd.DataFrame(results)
