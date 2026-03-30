# ⚙️ RPA + AI Hybrid Customer Query Processor

An enterprise automation pipeline that combines **RPA-style data ingestion and routing** with an **AI/NLP classification layer** — exactly the pattern used in real-world Automation Anywhere + IQ Bot / NLP integrations.

This is Sushma's **differentiator project** — pure AI engineers can't build this, and pure RPA engineers wouldn't think of it.

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│              RPA + AI HYBRID PIPELINE                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: INGEST (RPA layer)                             │
│  ┌──────────────────────┐                               │
│  │ Read CSV / database  │  ← Simulates AA bot reading   │
│  │ Parse query records  │    shared mailbox or DB       │
│  └──────────┬───────────┘                               │
│             │                                           │
│  Step 2: CLASSIFY (AI layer)                            │
│  ┌──────────▼───────────┐                               │
│  │ HuggingFace NLP      │  ← Zero-shot classification   │
│  │ Zero-shot classifier │    no training data needed    │
│  │ (bart-large-mnli)    │                               │
│  └──────────┬───────────┘                               │
│             │                                           │
│  Step 3: ROUTE (RPA logic)                              │
│  ┌──────────▼───────────┐                               │
│  │ Confidence check     │  ← Threshold-based routing    │
│  │ ≥ threshold → auto   │    (Human-in-the-Loop)        │
│  │ < threshold → flag   │                               │
│  └──────────┬───────────┘                               │
│             │                                           │
│  Step 4: ACT (Handler automation)                       │
│  ┌──────────▼───────────┐                               │
│  │ Billing handler      │                               │
│  │ Tech support handler │  ← Simulates AA bot actions   │
│  │ Order status handler │    (ticket creation, DB       │
│  │ Refund handler       │    updates, notifications)    │
│  │ General enquiry      │                               │
│  └──────────┬───────────┘                               │
│             │                                           │
│  Step 5: REPORT (RPA output)                            │
│  ┌──────────▼───────────┐                               │
│  │ Summary dashboard    │  ← Simulates AA report bot    │
│  │ CSV export           │                               │
│  │ HTML report          │                               │
│  └──────────────────────┘                               │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 📥 **CSV ingestion** — upload your own or use built-in sample data
- 🤖 **Zero-shot NLP** — classifies any query without training data
- 🔀 **Confidence-based routing** — adjustable threshold for human-in-the-loop
- 📊 **Live dashboard** — KPIs, category breakdown, full results table
- ⬇️ **CSV export** — download processed results
- 📋 **Audit trail** — every decision logged with confidence score

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rpa-ai-hybrid-processor.git
cd rpa-ai-hybrid-processor
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> Note: First run downloads the HuggingFace model (~1.6GB). Subsequent runs use the cache.

### 4. Run
```bash
streamlit run app.py
```

**No API key needed** — NLP model runs fully locally.

---

## 📁 Project Structure

```
project3-rpa-ai-hybrid/
├── app.py                  # Streamlit UI (3-tab pipeline view)
├── processor.py            # Core pipeline: ingest → classify → route → handle
├── report_generator.py     # HTML report generation
├── sample_queries.csv      # Sample CSV for testing uploads
├── requirements.txt
└── README.md
```

---

## 🧪 Classification Categories

| Category | Example Query |
|----------|--------------|
| `billing` | "I was charged twice for my order" |
| `technical_support` | "My app crashes on startup" |
| `order_status` | "Where is my order ORD-5521?" |
| `refund` | "I want a refund for this month" |
| `general_enquiry` | "What are your business hours?" |

---

## 🔑 Key Design Decisions

- **Zero-shot classification**: Uses `facebook/bart-large-mnli` — no labelled training data required, works out of the box for new categories
- **Confidence threshold**: Configurable via UI slider — queries below threshold go to human review, above threshold are auto-processed
- **Human-in-the-Loop**: Low-confidence queries are never auto-acted on — mirrors enterprise compliance requirements
- **Local model**: No external API calls for NLP — reduces cost and latency, increases data privacy

---

## 🆚 Why This Project Stands Out

Most GitHub AI projects are either pure ML notebooks or simple chatbots. This project demonstrates:
1. **Enterprise automation thinking** — structured pipeline with audit trail
2. **Hybrid AI + RPA pattern** — the real-world pattern, not academic demos
3. **Production considerations** — Human-in-the-Loop, confidence thresholds, error logging
4. **Business context** — solves a real Finance/Operations problem

---

## 📌 Author

**Sushma Nadkarni** — RPA Solutions Lead & AI Automation Engineer  
[LinkedIn](https://linkedin.com/in/yourprofile)
