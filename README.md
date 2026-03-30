# ⚙️ AI Customer Query Classifier

An end-to-end **NLP classification pipeline** that automatically reads customer queries, understands what they are asking for, and routes them to the right team — without any manual effort.

Built with **HuggingFace Transformers** and **Streamlit**. No paid APIs. Runs on your laptop.

---

## 🤔 What Problem Does This Solve?

Every business receives hundreds of customer queries daily:
- *"I was charged twice"*
- *"My app keeps crashing"*
- *"Where is my order?"*

Traditionally, a human reads each query and decides where to send it. This is slow, expensive, and inconsistent.

This project automates that decision using **Zero-Shot NLP Classification** — the AI reads each query and decides the category on its own, with no prior training on your specific data.

---

## 🧠 How It Works

```
Customer Query (text)
        ↓
HuggingFace NLP Model
(facebook/bart-large-mnli)
        ↓
Predicted Category + Confidence Score
        ↓
    ┌───┴───┐
High confidence    Low confidence
(≥ threshold)      (< threshold)
    ↓                   ↓
Auto-processed     Sent to human review
(bot handles it)   (Human-in-the-Loop)
        ↓
Summary Report + CSV Export
```

That's it. Simple, transparent, and explainable — exactly what production AI needs to be.

---

## ✨ What This Project Demonstrates

| Skill | How it shows up here |
|---|---|
| NLP / Text Classification | Zero-shot classification using BART model |
| Model selection | Chose `bart-large-mnli` — pre-trained on NLI tasks, works without fine-tuning |
| Confidence thresholds | Configurable slider — balances automation vs human oversight |
| Human-in-the-Loop | Low confidence queries never auto-processed — a production AI safety pattern |
| Data pipeline | CSV ingestion → classify → route → export |
| UI / Deployment | Streamlit app — model accessible without writing a single API |
| No data leakage | Model runs 100% locally — no customer data leaves the machine |

---

## 🚀 Run It Yourself

```bash
# 1. Clone
git clone [https://github.com/nerdygeek888/ai_classifier_agent.git]
cd ai_classifier_agent

# 2. Install
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

> **Note:** First run downloads the NLP model (~1.6GB). After that it loads from cache in seconds. No API key needed.

---

## 📊 Sample Results

Given these queries, the model predicts:

| Query | Predicted Category | Confidence |
|---|---|---|
| "I was charged twice for my order" | billing | 94% |
| "My app crashes on startup" | technical_support | 91% |
| "Where is my order ORD-5521?" | order_status | 89% |
| "I want a refund for this month" | refund | 87% |
| "What are your business hours?" | general_enquiry | 76% |

---

## 🗂️ Project Structure

```
├── app.py              → Streamlit UI (input, pipeline, report tabs)
├── processor.py        → NLP classification + routing logic
├── report_generator.py → HTML summary report
├── sample_queries.csv  → Test data to try the upload feature
└── requirements.txt    → All dependencies
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| `facebook/bart-large-mnli` | Zero-shot text classification |
| HuggingFace Transformers | Model loading and inference |
| Streamlit | Interactive web UI |
| Pandas | Data handling and CSV export |
| Python 3.11 | Core language |

---

## 💡 Key Technical Decisions

**Why zero-shot classification?**
Most classification projects require labelled training data (hundreds of examples per category). Zero-shot classification skips that entirely — the model uses its understanding of language to classify into any category you define. This makes it immediately usable for any business domain without data collection.

**Why a confidence threshold?**
AI models are not always right. Instead of blindly automating every decision, queries below a set confidence score are flagged for human review. This is a standard pattern in production ML systems — it keeps the automation rate high while protecting against wrong decisions.

**Why local model?**
Customer query data is sensitive. Running the model locally means no data is sent to external APIs. This matters in Finance, Healthcare, and Legal domains.

---

## 📌 Author

**Sushma Nadkarni** — AI Automation Engineer | Python | NLP | HuggingFace  
🔗 [LinkedIn](https://www.linkedin.com/in/sushmanadkarni-14087a117/) · [GitHub](https://github.com/nerdygeek888)
