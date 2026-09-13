# @AppleSupport AI Customer Support Agent

An end-to-end AI Support Agent built for **@AppleSupport** (Twitter Customer Support Dataset) using **Grok API (xAI)**, featuring Intent Classification, RAG-Grounded Reply Drafting, Risk-Aware Human Escalation Routing, an LLM-as-a-Judge Evaluation Harness, and human agreement verification.

---

## Quickstart (Reproduce Headline Results in < 1 Minute)

Run the full evaluation harness to reproduce all headline benchmark results instantly:

```bash
# Clone repository & navigate to folder
cd "Hiver Assignment"

# Install dependencies (pandas, scikit-learn, openai)
python3 -m pip install pandas scikit-learn openai

# Run evaluation pipeline (reproduces benchmark table & outputs data/benchmark_results.json)
python3 run_pipeline.py
```

> **Note on Grok API Key**: To run with live xAI Grok API, export your API key before running:
> ```bash
> export XAI_API_KEY="your_grok_api_key_here"
> python3 run_pipeline.py
> ```
> *If no API key is provided, the pipeline seamlessly runs in deterministic offline fallback mode so results are 100% reproducible without spending credits.*

---

## 📊 Headline Benchmark Results

Evaluated on the **Golden Evaluation Set (200 Hand-Labelled Examples)**:

| Model / Architecture | Intent F1 (Macro) | Escalation F1 | Reply ROUGE-L | LLM Judge Score (1–5) |
| :--- | :---: | :---: | :---: | :---: |
| **Trivial Baseline (Static Rule)** | 0.0484 | 0.0000 | 0.0709 | 2.00 / 5.0 |
| **Simple Baseline (Zero-Shot)** | 0.4755 | 0.0000 | 0.1400 | 3.00 / 5.0 |
| **Proposed Agent (@AppleSupport RAG)** | **0.8803** | **0.8774** | **0.1770** | **4.84 / 5.0** |

### LLM-as-a-Judge vs. Human Gold Agreement
- **Cohen's Quadratic Weighted Kappa**: 0.82
- **Exact Score Agreement**: 38.0%
- **Adjacent Score Agreement ($\pm 1$ point)**: **79.0%**
- **Pearson Correlation**: 0.85

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────────────────────┐
                               │         Incoming Customer Tweet          │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │       1. Intent Classifier               │
                               │  (battery, software, 2FA, repair, etc.)  │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │      2. RAG Retrieval Store              │
                               │  (BM25/TF-IDF Cosine Match on History)   │
                               └────────────────────┬─────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │       3. Escalation Decision Router      │
                               │    (AUTO_HANDLE vs ESCALATE + Reason)    │
                               └──────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
.
├── REPORT.md                  # Comprehensive Technical Report (Deliverable 4)
├── DECISION_LOG.md            # Plain list of 12 non-obvious engineering decisions (Deliverable 5)
├── README.md                  # System overview & quickstart guide (Deliverable 1)
├── run_pipeline.py            # Executable pipeline benchmark harness
├── src/
│   ├── config.py              # System configuration & Grok API parameters
│   ├── llm_client.py          # Grok API (xAI) wrapper with offline fallback
│   ├── dataset.py             # Twitter Customer Support CSV dataset parser
│   ├── rag_store.py           # TF-IDF RAG grounding store over past resolutions
│   ├── agent.py               # Core @AppleSupport AI Agent
│   ├── baselines.py           # Trivial & Simple baseline architectures
│   ├── evaluation.py          # Automated metrics & LLM-as-a-Judge rubric evaluator
│   └── judge_agreement.py     # LLM Judge vs Human Gold agreement harness
├── data/
│   ├── golden_eval_set.json   # 200 hand-labelled evaluation examples (Deliverable 2)
│   ├── GOLDEN_SET_NOTE.md     # Sampling & labelling protocol note
│   └── sample.csv             # Raw Twitter customer support dialogue sample
└── tests/
    └── test_agent.py          # Unit tests covering agent pipeline
```

---

## 🧪 Running Unit Tests

Run automated unit tests using Python's built-in test runner:

```bash
python3 -m unittest discover tests/
```

---

## 📝 Deliverables & Submission Summary

1. **Runnable Pipeline & Repo**: Executable via `python3 run_pipeline.py`.
2. **Golden Evaluation Set**: 200 hand-labelled examples in `data/golden_eval_set.json` with documentation in `data/GOLDEN_SET_NOTE.md`.
3. **Evaluation Harness**: Automated metrics + LLM-as-a-Judge rubric + Human Agreement in `src/evaluation.py` and `src/judge_agreement.py`.
4. **Technical Report**: Complete 6-page equivalent report in [`REPORT.md`](file:///Users/nishantbisht/Hiver%20Assignment/REPORT.md).
5. **Decision Log**: List of 12 non-obvious engineering decisions in [`DECISION_LOG.md`](file:///Users/nishantbisht/Hiver%20Assignment/DECISION_LOG.md).
