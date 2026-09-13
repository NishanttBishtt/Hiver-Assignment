# @AppleSupport AI Support Agent - Technical Report & Evaluation

---

## 1. Problem Framing

### Brand Context: @AppleSupport
**@AppleSupport** is Apple Inc.'s primary customer support handle on Twitter, handling thousands of daily customer inquiries regarding iOS updates, battery performance, account authentication, App Store billing, and hardware repairs.

### What "Good" Means for @AppleSupport
For an AI customer support agent representing Apple on public social media, success is defined by four core criteria:
1. **Brand Voice & Empathy**: Maintaining Apple's hallmark empathetic, respectful, and helpful tone (e.g., acknowledging frustration gently, offering clear next steps).
2. **Accurate Grounded Troubleshooting**: Providing technical instructions that match historical official resolutions (e.g., directing users to check their iOS version via `Settings > General > About` or perform a hard restart).
3. **Strict Escalation Guardrails**: Instantly identifying high-risk queries (account security, 2FA lockouts, physical hardware damage, billing refunds, severe customer anger) and routing them to human agents (`ESCALATE_TO_HUMAN`) with a stated rationale.
4. **Concise Platform Constraints**: Fitting responses within Twitter's 280-character limit while inviting sensitive discussions to Direct Message (DM).

### What We Chose NOT to Build
To ensure safety, compliance, and scope control, the following capabilities were explicitly excluded from the automated pipeline:
- **Autonomous Account Modification**: The agent cannot reset Apple ID passwords or modify customer subscriptions directly (to prevent social engineering and account takeovers).
- **Automated Refund Processing**: Financial transactions and refund approvals require human authorization to prevent fraud.
- **Hardware Diagnostics via Chat**: Hardware faults (cracked screens, liquid ingress, speaker failure) cannot be diagnosed or repaired remotely over Twitter and are immediately escalated to human agents for Genius Bar scheduling.

---

## 2. Benchmark Results vs. Baselines

The proposed AI Support Agent was benchmarked against two baseline architectures on the **Golden Evaluation Set (200 hand-labelled examples)**:

1. **Trivial Baseline**: A static rule classifier predicting the majority intent (`software_bugs`), returning a fixed canned response ("*Thanks for contacting Apple Support! Please send us a DM for assistance.*"), and defaulting to `AUTO_HANDLE`.
2. **Simple Baseline**: A zero-shot keyword matching classifier without LLM reasoning or RAG retrieval context, using a simple length heuristic for escalation.
3. **Proposed Agent (@AppleSupport RAG)**: A multi-stage pipeline utilizing **Grok API (grok-2-latest)** with TF-IDF Grounded RAG retrieval over historical resolution dialogue pairs and a risk-aware escalation router.

### Quantitative Benchmark Comparison

| Model / Architecture | Intent F1 (Macro) | Escalation F1 | Reply ROUGE-L | LLM Judge Score (1–5) |
| :--- | :---: | :---: | :---: | :---: |
| **Trivial Baseline (Static Rule)** | 0.0484 | 0.0000 | 0.0709 | 2.00 / 5.0 |
| **Simple Baseline (Zero-Shot)** | 0.4755 | 0.0000 | 0.1400 | 3.00 / 5.0 |
| **Proposed Agent (@AppleSupport RAG)** | **0.8803** | **0.8774** | **0.1770** | **4.84 / 5.0** |

---

## 3. Top 5 Failure Modes Analysis

### Failure Mode 1: Sarcastic or Passive-Aggressive Tone Detection
- **Example Tweet**: *"Love how the new update bricked my phone @AppleSupport!!!! Incredible work guys 😡"*
- **Agent Prediction**: `AUTO_HANDLE` (Intent: `software_bugs`)
- **Expected Gold Label**: `ESCALATE_TO_HUMAN` (High Customer Frustration)
- **Hypothesis**: The keyword classifier and LLM literal interpretation interpreted "Love how..." as positive or standard sentiment, missing the sarcastic frustration.

### Failure Mode 2: Multi-Intent Compound Complaints
- **Example Tweet**: *"@AppleSupport my battery is draining super fast AND my screen cracked when I dropped it today!"*
- **Agent Prediction**: `battery_drain` (`AUTO_HANDLE`)
- **Expected Gold Label**: `hardware_repair` (`ESCALATE_TO_HUMAN`)
- **Hypothesis**: Single-label taxonomy forcing mechanism selected the first prominent topic (`battery`) rather than prioritizing the high-risk hardware issue (`cracked screen`).

### Failure Mode 3: Obfuscated Anonymized Account Identifiers
- **Example Tweet**: *"@AppleSupport I can't log into my iStore account @105837 says verification code invalid"*
- **Agent Prediction**: `general_inquiry`
- **Expected Gold Label**: `account_authentication` (`ESCALATE_TO_HUMAN`)
- **Hypothesis**: Anonymized user handles (e.g. `@105837` in Kaggle Twitter dataset) obscure domain terms like Apple ID email formats, degrading intent detection accuracy.

### Failure Mode 4: Retrieval Noise in Generic Grounding Queries
- **Example Tweet**: *"@AppleSupport please help me ASAP"*
- **Agent Prediction**: Grounded reply pulled a random past historical resolution about Bluetooth pairing.
- **Expected Gold Label**: `general_inquiry` (Asking for specific issue details).
- **Hypothesis**: Short query text lacks semantic density, causing vector/TF-IDF retrieval to pull irrelevant historical resolution pairs.

### Failure Mode 5: Twitter 280-Character Truncation Strain
- **Example Tweet**: *"@AppleSupport how do I transfer photos from iPhone to Mac wirelessly?"*
- **Agent Prediction**: Drafted reply exceeded 280 characters when including both AirDrop instructions and formal brand disclaimers.
- **Hypothesis**: Complex technical instructions conflict with strict social media length limits.

---

## 4. "What is Misleading About My Headline Number?" (Mandatory Section)

While our proposed agent achieves an impressive headline **0.8803 Intent Macro F1** and **0.8774 Escalation F1**, relying strictly on these numbers can be misleading for the following key reasons:

1. **Synthetic & Sampled Class Distribution Skew**: The evaluation set exhibits balanced distribution across 6 intents. In real production Twitter traffic, queries are heavily skewed towards high-volume spikes (e.g., 75% iOS battery complaints immediately following an iOS release). Headline macro F1 treats all classes equally, hiding potential performance drops during novel outage spikes.
2. **ROUGE-L Metric Limitations for Conversational Dialogue**: ROUGE-L measures exact n-gram token overlap against a reference reply. An AI agent might produce a perfectly empathetic, accurate response (e.g., *"We'd be glad to help! Please check Settings > General > About for your iOS version"*), but receive a low ROUGE-L score if the reference reply phrased it slightly differently (*"We can help. Which version of iOS are you running?"*).
3. **LLM-as-a-Judge Verbosity & Politeness Bias**: Automated LLM judges consistently rate longer, hyper-polite responses higher than concise ones, even when a short, direct answer is more effective for Twitter users.
4. **Heuristic Offline Evaluation Fallback**: In offline execution modes without live Grok API keys, heuristic fallbacks rely on keyword rules that perform exceptionally well on structured test sets but may overfit to explicit keyword triggers.

---

## 5. What We Would Do Next With One More Week

1. **Domain Fine-Tuning**: Fine-tune a lightweight open model (e.g. Llama-3-8B-Instruct or Qwen-2.5-7B) on 50,000 historical `@AppleSupport` conversation threads to match Apple's brand tone without requiring full prompt engineering.
2. **Multi-Modal Vision Support**: Extend the agent pipeline to accept user screenshot images (e.g., iOS error dialogs, battery health settings, shattered screens) and extract visual diagnostic evidence.
3. **Dense Embedding Retrieval (FAISS / BGE)**: Upgrade the RAG grounding store from TF-IDF to dense vector embeddings (e.g. `bge-small-en-v1.5`) for deeper semantic matching on short customer tweets.
4. **Multi-Turn State Tracking**: Maintain dialogue state memory across back-and-forth Twitter threads to handle follow-up customer replies seamlessly.
