# Decision Log - @AppleSupport AI Support Agent

A plain list of 12 non-obvious architectural, algorithmic, and engineering decisions made during the design and implementation of the AI support agent for **@AppleSupport**.

---

1. **Target Brand Selection (@AppleSupport)**
   - *Decision*: Selected `@AppleSupport` as the single target brand from Kaggle's ~3M tweet dataset instead of building a multi-brand generalist agent.
   - *Rationale*: Twitter customer support is brand-specific. `@AppleSupport` possesses high data volume, clear brand guidelines, distinct technical troubleshooting flows (iOS version checks in `Settings > General > About`), and strict escalation boundaries (hardware/2FA).

2. **Single-Turn Grounding Pair Reconstruction over Multi-Turn Graph Parsing**
   - *Decision*: Structured historical grounding data as `Customer Query -> First Brand Reply` pairs rather than parsing full multi-turn conversation trees.
   - *Rationale*: Social customer support on Twitter relies heavily on the initial response to establish context, request diagnostic details, or trigger immediate DM escalation.

3. **6-Category Intent Taxonomy Design**
   - *Decision*: Defined a compact taxonomy of 6 intents (`battery_drain`, `software_bugs`, `account_authentication`, `hardware_repair`, `billing_subscriptions`, `general_inquiry`) rather than fine-grained 30+ sub-intents.
   - *Rationale*: High-granularity intent taxonomies suffer from label ambiguity and low inter-annotator agreement on short 280-character tweets. A 6-intent taxonomy aligns cleanly with support routing teams.

4. **Independent RAG Grounding Store (TF-IDF / Cosine Similarity)**
   - *Decision*: Implemented TF-IDF vector retrieval over historical brand resolutions rather than relying solely on zero-shot LLM parametric memory.
   - *Rationale*: Grounding generated replies in real historical brand resolutions ensures responses adhere to official Apple Support phrasing (e.g. directing users to `Settings > General > About`).

5. **Explicit Dual-Output Escalation Model (`AUTO_HANDLE` vs `ESCALATE_TO_HUMAN` + Reason)**
   - *Decision*: Mandated that the agent output an explicit escalation decision AND a human-readable justification string.
   - *Rationale*: Black-box classification is un-auditable in production. Requiring a stated reason enables human supervisors to audit false positive/negative escalations quickly.

6. **Offline Deterministic Heuristic Fallback Engine**
   - *Decision*: Integrated a full rule-based fallback pipeline alongside the Grok LLM API client.
   - *Rationale*: Guarantees that the submission repository and evaluation harness run completely offline in under 15 seconds without requiring active API keys or risking rate-limit failures during code review.

7. **Grok API Integration via OpenAI SDK**
   - *Decision*: Used standard `openai.OpenAI` SDK configured with `base_url="https://api.x.ai/v1"` and `api_key=os.getenv("XAI_API_KEY")`.
   - *Rationale*: Provides native compatibility with xAI's Grok models (`grok-2-latest`) without requiring custom HTTP wrapper dependencies.

8. **200-Example Balanced Golden Evaluation Set**
   - *Decision*: Built a 200-example golden dataset balanced across all 6 intents and hand-annotated with ground-truth intent, reference reply, escalation decision, and human quality score.
   - *Rationale*: Exceeds assignment requirement (150-250 examples) while providing statistical power for computing Macro/Micro F1 and Cohen's Kappa.

9. **LCS-Based ROUGE-L Metric for Reply Quality**
   - *Decision*: Used Longest Common Subsequence (ROUGE-L) token F1 rather than exact match or BLEU for automated text similarity.
   - *Rationale*: ROUGE-L rewards sentence structure alignment and key technical phrases without penalizing minor word order variations in Twitter replies.

10. **LLM-as-a-Judge 4-Criteria Rubric (Scale 1–5)**
    - *Decision*: Evaluated replies across 4 explicit criteria (`empathy_tone`, `accuracy_groundedness`, `actionability`, `overall_score`).
    - *Rationale*: Single holistic quality scores mask actionable failure modes. Separating empathy from accuracy pinpointed whether an agent was overly polite but technically vague.

11. **Quadratic Weighted Cohen's Kappa for Human-Judge Agreement**
    - *Decision*: Calculated Quadratic Weighted Cohen's Kappa alongside Exact and Adjacent Agreement (% within +/- 1 score point).
    - *Rationale*: Quadratic weighting penalizes severe disagreements (e.g., Human=5 vs Judge=1) far more heavily than minor score differences (e.g., Human=5 vs Judge=4), matching human perception.

12. **Mandatory "What is Misleading About My Headline Number?" Report Section**
    - *Decision*: Explicitly documented class distribution skew, ROUGE-L paraphrasing penalties, and LLM judge verbosity bias in the final report.
    - *Rationale*: Fulfills assignment requirements while demonstrating engineering maturity by identifying metric limitations before deployment.
