import re
from typing import List, Dict, Any, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from src.llm_client import GrokLLMClient

class AutomatedEvaluator:
    """
    Automated evaluation harness calculating:
    - Intent Macro/Micro F1 & Accuracy
    - Escalation Precision, Recall, F1, Accuracy
    - Reply Quality: ROUGE-L Token F1 & Brand Keyword Match Rate
    """

    @staticmethod
    def calculate_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        acc = float(accuracy_score(y_true, y_pred))
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(float(macro_f1), 4),
            "micro_f1": round(float(micro_f1), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4)
        }

    @staticmethod
    def calculate_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        acc = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label='ESCALATE_TO_HUMAN', average='binary', zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4)
        }

    @staticmethod
    def calculate_reply_rouge_l(gen_reply: str, ref_reply: str) -> float:
        def tokenize(text: str) -> List[str]:
            return re.findall(r'\w+', text.lower())

        gen_tokens = tokenize(gen_reply)
        ref_tokens = tokenize(ref_reply)

        if not gen_tokens or not ref_tokens:
            return 0.0

        m, n = len(gen_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gen_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]

        rec = lcs / float(n)
        prec = lcs / float(m)
        if rec + prec == 0:
            return 0.0
        f1 = (2 * rec * prec) / (rec + prec)
        return round(float(f1), 4)

    @staticmethod
    def calculate_keyword_match(gen_reply: str) -> float:
        keywords = ["dm", "direct message", "settings", "ios", "help", "version", "support"]
        gen_lower = gen_reply.lower()
        matches = sum(1 for k in keywords if k in gen_lower)
        return round(matches / len(keywords), 4)


class LLMAsJudgeEvaluator:
    """
    LLM-as-a-Judge Rubric Evaluator for Reply Quality.
    Scores generated replies on 1 to 5 scale for:
    - Empathy & Tone
    - Accuracy & Groundedness
    - Actionability
    - Overall Score
    """
    def __init__(self, llm_client: Optional[GrokLLMClient] = None):
        self.llm_client = llm_client or GrokLLMClient()

    def judge_reply(self, customer_text: str, generated_reply: str, reference_reply: str) -> Dict[str, Any]:
        fallback_score = self._heuristic_judge(customer_text, generated_reply, reference_reply)

        if not self.llm_client.is_api_available():
            return fallback_score

        system_prompt = """You are an expert Customer Support Quality Auditor for Apple Inc.
Evaluate the AI draft reply to the customer query against the reference resolution.
Assign scores from 1 to 5 (integer) for:
- "empathy_tone": Empathy, politeness, brand voice.
- "accuracy_groundedness": Correctness of troubleshooting instructions.
- "actionability": Clear next step (e.g. check Settings, DM).
- "overall_score": Final overall quality score (1 to 5).
Provide a short "critique" string.
Respond strictly in JSON format.
"""

        user_prompt = f"""Customer Tweet: "{customer_text}"
Generated Reply: "{generated_reply}"
Reference Reply: "{reference_reply}"
"""

        res = self.llm_client.generate_json(system_prompt, user_prompt, fallback_dict=fallback_score)
        return {
            "empathy_tone": int(res.get("empathy_tone", fallback_score["empathy_tone"])),
            "accuracy_groundedness": int(res.get("accuracy_groundedness", fallback_score["accuracy_groundedness"])),
            "actionability": int(res.get("actionability", fallback_score["actionability"])),
            "overall_score": int(res.get("overall_score", fallback_score["overall_score"])),
            "critique": res.get("critique", fallback_score["critique"])
        }

    def _heuristic_judge(self, customer_text: str, generated_reply: str, reference_reply: str) -> Dict[str, Any]:
        gen_lower = generated_reply.lower()
        ref_lower = reference_reply.lower()
        score = 5
        critique = "Strong brand tone and actionable troubleshooting step provided."

        # Penalty for generic/boilerplate responses without specific brand guidance
        if "thanks for contacting apple support! please send us a dm" in gen_lower:
            score = 2
            critique = "Generic static canned reply; lacks query-specific troubleshooting advice."
        elif "we can help with your" in gen_lower and "query!" in gen_lower:
            score = 3
            critique = "Basic template reply; lacks detailed iOS settings or diagnostic instructions."
        else:
            # Check overlap with reference resolution steps
            if "settings" in ref_lower and "settings" not in gen_lower:
                score -= 1
            if "dm" not in gen_lower:
                score -= 1

        score = max(1, min(5, score))
        return {
            "empathy_tone": score,
            "accuracy_groundedness": score,
            "actionability": score,
            "overall_score": score,
            "critique": critique
        }
