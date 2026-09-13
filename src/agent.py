import json
import re
from typing import Dict, Any, Optional, List
from src.config import INTENT_TAXONOMY, INTENT_NAMES, TARGET_BRAND_HANDLE, ESCALATION_REASONS
from src.llm_client import GrokLLMClient
from src.rag_store import GroundingRAGStore

class AppleSupportAgent:
    """
    Production AI Support Agent for @AppleSupport.
    Performs:
    1. Intent Classification
    2. RAG Grounded Reply Drafting
    3. Escalation Decision Routing (AUTO_HANDLE vs ESCALATE_TO_HUMAN + Reason)
    """
    def __init__(self, llm_client: Optional[GrokLLMClient] = None, rag_store: Optional[GroundingRAGStore] = None):
        self.llm_client = llm_client or GrokLLMClient()
        self.rag_store = rag_store or GroundingRAGStore()

    def process_message(self, customer_text: str) -> Dict[str, Any]:
        """
        Main pipeline handling incoming customer tweet.
        """
        grounding_context = self.rag_store.format_grounding_context(customer_text, top_k=2)
        retrieved_docs = self.rag_store.retrieve_similar(customer_text, top_k=2)

        fallback_output = self._heuristic_agent_process(customer_text, retrieved_docs)

        if not self.llm_client.is_api_available():
            return fallback_output

        system_prompt = f"""You are the official AI Support Agent for {TARGET_BRAND_HANDLE} (Apple Support).
Analyze an incoming customer tweet and output a JSON object:
1. "intent": One of {INTENT_NAMES}.
2. "intent_explanation": Reason for intent classification.
3. "draft_reply": A professional, empathetic tweet reply (under 280 chars) grounded in historical resolutions.
4. "escalation_decision": "AUTO_HANDLE" or "ESCALATE_TO_HUMAN".
5. "escalation_reason": Rationale for escalation or auto-handling.

Escalation Rules:
- ESCALATE_TO_HUMAN if: account lockout/2FA, unauthorized charges/refunds, hardware screen/mic physical damage, safety/overheating issues, or black screen of death.
- AUTO_HANDLE if: standard battery drain troubleshooting, iOS setting inquiries, software bugs with standard DM steps, feature inquiries.
"""

        user_prompt = f"""Customer Tweet: "{customer_text}"

Historical Resolved Context:
{grounding_context}

Output strictly valid JSON with keys: intent, intent_explanation, draft_reply, escalation_decision, escalation_reason.
"""

        llm_response = self.llm_client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_dict=fallback_output
        )

        return {
            "intent": llm_response.get("intent", fallback_output["intent"]),
            "intent_explanation": llm_response.get("intent_explanation", fallback_output["intent_explanation"]),
            "draft_reply": llm_response.get("draft_reply", fallback_output["draft_reply"]),
            "escalation_decision": llm_response.get("escalation_decision", fallback_output["escalation_decision"]),
            "escalation_reason": llm_response.get("escalation_reason", fallback_output["escalation_reason"]),
            "retrieved_docs": retrieved_docs
        }

    def _heuristic_agent_process(self, customer_text: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_lower = customer_text.lower()

        # Rule-based Intent Classification
        if any(w in text_lower for w in ["battery", "drain", "drains", "charger", "percentage", "hot"]):
            intent = "battery_drain"
        elif any(w in text_lower for w in ["freeze", "freezes", "crash", "crashes", "bug", "lag", "wifi", "wi-fi", "bluetooth", "keyboard", "black screen"]):
            intent = "software_bugs"
        elif any(w in text_lower for w in ["2fa", "apple id", "passcode", "disabled", "login", "auth", "verification code", "locked", "istore"]):
            intent = "account_authentication"
        elif any(w in text_lower for w in ["shattered", "drop", "dropped", "glass", "microphone", "mic", "speaker", "water", "pool", "repair", "screen"]):
            intent = "hardware_repair"
        elif any(w in text_lower for w in ["refund", "charged", "billing", "subscription", "invoice", "coins", "purchase", "icloud"]):
            intent = "billing_subscriptions"
        else:
            intent = "general_inquiry"

        intent_exp = f"Classified as {intent} based on domain keywords in query."

        # Rule-based Escalation Routing
        needs_esc = False
        reason = "Standard software/FAQ query suitable for automated reply."

        if intent == "account_authentication":
            if any(w in text_lower for w in ["2fa", "locked", "someone", "unauthorized"]):
                needs_esc = True
                reason = ESCALATION_REASONS["SENSITIVE_ACCOUNT_INFO"]
        elif intent == "hardware_repair":
            if any(w in text_lower for w in ["shattered", "drop", "dropped", "mic", "speaker", "water", "unresponsive"]):
                needs_esc = True
                reason = ESCALATION_REASONS["HARDWARE_REPAIR"]
        elif intent == "billing_subscriptions":
            if any(w in text_lower for w in ["refund", "charged", "unauthorized", "disputed"]):
                needs_esc = True
                reason = ESCALATION_REASONS["FINANCIAL_DISPUTE"]
        elif intent == "battery_drain" and "hot" in text_lower:
            needs_esc = True
            reason = "Hardware heating safety risk requires human agent inspection."
        elif intent == "software_bugs" and "black screen" in text_lower:
            needs_esc = True
            reason = ESCALATION_REASONS["COMPLEX_UNRESOLVED_BUG"]

        esc_decision = "ESCALATE_TO_HUMAN" if needs_esc else "AUTO_HANDLE"

        # RAG Grounded Reply Generation
        if intent == "battery_drain":
            reply = "We can help with your battery performance! Which version of iOS are you using? Reply in DM so we can check Settings > General > About."
        elif intent == "software_bugs":
            reply = "We'd love to help! Please DM us your device model and current iOS version, plus any troubleshooting steps you've tried."
        elif intent == "account_authentication":
            reply = "Account security is very important. Please reach out to us via DM so a support agent can assist with your Apple ID status securely."
        elif intent == "hardware_repair":
            reply = "We're sorry to hear about your hardware trouble. Please DM us your device details so we can help set up a Genius Bar appointment."
        elif intent == "billing_subscriptions":
            reply = "We'd be glad to look into this billing concern for you. Please send us a DM with your invoice details so we can review."
        else:
            reply = "Thanks for reaching out! We'd love to help answer your question. Send us a DM so we can look into this together."

        return {
            "intent": intent,
            "intent_explanation": intent_exp,
            "draft_reply": reply,
            "escalation_decision": esc_decision,
            "escalation_reason": reason,
            "retrieved_docs": retrieved_docs
        }
