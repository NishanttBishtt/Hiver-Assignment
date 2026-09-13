from typing import Dict, Any, List

class TrivialBaseline:
    """
    Trivial Baseline:
    - Intent: Always predicts majority intent ('software_bugs')
    - Reply: Static canned boilerplate response
    - Escalation: Always predicts 'AUTO_HANDLE'
    """
    def process_message(self, customer_text: str) -> Dict[str, Any]:
        return {
            "intent": "software_bugs",
            "intent_explanation": "Trivial baseline always predicts majority class: software_bugs.",
            "draft_reply": "Thanks for contacting Apple Support! Please send us a DM for assistance.",
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "Trivial baseline default auto-handle decision.",
            "retrieved_docs": []
        }

class SimpleBaseline:
    """
    Simple Baseline:
    - Intent: Naive keyword matching without LLM context
    - Reply: Generic prompt reply without RAG grounding context
    - Escalation: Simple length heuristic (if text > 100 chars, escalate)
    """
    def process_message(self, customer_text: str) -> Dict[str, Any]:
        text_lower = customer_text.lower()
        
        # Naive intent keyword match
        if "battery" in text_lower:
            intent = "battery_drain"
        elif "freeze" in text_lower or "crash" in text_lower or "ios" in text_lower:
            intent = "software_bugs"
        elif "password" in text_lower or "id" in text_lower:
            intent = "account_authentication"
        elif "repair" in text_lower or "screen" in text_lower:
            intent = "hardware_repair"
        elif "refund" in text_lower or "billing" in text_lower:
            intent = "billing_subscriptions"
        else:
            intent = "general_inquiry"

        # Naive escalation heuristic: message length > 120 chars -> ESCALATE
        escalation = "ESCALATE_TO_HUMAN" if len(customer_text) > 120 else "AUTO_HANDLE"
        reason = "Message length exceeds threshold" if escalation == "ESCALATE_TO_HUMAN" else "Short message auto-handled"

        draft = f"We can help with your {intent.replace('_', ' ')} query! Send us a DM."

        return {
            "intent": intent,
            "intent_explanation": "Simple keyword heuristic classification.",
            "draft_reply": draft,
            "escalation_decision": escalation,
            "escalation_reason": reason,
            "retrieved_docs": []
        }
