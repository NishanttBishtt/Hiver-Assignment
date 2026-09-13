import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Target Brand Selection
TARGET_BRAND = "AppleSupport"
TARGET_BRAND_HANDLE = "@AppleSupport"

# LLM API Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("GROK_MODEL") or "groq/compound"

# Intent Taxonomy (Derived from real @AppleSupport customer interactions)
INTENT_TAXONOMY = {
    "battery_drain": "Battery life draining rapidly, overheating, or battery health degradation post-update.",
    "software_bugs": "iOS/app crashes, freezing screen, Wi-Fi/Bluetooth disconnection, keyboard or notification glitches.",
    "account_authentication": "Apple ID locked/disabled, 2FA code issue, password reset, App Store verification.",
    "hardware_repair": "Physical screen damage, defective speaker/mic, water damage, hardware button failure.",
    "billing_subscriptions": "In-app purchase refund request, unauthorized charge, Apple Music/iCloud billing query.",
    "general_inquiry": "How-to questions, update availability, feature settings, device compatibility."
}

INTENT_NAMES = list(INTENT_TAXONOMY.keys())

# Escalation Policies
ESCALATION_REASONS = {
    "SENSITIVE_ACCOUNT_INFO": "Customer requires account-level authentication or credential handling.",
    "HARDWARE_REPAIR": "Requires physical inspection, repair setup, or Genius Bar appointment.",
    "HIGH_CUSTOMER_FRUSTRATION": "Customer exhibits extreme frustration, anger, or explicit human agent request.",
    "FINANCIAL_DISPUTE": "Involves unauthorized charges or refund processing requiring secure verification.",
    "COMPLEX_UNRESOLVED_BUG": "Persistent technical issue unresolved after standard troubleshooting."
}

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
GOLDEN_SET_PATH = os.path.join(DATA_DIR, "golden_eval_set.json")
GOLDEN_SET_NOTE_PATH = os.path.join(DATA_DIR, "GOLDEN_SET_NOTE.md")
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REPORT.md")
DECISION_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "DECISION_LOG.md")
