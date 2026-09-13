import os
import re
import json
import logging
from typing import Dict, Any, Optional
from src.config import GROQ_API_KEY, GROQ_BASE_URL, DEFAULT_MODEL, INTENT_TAXONOMY

logger = logging.getLogger(__name__)

class GrokLLMClient:
    """
    LLM API Client for Grok (xAI API).
    Uses standard OpenAI SDK pointing to https://api.x.ai/v1 with fallback capabilities.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.base_url = base_url or GROQ_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
                logger.info(f"Initialized Grok client with model {self.model} at {self.base_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client for Grok API: {e}. Falling back to heuristic mode.")
        else:
            logger.info("No GROK_API_KEY provided. Running in deterministic heuristic mode.")

    def is_api_available(self) -> bool:
        return self.client is not None

    def generate_json(self, system_prompt: str, user_prompt: str, fallback_dict: Dict[str, Any], temperature: float = 0.0) -> Dict[str, Any]:
        """
        Sends request to Grok API expecting a structured JSON response.
        Falls back to fallback_dict if API is unavailable or call fails.
        """
        if not self.client:
            return fallback_dict

        try:
            # Avoid response_format restriction for models that produce reasoning tokens before JSON
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt + "\nRespond strictly with a valid raw JSON object. Do not include markdown codeblocks or thinking tags."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=256
            )
            content = response.choices[0].message.content or ""
            # Strip reasoning <think>...</think> tags if present
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            # Extract JSON substring
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Grok API call failed: {e}. Using deterministic fallback.")
            return fallback_dict

    def generate_text(self, system_prompt: str, user_prompt: str, fallback_text: str, temperature: float = 0.3) -> str:
        """
        Generates freeform text from Grok API.
        """
        if not self.client:
            return fallback_text

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Grok API text call failed: {e}. Using fallback text.")
            return fallback_text
