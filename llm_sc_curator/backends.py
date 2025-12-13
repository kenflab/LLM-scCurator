
from abc import ABC, abstractmethod
import os
import json
import time

class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    Ensures a unified interface: input string -> output string.
    """
    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        pass

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x)
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

class GeminiBackend(BaseLLMBackend):
    def __init__(self, api_key, model_name='models/gemini-2.0-flash', temperature=0.0):
        """
        Gemini Backend.
        Args:
            temperature (float): Set to 0.0 for maximum determinism in JSON mode.
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install google-generativeai to use GeminiBackend.")


        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        try:
            # Gemini specific config for JSON
            if json_mode:
                generation_config = {
                    "temperature": self.temperature,
                    "response_mime_type": "application/json"
                }
            else:
                generation_config = {
                    "temperature": self.temperature
                }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            # Error fallback: Return a valid JSON error string if in JSON mode
            if json_mode:
                return json.dumps({"cell_type": "Error", "confidence": "Low", "reasoning": str(e)})
            return f"Gemini Error: {e}"

class OpenAIBackend(BaseLLMBackend):
    def __init__(self, api_key, model_name='gpt-4o', temperature=0.0, seed=42):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai package to use OpenAIBackend.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "seed": self.seed
            }

            # OpenAI specific config for JSON
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if json_mode:
                return json.dumps({"cell_type": "Error", "confidence": "Low", "reasoning": str(e)})
            return f"OpenAI Error: {e}"

class LocalLLMBackend(BaseLLMBackend):
    """Placeholder for future local models"""

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        return json.dumps({"cell_type": "Local_Pending", "confidence": "Low", "reasoning": "Not implemented"})
