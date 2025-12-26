from abc import ABC, abstractmethod
import json
import time
import logging

class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    This interface enforces a backend-agnostic contract:

    - Input: a prompt string (optionally requesting strict JSON output).
    - Output: a single string (either free-form text or a JSON object encoded as text).

    Implementations may wrap cloud APIs (e.g., Gemini/OpenAI) or local models, but must
    expose a stable `generate()` method to keep downstream annotation code unchanged.
    """

    @abstractmethod
    def generate(self, prompt: str, json_mode: bool = False) -> str:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the backend.
        json_mode : bool, default=False
            If True, the backend should attempt to return a valid JSON object encoded
            as a string (not fenced code blocks).

        Returns
        -------
        str
            Model output as a string. In JSON mode, this should be a JSON object encoded
            as a string (e.g., '{"cell_type": "...", "confidence": "...", "reasoning": "..."}').
        """
        pass

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Decorator implementing exponential backoff retries for LLM calls.

    This decorator is designed for two practical failure modes:

    1) Exceptions raised by a backend call (network timeouts, transient 5xx, etc.).
    2) "Soft errors" returned as strings (e.g., 'Gemini Error: ...') or JSON objects
       with `cell_type == "Error"` when `json_mode=True` (useful for backends that
       catch exceptions and return an error payload instead of raising).

    The retry policy aims to improve batch robustness while limiting unnecessary token
    usage by avoiding retries for non-recoverable conditions (e.g., authentication or
    invalid requests).

    Parameters
    ----------
    retries : int, default=3
        Number of retries after the initial attempt (total attempts = retries + 1).
    backoff_in_seconds : int or float, default=1
        Base delay (seconds) for exponential backoff. The sleep schedule is:
        backoff_in_seconds * 2**attempt, with a small random jitter.

    Returns
    -------
    callable
        A decorator that wraps a function, retrying on retryable exceptions or
        retryable error-return payloads.

    Notes
    -----
    - Logging is emitted at DEBUG level for retry attempts.
    - This decorator does not alter successful outputs; it only changes failure-path
      behavior by adding retry attempts.
    """
    import random
    from functools import wraps

    NON_RETRYABLE_PATTERNS = [
        "401", "403", "invalid_api_key", "unauthorized", "forbidden",
        "400", "bad request", "invalid_request",
        "model not found",
        # NOTE: removed broad "not found" to avoid false non-retryable matches
        "insufficient_quota", "quota", "billing"
    ]

    RETRYABLE_HINTS = [
        "429", "rate limit",
        "500", "502", "503", "504",
        "timeout", "timed out", "connection", "temporarily", "unavailable"
    ]

    def _should_retry_error_text(err_text: str) -> bool:
        t = (err_text or "").lower()
        if any(p in t for p in NON_RETRYABLE_PATTERNS):
            return False
        if any(h in t for h in RETRYABLE_HINTS):
            return True
        # Conservative by default to avoid token waste
        return False

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            json_mode = bool(kwargs.get("json_mode", False))
            parsefail_used = False  # allow at most one retry for JSON parse failures

            while True:
                try:
                    result = func(*args, **kwargs)

                    # --- Returned-error detection (needed for GeminiBackend) ---
                    retryable = False
                    if result is None:
                        retryable = True
                    elif isinstance(result, str):
                        s = result.strip()

                        if s.startswith("Gemini Error:") or s.startswith("OpenAI Error:"):
                            retryable = _should_retry_error_text(s)

                        elif json_mode and s.startswith("{") and s.endswith("}"):
                            try:
                                obj = json.loads(s)
                                if isinstance(obj, dict) and obj.get("cell_type") == "Error":
                                    retryable = _should_retry_error_text(obj.get("reasoning", ""))
                            except Exception:
                                # JSON mode requested but got unparsable JSON:
                                # allow *one* retry (often transient truncation)
                                retryable = not parsefail_used
                                parsefail_used = True

                    if retryable:
                        if x == retries:
                            return result
                        sleep = backoff_in_seconds * (2 ** x)
                        sleep *= (1.0 + random.uniform(-0.1, 0.1))
                        logging.debug("LLM retry (returned error). attempt=%d sleep=%.2fs", x + 1, sleep)
                        time.sleep(max(0.0, sleep))
                        x += 1
                        continue

                    return result

                except Exception as e:
                    msg = str(e)
                    if not _should_retry_error_text(msg):
                        raise
                    if x == retries:
                        raise
                    sleep = backoff_in_seconds * (2 ** x)
                    sleep *= (1.0 + random.uniform(-0.1, 0.1))
                    logging.debug("LLM retry (exception). attempt=%d sleep=%.2fs err=%s", x + 1, sleep, msg)
                    time.sleep(max(0.0, sleep))
                    x += 1

        return wrapper
    return decorator


class GeminiBackend(BaseLLMBackend):
    """
    Google Gemini backend for LLM-scCurator.

    This backend wraps `google-generativeai` and supports an optional JSON mode by
    setting the response MIME type to `application/json`.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    model_name : str, default="models/gemini-2.0-flash"
        Gemini model identifier.
    temperature : float, default=0.0
        Sampling temperature. Use 0.0 to maximize determinism (recommended for JSON mode).

    Raises
    ------
    ImportError
        If `google-generativeai` is not installed.

    Notes
    -----
    On exceptions, this backend returns a soft error payload (string or JSON string)
    rather than raising, to keep downstream annotation pipelines robust.
    """

    def __init__(self, api_key, model_name='models/gemini-2.0-flash', temperature=0.0):
        """
        Initialize the Gemini backend.

        Parameters
        ----------
        api_key : str
            Gemini API key.
        model_name : str, default="models/gemini-2.0-flash"
            Gemini model identifier.
        temperature : float, default=0.0
            Sampling temperature (0.0 recommended for deterministic JSON responses).
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
        """
        Generate a completion from Gemini.

        Parameters
        ----------
        prompt : str
            Prompt text to send to Gemini.
        json_mode : bool, default=False
            If True, request JSON output via `response_mime_type="application/json"`.

        Returns
        -------
        str
            Model output string. If an exception occurs:
            - In JSON mode: returns a JSON string with keys {cell_type, confidence, reasoning}.
            - Otherwise: returns a human-readable error string prefixed with "Gemini Error:".
        """
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
    """
    OpenAI Chat Completions backend for LLM-scCurator.

    This backend uses the official `openai` Python package and supports JSON mode via
    `response_format={"type": "json_object"}`.

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    model_name : str, default="gpt-4o"
        Model identifier.
    temperature : float, default=0.0
        Sampling temperature (0.0 recommended for structured outputs).
    seed : int, default=42
        Seed forwarded to the API when supported by the selected model. If the request
        fails with seed enabled, the backend retries once without the seed.

    Raises
    ------
    ImportError
        If the `openai` package is not installed.

    Notes
    -----
    In contrast to GeminiBackend, OpenAIBackend attempts a seed-based call first and
    falls back to a seed-free call if needed. On errors it returns a soft error payload
    (string or JSON string), which enables batch pipelines to continue.
    """

    def __init__(self, api_key, model_name='gpt-4o', temperature=0.0, seed=42):
        """
        Initialize the OpenAI backend.

        Parameters
        ----------
        api_key : str
            OpenAI API key.
        model_name : str, default="gpt-4o"
            Model identifier.
        temperature : float, default=0.0
            Sampling temperature (0.0 recommended for structured outputs).
        seed : int, default=42
            Seed forwarded to the API when supported.
        """
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

        """
        Generate a completion from OpenAI Chat Completions.

        Parameters
        ----------
        prompt : str
            Prompt text to send to the OpenAI API.
        json_mode : bool, default=False
            If True, request a JSON object response format.

        Returns
        -------
        str
            Model output string. If an exception occurs:
            - In JSON mode: returns a JSON string with keys {cell_type, confidence, reasoning}.
            - Otherwise: returns a human-readable error string prefixed with "OpenAI Error:".

        Notes
        -----
        The backend first attempts a request with `seed` enabled. If that fails (e.g.,
        due to model/endpoint incompatibility), it retries once without `seed` before
        returning a soft error payload.
        """
        def _call(with_seed: bool):
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if with_seed:
                kwargs["seed"] = self.seed
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        first_exc = None
        try:
            return _call(with_seed=True)
        except Exception as e:
            first_exc = e
            try:
                return _call(with_seed=False)
            except Exception as e2:
                # preserve the most informative error message
                err = e2 if e2 is not None else first_exc
                if json_mode:
                    return json.dumps({"cell_type": "Error", "confidence": "Low", "reasoning": str(err)})
                return f"OpenAI Error: {err}"

class LocalLLMBackend(BaseLLMBackend):
    """
    Placeholder backend for future local model integrations.

    This backend currently returns a fixed JSON payload indicating that local inference
    is not implemented. It exists to document the intended extension point and to keep
    the public API stable.

    Notes
    -----
    Use this class as a template when integrating a local LLM runner (e.g., llama.cpp,
    vLLM, or an on-premise service) into the `BaseLLMBackend` interface.
    """

    @retry_with_backoff(retries=3)
    def generate(self, prompt: str, json_mode: bool = False) -> str:

        """
        Return a placeholder response (not implemented).

        Parameters
        ----------
        prompt : str
            Unused placeholder parameter.
        json_mode : bool, default=False
            If True, returns a JSON object string.

        Returns
        -------
        str
            A JSON string with `cell_type="Local_Pending"`, indicating that the local
            backend is not yet implemented.
        """
        return json.dumps({"cell_type": "Local_Pending", "confidence": "Low", "reasoning": "Not implemented"})
