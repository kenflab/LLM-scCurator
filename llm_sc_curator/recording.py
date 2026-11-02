"""Opt-in records of prompts and raw text at the backend interface."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .backends import BaseLLMBackend


class ResponseAuditError(RuntimeError):
    """The requested response record could not be written reliably."""


class RecordingBackend(BaseLLMBackend):
    """Record each backend call without changing its prompt or return value.

    The JSONL contains a ``call_start`` followed by ``call_response`` or
    ``call_error`` with a shared ``call_id``. Start records are flushed before
    inference. A start without a terminal event identifies an interrupted call.
    ``raw_response_text`` is the unmodified string returned by ``generate``;
    it is not the provider's complete SDK/HTTP response. Retries performed inside
    the wrapped backend are not visible here. Repeated calls from ``annotate``
    (including malformed-JSON retries) each receive their own record.

    Use one log per process. The parent directory must already exist. Logs may
    contain the biological context in the prompt; choose an appropriate path.
    Only the named model/settings below are inspected, never credentials or a
    backend object's ``__dict__``. Optional metadata keys identify a run without
    silently copying arbitrary configuration or secrets.
    """

    METADATA_KEYS = frozenset({
        "run_id", "dataset_id", "cluster_id", "condition", "replicate", "code_ref"
    })
    SETTING_KEYS = ("model_name", "temperature", "seed", "timeout")

    def __init__(self, backend, audit_path, metadata=None):
        if not isinstance(backend, BaseLLMBackend):
            raise TypeError("backend must inherit from BaseLLMBackend")
        self.backend = backend
        self.audit_path = Path(audit_path)
        self.metadata = dict(metadata or {})

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _write(handle, record):
        try:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        except (OSError, TypeError, ValueError) as exc:
            raise ResponseAuditError("Could not write the requested response audit log") from exc

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        # Validate and serialize before opening the file or making a model call.
        unknown = set(self.metadata) - self.METADATA_KEYS
        if any(value is not None and not isinstance(value, (str, int, float, bool))
               for value in self.metadata.values()):
            raise ResponseAuditError("Audit metadata values must be scalar run/cluster identifiers")
        if unknown:
            raise ResponseAuditError("Unsupported audit metadata key; use run/cluster identifiers only")
        settings = {}
        for key in self.SETTING_KEYS:
            value = getattr(self.backend, key, None)
            if value is None or isinstance(value, (str, int, float, bool)):
                settings[key] = value
        call_id = uuid4().hex
        start = {
            "schema_version": 1,
            "event": "call_start",
            "call_id": call_id,
            "timestamp_utc": self._utc_now(),
            "backend_class": type(self.backend).__name__,
            "configured_settings": settings,
            "metadata": dict(self.metadata),
            "json_mode": bool(json_mode),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "capture_scope": "backend_returned_text",
        }
        try:
            # Prevent non-JSON metadata from triggering an inference without a record.
            json.dumps(start, ensure_ascii=False, allow_nan=False)
            handle = self.audit_path.open("a", encoding="utf-8", newline="\n")
        except (OSError, TypeError, ValueError) as exc:
            raise ResponseAuditError("Could not initialize the requested response audit log") from exc

        try:
            self._write(handle, start)
            try:
                response = self.backend.generate(prompt, json_mode=json_mode)
            except Exception as exc:
                self._write(handle, {
                    "schema_version": 1, "event": "call_error", "call_id": call_id,
                    "timestamp_utc": self._utc_now(),
                    # Do not copy exception text: provider errors can include credentials.
                    "exception_type": type(exc).__name__,
                })
                raise
            self._write(handle, {
                "schema_version": 1, "event": "call_response", "call_id": call_id,
                "timestamp_utc": self._utc_now(), "raw_response_text": response,
            })
            return response

        finally:
            try:
                handle.close()
            except OSError as exc:
                raise ResponseAuditError("Could not close the requested response audit log") from exc
