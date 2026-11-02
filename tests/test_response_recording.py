"""Offline tests: no provider SDK, credentials, network, or model calls required."""
import importlib.util
import json
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# Import the real annotation code in an isolated namespace. Scanpy is unused by
# annotate(); substitute only that import if the optional test environment lacks it.
PKG = "_llmsc_recording_tests"
package = types.ModuleType(PKG)
package.__path__ = [str(ROOT / "llm_sc_curator")]
sys.modules[PKG] = package

def load(name):
    full = PKG + "." + name
    spec = importlib.util.spec_from_file_location(full, ROOT / "llm_sc_curator" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module

backends = load("backends")
recording = load("recording")
try:
    import scanpy  # noqa: F401
    core = load("core")
except ModuleNotFoundError as exc:
    if exc.name != "scanpy":
        raise
    sys.modules["scanpy"] = types.ModuleType("scanpy")
    try:
        core = load("core")
    finally:
        sys.modules.pop("scanpy", None)

RAW = '```json\n{"cell_type":"T cell", "confidence":"High", "reasoning":"CD3D supports T cells"}\n```'

class StubBackend(backends.BaseLLMBackend):
    model_name = "offline-test-model"
    temperature = 0.0
    seed = 42
    api_key = "NEVER_RECORD_THIS_KEY"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, json_mode=False):
        self.calls.append((prompt, json_mode))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item

class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "calls.jsonl"

    def events(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_exact_prompt_raw_output_and_settings_without_credentials(self):
        stub = StubBackend([RAW])
        wrapper = recording.RecordingBackend(stub, self.path, {"run_id": "offline"})
        prompt = "Tissue: 肺\nInput Genes: [CD3D, CD3E]"
        self.assertEqual(wrapper.generate(prompt, json_mode=True), RAW)
        start, done = self.events()
        self.assertEqual(start["prompt"], prompt)
        self.assertEqual(done["raw_response_text"], RAW)
        self.assertEqual(start["call_id"], done["call_id"])
        self.assertEqual(start["configured_settings"]["model_name"], "offline-test-model")
        self.assertNotIn(stub.api_key, self.path.read_text())
        self.assertNotIn("api_key", self.path.read_text())
        self.assertEqual(stub.calls, [(prompt, True)])

    def test_annotation_json_retry_keeps_every_received_raw_text(self):
        stub = StubBackend(["malformed output", RAW])
        curator = core.LLMscCurator(backend=recording.RecordingBackend(stub, self.path))
        result = curator.annotate(["CD3D"], max_retries=2, retry_sleep=0)
        self.assertEqual(result, {"cell_type": "T cell", "confidence": "High", "reasoning": "CD3D supports T cells"})
        events = self.events()
        self.assertEqual([e["event"] for e in events], ["call_start", "call_response"] * 2)
        self.assertEqual([e["raw_response_text"] for e in events if e["event"] == "call_response"], ["malformed output", RAW])
        self.assertEqual(len({e["call_id"] for e in events}), 2)

    def test_each_prompt_records_actual_auto_and_explicit_context(self):
        stub = StubBackend([RAW, RAW, RAW])
        curator = core.LLMscCurator(backend=recording.RecordingBackend(stub, self.path))
        curator.last_inferred_context = "T cell"
        curator.annotate(["CD3D"], context={"Tissue": "lung"})
        curator.annotate(["CD3D"], use_auto_context=False)
        curator.annotate(["CD79A"], cell_type="B cell")
        prompts = [e["prompt"] for e in self.events() if e["event"] == "call_start"]
        self.assertIn("Inferred Parent Lineage: T cell", prompts[0])
        self.assertNotIn("Inferred Parent Lineage", prompts[1])
        self.assertNotIn("Tissue: lung", prompts[1])
        self.assertIn("Known Parent Lineage: B cell", prompts[2])
        self.assertNotIn("Inferred Parent Lineage", prompts[2])

    def test_log_initialization_failure_stops_before_inference_and_retries(self):
        stub = StubBackend([RAW])
        wrapper = recording.RecordingBackend(stub, self.path / "missing" / "log.jsonl")
        curator = core.LLMscCurator(backend=wrapper)
        with self.assertRaises(recording.ResponseAuditError):
            curator.annotate(["CD3D"], max_retries=3, retry_sleep=0)
        self.assertEqual(stub.calls, [])

    def test_response_log_failure_does_not_repeat_completed_inference(self):
        stub = StubBackend([RAW, RAW, RAW])
        wrapper = recording.RecordingBackend(stub, self.path)
        curator = core.LLMscCurator(backend=wrapper)
        original = wrapper._write
        def fail_terminal(handle, record):
            if record["event"] == "call_response":
                raise recording.ResponseAuditError("disk full")
            return original(handle, record)
        with patch.object(wrapper, "_write", side_effect=fail_terminal):
            with self.assertRaises(recording.ResponseAuditError):
                curator.annotate(["CD3D"], max_retries=3, retry_sleep=0)
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual([e["event"] for e in self.events()], ["call_start"])

    def test_backend_exception_is_preserved_without_copying_error_secrets(self):
        error = RuntimeError("NEVER_RECORD_THIS_KEY")
        stub = StubBackend([error])
        with self.assertRaises(RuntimeError) as caught:
            recording.RecordingBackend(stub, self.path).generate("test")
        self.assertIs(caught.exception, error)
        self.assertEqual(self.events()[1]["exception_type"], "RuntimeError")
        self.assertNotIn("NEVER_RECORD_THIS_KEY", self.path.read_text())

    def test_invalid_metadata_is_rejected_before_inference(self):
        for metadata in [{"api_key": "secret"}, {"run_id": {"api_key": "secret"}}, {"replicate": float("nan")}]:
            with self.subTest(metadata=metadata):
                stub = StubBackend([RAW])
                with self.assertRaises(recording.ResponseAuditError):
                    recording.RecordingBackend(stub, self.path, metadata).generate("test")
                self.assertEqual(stub.calls, [])

    def test_unwrapped_annotation_keeps_original_schema(self):
        curator = core.LLMscCurator(backend=StubBackend([RAW]))
        result = curator.annotate(["CD3D"])
        self.assertEqual(set(result), {"cell_type", "confidence", "reasoning"})
        self.assertFalse(self.path.exists())

    def test_close_failure_does_not_repeat_completed_inference(self):
        class CannotClose(io.StringIO):
            def close(self):
                raise OSError("close failed")
        stub = StubBackend([RAW, RAW])
        curator = core.LLMscCurator(backend=recording.RecordingBackend(stub, self.path))
        handle = CannotClose()
        with patch.object(Path, "open", return_value=handle):
            with self.assertRaises(recording.ResponseAuditError):
                curator.annotate(["CD3D"], max_retries=3, retry_sleep=0)
        self.assertEqual(len(stub.calls), 1)
        self.assertIn('"call_response"', handle.getvalue())

    def test_package_exports_recording_classes(self):
        name = PKG + "_exports"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "llm_sc_curator" / "__init__.py",
            submodule_search_locations=[str(ROOT / "llm_sc_curator")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        old_scanpy = sys.modules.get("scanpy")
        sys.modules["scanpy"] = types.ModuleType("scanpy")
        try:
            spec.loader.exec_module(module)
        finally:
            if old_scanpy is None:
                sys.modules.pop("scanpy", None)
            else:
                sys.modules["scanpy"] = old_scanpy
        self.assertIn("RecordingBackend", module.__all__)
        self.assertIn("ResponseAuditError", module.__all__)
        self.assertTrue(issubclass(module.ResponseAuditError, RuntimeError))

    def test_multiple_calls_append_with_per_call_metadata(self):
        wrapper = recording.RecordingBackend(StubBackend([RAW, RAW]), self.path, {"cluster_id": "a"})
        wrapper.generate("first")
        wrapper.metadata["cluster_id"] = "b"
        wrapper.generate("second")
        starts = [e for e in self.events() if e["event"] == "call_start"]
        self.assertEqual([e["metadata"]["cluster_id"] for e in starts], ["a", "b"])
        self.assertEqual([e["prompt"] for e in starts], ["first", "second"])

if __name__ == "__main__":
    unittest.main()
