"""Worker diagnostics must not install asynchronous all-thread dump timers."""

from types import SimpleNamespace

import pytest

from scripts import run_llm_suite as suite


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("failed", 1)])
def test_worker_past_300_seconds_keeps_logs_without_dump_timer(
        monkeypatch, tmp_path, capsys, status, exit_code):
    enabled = []
    monkeypatch.setattr(suite.faulthandler, "enable", lambda **kw: enabled.append(kw))

    def forbidden(*args, **kwargs):
        pytest.fail("Workers must not install or manage delayed traceback dumps")

    monkeypatch.setattr(suite.faulthandler, "dump_traceback_later", forbidden)
    monkeypatch.setattr(suite.faulthandler, "cancel_dump_traceback_later", forbidden)
    clock = [0.0]
    monkeypatch.setattr(suite.time, "monotonic", lambda: clock[0])

    class Client:
        model = "offline-test"

        def chat(self, *args, **kwargs):
            clock[0] += 301.0  # virtual elapsed time; no paid API or 5-minute sleep
            self.last_call_metadata = {"ok": True, "total_tokens": 3,
                                       "finish_reason": "stop"}
            return "ok"

        chat_with_tools = chat

    monkeypatch.setattr("evaluation.multi_agent_runner.AgentClient", Client)

    def run_manifest(*args, **kwargs):
        client = Client()
        assert client.chat([]) == "ok"
        assert client.chat_with_tools([], []) == "ok"
        return {"variants": {"test-scene": {"status": status}}}

    monkeypatch.setattr("evaluation.experiments.batch_runner.run_manifest", run_manifest)
    suite.write_json(tmp_path / "configuration.json", {})
    args = SimpleNamespace(output=tmp_path, study="Basic", worker="test-scene",
                           model="offline-test", base_url="https://example.invalid/v1",
                           context_window=32768)
    assert suite.worker(args) == exit_code
    assert enabled == [{"all_threads": False}]
    output = capsys.readouterr().out
    assert output.count('"event": "model_call_start"') == 2
    assert output.count('"event": "model_call"') == 2
    assert output.count('"elapsed_s": 301.0') == 2
