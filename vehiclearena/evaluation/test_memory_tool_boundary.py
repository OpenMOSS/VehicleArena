"""Offline callback replay for the Basic036 float/string failure class."""
import copy
import json
from types import SimpleNamespace

import pytest

from evaluation.multi_agent_runner import (
    _dispatch_memory_search, make_llm_agent_callback,
)
from evaluation.driving_eval import _make_memory_search_tool
from jsonschema import Draft202012Validator
from simulation.memory import SessionHistory
from vehiclearena import VehicleWorld


class ScriptedClient:
    model = 'offline-memory-regression'
    api_base = 'local'
    temperature = 0
    max_tokens = 256

    def __init__(self, calls):
        self.script = list(calls)
        self.requests = []
        self.last_call_metadata = {}

    def chat_with_tools(self, messages, tools):
        self.requests.append(copy.deepcopy(messages))
        name, raw = self.script.pop(0) if self.script else ('finish', '{}')
        call = SimpleNamespace(id=f'tool-{len(self.requests)}', type='function',
            function=SimpleNamespace(name=name, arguments=raw))
        self.last_call_metadata = {'ok': True, 'finish_reason': 'tool_calls'}
        return SimpleNamespace(role='assistant', content='', tool_calls=[call]), 2, 1, 1


def populated_history():
    history = SessionHistory()
    history.record_wake(time_s=81.5, passenger_messages=['please drive smoothly'])
    return history


def test_memory_timestamp_uses_world_time_not_tool_call_order():
    assert SessionHistory.make_timestamp(0.0) == "0m00s"
    assert SessionHistory.make_timestamp(1.8) == "0m02s"
    assert SessionHistory.make_timestamp(81.5) == "1m22s"


@pytest.mark.parametrize('arguments', [
    {'keyword': 'please', 'time_to': '81.9'},
    {'keyword': 'please', 'time_from': '81'},
    {'keyword': 'please', 'time_to': True},
    {'keyword': 'please', 'time_to': None},
    {'keyword': 'please', 'time_to': float('nan')},
    {'keyword': 'please', 'time_to': float('inf')},
    {'keyword': 'please', 'time_from': -1},
    {'keyword': 'please', 'time_from': 82, 'time_to': 81.9},
    {'keyword': 'please', 'context': '1'},
    {'keyword': 'please', 'context': -1},
    {'keyword': 'please', 'context': 101},
    {'keyword': 'please', 'limit': '30'},
    {'keyword': 'please', 'limit': 0},
    {'keyword': 'please', 'limit': 201},
    {'keyword': 'please', 'limit': True},
    {'keyword': 'please', 'scope': 'unknown'},
    {'keyword': 'please', 'unexpected': 1},
    {'keyword': []}, {}, [], None,
])
def test_invalid_memory_arguments_return_receipt_without_search(arguments):
    history = populated_history()
    before = copy.deepcopy(history.recall_all())
    result = _dispatch_memory_search(history, arguments)
    assert result['success'] is False
    assert result['error'] == 'invalid_tool_arguments'
    assert history.recall_all() == before


def test_decimal_simulation_time_is_advertised_and_supported():
    args = {'keyword': 'please', 'scope': 'passenger', 'time_from': 81.4,
            'time_to': 81.9, 'context': 0, 'limit': 30}
    Draft202012Validator(_make_memory_search_tool()['function']['parameters']).validate(args)
    assert 'please drive smoothly' in _dispatch_memory_search(populated_history(), args)


def test_schema_valid_float_counts_are_safe_for_python_range():
    result = _dispatch_memory_search(populated_history(), {
        'keyword': 'please', 'context': 1.0, 'limit': 30.0})
    assert 'please drive smoothly' in result


def test_wrong_type_then_corrected_call_finishes_and_next_wake_runs():
    bad = json.dumps({'keyword': 'please', 'time_to': '81.9'})
    good = json.dumps({'keyword': 'please', 'time_to': 81.9})
    client = ScriptedClient([('memory_search', bad), ('memory_search', good), ('finish', '{}')])
    callback = make_llm_agent_callback('ego', client, max_turns=4)
    world, history = VehicleWorld(), populated_history()
    callback(world, 81.9, ['please drive smoothly'], history, 85)
    logs = callback._state['tool_call_log']
    assert logs[0]['result']['error'] == 'invalid_tool_arguments'
    assert logs[0]['raw_arguments'] == bad
    assert 'please drive smoothly' in logs[1]['result']
    assert logs[2]['function'] == 'finish'
    assert not callback._state['infrastructure_errors']
    assert any(m.get('role') == 'tool' and 'invalid_tool_arguments' in m['content']
               for m in client.requests[1])
    callback(world, 82.9, [], history, 86)
    assert len(callback._state['all_messages']) == 2
    assert not callback._state['infrastructure_errors']


@pytest.mark.parametrize('raw', ['{', '[]', 'null', '"text"', '42'])
def test_non_object_or_malformed_arguments_do_not_execute_defaults(raw):
    client = ScriptedClient([('finish', raw), ('finish', '{}')])
    callback = make_llm_agent_callback('ego', client)
    callback(VehicleWorld(), 0, [], SessionHistory(), 0)
    assert callback._state['tool_call_log'][0]['result']['error'] == 'invalid_tool_arguments'
    assert len(client.requests) == 2  # invalid finish must not terminate the wake


def test_unexpected_dispatch_failure_preserves_response_arguments_and_stack():
    class BrokenHistory(SessionHistory):
        def search(self, **kwargs):
            started = callback._state['tool_call_log'][-1]
            assert started['execution_status'] == 'started'
            assert started['arguments'] == kwargs
            raise RuntimeError('injected internal search bug')

    raw = '{"keyword":"please","time_to":81.9}'
    client = ScriptedClient([('memory_search', raw)])
    callback = make_llm_agent_callback('ego', client)
    with pytest.raises(RuntimeError, match='injected internal search bug'):
        callback(VehicleWorld(), 81.9, [], BrokenHistory(), 85)
    state = callback._state
    record, = state['tool_call_log']
    assert record['raw_arguments'] == raw
    assert record['execution_status'] == 'raised'
    assert 'test_memory_tool_boundary.py' in record['traceback']
    assert 'injected internal search bug' in record['traceback']
    error = state['infrastructure_errors'][-1]
    assert error['tool_call']['function'] == 'memory_search'
    assert error['model_response']['tool_calls'][0]['function']['arguments'] == raw
    partial, = state['all_messages']
    assert partial['incomplete'] is True
    assert partial['messages'][-1]['tool_calls'][0]['function']['arguments'] == raw
    # These are the exact state keys serialized by the experiment exporter.
    json.dumps({k: state[k] for k in ('infrastructure_errors', 'tool_call_log', 'all_messages')})
    assert state['_active_tool_call'] is None
