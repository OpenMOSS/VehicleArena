"""Contracts for the calibrated ratio-scheduled test launcher."""

from collections import Counter, deque

from scripts.run_test_suite import (
    DEFAULT_SELECTION, MODEL_PRESETS, _load_scene_ids,
)
from scripts.run_llm_suite import choose_ratio_study, parse_study_ratio


def test_frozen_test_split_is_80_basic_32_calibrated_multillm():
    scene_ids = _load_scene_ids(DEFAULT_SELECTION)
    assert len(scene_ids) == 112
    assert Counter(
        "Basic" if scene_id.startswith("basic_") else "MultiLLM"
        for scene_id in scene_ids
    ) == {"Basic": 80, "MultiLLM": 32}


def test_ratio_scheduler_keeps_three_slot_pool_at_two_to_one():
    queues = {
        "Basic": deque(range(8)),
        "MultiLLM": deque(range(4)),
    }
    weights = parse_study_ratio("2:1")
    active = {"Basic": 0, "MultiLLM": 0}
    submitted = []
    for _ in range(3):
        study = choose_ratio_study(queues, active, weights)
        submitted.append(study)
        queues[study].popleft()
        active[study] += 1
    assert submitted == ["Basic", "MultiLLM", "Basic"]
    assert active == {"Basic": 2, "MultiLLM": 1}

    # A completed Basic slot is refilled with Basic; a completed MultiLLM
    # slot is refilled with MultiLLM, preserving the active target.
    active["Basic"] -= 1
    study = choose_ratio_study(queues, active, weights)
    assert study == "Basic"
    active[study] += 1
    active["MultiLLM"] -= 1
    study = choose_ratio_study(queues, active, weights)
    assert study == "MultiLLM"


def test_model_presets_do_not_embed_provider_endpoints():
    assert MODEL_PRESETS == {
        "qwen3.8-27b": {"model": "Qwen3.8-27B"},
        "kimi-k3": {"model": "Kimi-K3", "reasoning_effort": "max"},
        "qwen3-vl-8b-instruct": {"model": "Qwen3-VL-8B-Instruct"},
        "qwen3-vl-32b-instruct": {"model": "Qwen3-VL-32B-Instruct"},
        "gpt-5.6-sol": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
    }
