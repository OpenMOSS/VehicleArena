"""Regression contract for the two formal VehicleArena studies."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.experiments.manifest import load_manifest
from evaluation.experiments.manifest import (
    ExperimentManifest, ExperimentVariant,
)
from evaluation.experiments.batch_runner import (
    ExperimentBatchRunner, _task_cabin_summary,
)
from evaluation.experiments.pipelines import prepare_scene_manifests
from evaluation.experiments.scene_catalog import (
    CATALOG_SCHEMA, FORMAL_DAYNIGHT_BRIGHT_PERIODS,
    FORMAL_DAYNIGHT_DARK_PERIODS,
    FORMAL_DAYNIGHT_FIRST_CHANGE_FRACTION, FORMAL_DAYNIGHT_PATTERNS,
    FORMAL_DAYNIGHT_PERIODS, FORMAL_DAYNIGHT_SECOND_CHANGE_FRACTION,
    FORMAL_WEATHER_CONDITIONS,
    FORMAL_WEATHER_FIRST_CHANGE_FRACTION, FORMAL_WEATHER_PATTERNS,
    FORMAL_WEATHER_SECOND_CHANGE_FRACTION,
    FORMAL_WEATHER_TARGET_CONDITIONS, LONGITUDINAL_COHORT_MAXIMUM,
    PINNED_REBUILT_ENVIRONMENT_PATTERNS,
    LONGITUDINAL_COHORT_MINIMUM, RELATED_BACKGROUND_BASE_MINIMUM,
    RELATED_BACKGROUND_MINIMUM,
    iter_catalog_scenes,
)
from module.daynight import DayNight
from simulation.multi_sim_engine import MultiScenario
from simulation.lane_level_runtime import LaneGeometryRuntime
from evaluation.experiments.environment_design import (
    DAYNIGHT_EDGES,
    WEATHER_TRANSITION_EDGES,
    weather_condition_allowed,
    weather_condition_month_allowed,
)


CATALOG = Path(__file__).parent / "experiments" / "scenarios"


def test_formal_catalog_has_only_two_studies_and_expected_counts():
    payload = json.loads((CATALOG / "catalog.json").read_text())
    assert payload["schema"] == CATALOG_SCHEMA
    assert payload["counts"] == {"Basic": 180, "MultiLLM": 40}
    assert payload["scene_count"] == 220
    assert set(payload["networks"]) == {e["network"] for e in payload["entries"]}
    matrix_counts = {"Basic": 180, "MultiLLM": 20}
    for study in payload["counts"]:
        matrix = json.loads((CATALOG / study / "matrix.json").read_text())
        assert matrix["task_count"] == matrix["scene_count"] == len(
            matrix["scene_ids"])
        assert matrix["scene_count"] == matrix_counts[study]
        catalog_scene_ids = {
            entry["scene_id"] for entry in payload["entries"]
            if entry["experiment_id"] == study
        }
        assert set(matrix["scene_ids"]) <= catalog_scene_ids
    assert not any((CATALOG / f"E{index}").exists() for index in range(9))


def test_every_evaluated_vehicle_has_a_distinct_destination_node():
    for _, scenario_path, _ in iter_catalog_scenes(CATALOG):
        scenario = json.loads(scenario_path.read_text())
        for vehicle in scenario["vehicles"]:
            if vehicle.get("is_evaluated"):
                assert vehicle["initial_node"] != vehicle["destination_node"]


def test_every_formal_task_has_auditable_related_and_cohort_trips():
    payload = json.loads((CATALOG / "catalog.json").read_text())
    summary = payload["related_background_traffic"]
    assert summary["minimum_per_task"] == RELATED_BACKGROUND_MINIMUM == 8
    assert (summary["base_minimum_per_task"]
            == RELATED_BACKGROUND_BASE_MINIMUM == 5)
    assert (summary["longitudinal_cohort_minimum_per_task"]
            == LONGITUDINAL_COHORT_MINIMUM == 3)
    assert (summary["longitudinal_cohort_maximum_per_task"]
            == LONGITUDINAL_COHORT_MAXIMUM == 8)
    assert summary["task_count"] == payload["scene_count"] == 220
    assert 2400 <= summary["added_vehicle_count"] <= 2500
    assert summary["longitudinal_cohort_vehicle_count"] == 1335
    assert (summary["added_vehicle_count"]
            - summary["longitudinal_cohort_vehicle_count"] == 1100)
    assert (summary["route_aligned_vehicle_count"]
            + summary["crossing_vehicle_count"]
            == summary["added_vehicle_count"])
    for entry, scenario_path, expected_path in iter_catalog_scenes(CATALOG):
        scenario = json.loads(scenario_path.read_text())
        expected = json.loads(expected_path.read_text())
        related = scenario["experiment_scene"][
            "related_background_traffic"]
        related_ids = related["vehicle_ids"]
        cohort_ids = related["longitudinal_cohort_vehicle_ids"]
        catalog_related = entry["related_background_traffic"]
        assert catalog_related["vehicle_ids"] == related_ids
        assert catalog_related["longitudinal_cohort_vehicle_count"] == len(
            cohort_ids)
        assert catalog_related["route_aligned_vehicle_count"] == len(
            related["route_aligned_vehicle_ids"])
        assert catalog_related["crossing_vehicle_count"] == len(
            related["crossing_vehicle_ids"])
        assert len(cohort_ids) in range(
            LONGITUDINAL_COHORT_MINIMUM,
            LONGITUDINAL_COHORT_MAXIMUM + 1)
        assert len(related_ids) == len(set(related_ids))
        assert len(related_ids) == RELATED_BACKGROUND_BASE_MINIMUM + len(
            cohort_ids)
        assert set(cohort_ids).issubset(
            related["route_aligned_vehicle_ids"])
        assert set(related_ids).issubset(
            scenario["experiment_scene"]["sumo_background_vehicle_ids"])
        vehicles = {vehicle["vehicle_id"]: vehicle
                    for vehicle in scenario["vehicles"]}
        assert all(
            vehicles[vehicle_id]["agent_config"]["type"] == "sumo"
            and not vehicles[vehicle_id]["is_evaluated"]
            and vehicles[vehicle_id]["initial_node"]
            != vehicles[vehicle_id]["destination_node"]
            for vehicle_id in related_ids)
        assertion = next(
            item for item in expected["setup_assertions"]
            if item["kind"] == "related_background_traffic")
        assert assertion["route_aligned_vehicle_ids"] == related[
            "route_aligned_vehicle_ids"]
        assert assertion["crossing_vehicle_ids"] == related[
            "crossing_vehicle_ids"]
        assert assertion["longitudinal_cohort_vehicle_ids"] == cohort_ids


def test_required_turn_tasks_replace_optional_passing_templates():
    from evaluation.experiments.scene_catalog import RETIRED_PASSING_TEMPLATES
    catalog = json.loads((CATALOG / "catalog.json").read_text())
    assert not any(
        e["source_template_id"].split("__", 1)[0]
        in RETIRED_PASSING_TEMPLATES
        and "sequence_entry" not in e.get("tags", [])
        and "separated_two_way_meeting" not in e.get("tags", [])
        for e in catalog["entries"])
    selected = [e for e in catalog["entries"] if "required_turn" in e["scene_id"]]
    assert len(selected) == 18
    assert {int(e["scene_id"].split("_")[1]) for e in selected} == (
        set(range(101, 109)) | set(range(196, 206)))
    assert len({e["network"] for e in selected}) == 18
    for entry in selected:
        scenario = json.loads((CATALOG / entry["scenario"]).read_text())
        expected = json.loads((CATALOG / entry["expected"]).read_text())
        assert any(a["kind"] == "required_lane_change_turn" for a in expected["setup_assertions"])
        assert scenario["vehicles"][0]["agent_config"]["type"] == "llm"
        assert [a["type"] for a in scenario["vehicles"][0]["initial_physical_state"]["lane_route_actions"]] == ["lane_change", "connector"]


def test_four_middle_queue_tasks_extend_signal_baseline():
    catalog = json.loads((CATALOG / "catalog.json").read_text())
    signals = [e for e in catalog["entries"] if e["source_template_id"].split("__")[0] == "baseline_signalized_intersection"]
    middle = [e for e in signals if "__middle__" in e["scene_id"]]
    assert len(signals) == 9 and len(middle) == 4
    assert {int(e["scene_id"].split("_")[1]) for e in middle} == {109, 110, 111, 112}
    for e in middle:
        s = json.loads((CATALOG / e["scenario"]).read_text())
        v = {v["vehicle_id"]: v for v in s["vehicles"]}
        assert s["experiment_scene"]["focal_vehicle_id"] == "ego"
        assert v["ego"]["agent_config"]["type"] == "llm"
        assert v["front"]["agent_config"]["type"] == v["rear"]["agent_config"]["type"] == "sumo"
        assert v["front"]["initial_physical_state"]["progress"] > v["ego"]["initial_physical_state"]["progress"] > v["rear"]["initial_physical_state"]["progress"]


def test_unsignalized_challenges_extend_existing_baselines():
    from evaluation.experiments.scene_catalog import (
        BASIC_UNSIGNALIZED_EXPANSION_SITES,
        UNSIGNALIZED_CHALLENGE_SITES,
    )
    catalog = json.loads((CATALOG / "catalog.json").read_text())
    selected = [e for e in catalog["entries"] if e["source_template_id"].split("__")[0] == "baseline_unsignalized_intersection"]
    assert len(selected) == 18
    new = [e for e in selected if "__challenge__" in e["scene_id"]]
    assert {int(e["scene_id"].split("_")[1]) for e in new} == (
        set(range(113, 117)) | set(range(177, 187)))
    assert {e["network"] for e in new} == {
        site[0] for site in (
            *UNSIGNALIZED_CHALLENGE_SITES,
            *BASIC_UNSIGNALIZED_EXPANSION_SITES,
        )}
    for e in new:
        s = json.loads((CATALOG / e["scenario"]).read_text())
        expected = json.loads((CATALOG / e["expected"]).read_text())
        assert [v["vehicle_id"] for v in s["vehicles"][:4]] == ["ego", "cross_1", "cross_2", "cross_3"]
        assert [v["agent_config"]["type"] for v in s["vehicles"][:4]] == ["llm", "sumo", "sumo", "sumo"]
        assert any(a["kind"] == "unsignalized_crossing_stream" for a in expected["setup_assertions"])
        basis = s["experiment_scene"]["weather_profile"].get("schedule_basis_s")
        if basis is not None:
            last_environment_event_s = max(
                [60.0 * frame["t"] for frame in s.get(
                    "weather_keyframes", [])]
                + [60.0 * frame["t"] for frame in s.get(
                    "daynight_keyframes", [])],
                default=0.0,
            )
            assert last_environment_event_s + 2.0 <= s["total_time_s"]


def test_half_of_each_study_has_all_weather_change_targets():
    by_study = {"Basic": [], "MultiLLM": []}
    controls = {"Basic": 0, "MultiLLM": 0}
    patterns = {"Basic": [], "MultiLLM": []}
    observed_changes = set()
    for entry, scenario_path, expected_path in iter_catalog_scenes(CATALOG):
        raw = json.loads(scenario_path.read_text())
        expected = json.loads(expected_path.read_text())
        profile = raw["experiment_scene"]["weather_profile"]
        task_index = int(entry["scene_id"].split("_", 2)[1])
        is_transition = profile["mode"] == "transition"
        assert is_transition == (task_index % 2 == 1)
        assert entry["weather_profile"] == profile
        if not is_transition:
            controls[entry["experiment_id"]] += 1
            assert "weather_keyframes" not in raw
            assert "stable_weather_control" in entry["tags"]
            continue

        by_study[entry["experiment_id"]].append(
            profile["primary_condition"])
        patterns[entry["experiment_id"]].append(profile["pattern"])
        basis_s = profile["schedule_basis_s"]
        # The authored schedule stays frozen when traffic is recalibrated.
        assert max(60.0 * frame["t"] for frame in raw[
            "weather_keyframes"]) + 10.0 <= raw["total_time_s"] + 1e-6
        assert (basis_s * FORMAL_WEATHER_FIRST_CHANGE_FRACTION[0] - 0.1
                <= profile["change_at_s"]
                <= basis_s * FORMAL_WEATHER_FIRST_CHANGE_FRACTION[1] + 0.1)
        assert raw["weather_keyframes"][0]["condition"] == profile[
            "initial_condition"]
        assert raw["weather_keyframes"][1]["condition"] == profile[
            "target_condition"]
        assert [item["condition"] for item in raw[
            "weather_keyframes"]] == profile["conditions"]
        assert all(edge in WEATHER_TRANSITION_EDGES for edge in zip(
            profile["conditions"], profile["conditions"][1:]))
        assert all(weather_condition_allowed(
            entry["network"], condition) for condition in profile[
                "conditions"])
        assert all(weather_condition_month_allowed(
            entry["network"], condition, profile["climate_month"])
            for condition in profile["conditions"])
        observed_changes.update(
            keyframe["condition"]
            for keyframe in raw["weather_keyframes"][1:])
        if profile["pattern"] == "onset":
            assert len(raw["weather_keyframes"]) == 2
            assert profile["second_condition"] is None
            assert profile["second_change_at_s"] is None
        else:
            assert len(raw["weather_keyframes"]) == 3
            assert (basis_s * FORMAL_WEATHER_SECOND_CHANGE_FRACTION[0]
                    - 0.1 <= profile["second_change_at_s"]
                    <= basis_s * FORMAL_WEATHER_SECOND_CHANGE_FRACTION[1]
                    + 0.1)
            assert profile["second_change_at_s"] > profile["change_at_s"]
            assert raw["weather_keyframes"][-1]["condition"] == profile[
                "second_condition"]
        assert (max(60.0 * keyframe["t"]
                    for keyframe in raw["weather_keyframes"])
                + 2.0 <= basis_s + 1e-9)
        assert any(
            assertion["kind"] == "weather_transition"
            for assertion in expected["setup_assertions"])
        assert "weather_change" in entry["tags"]

    assert {study: len(targets) for study, targets in by_study.items()} == {
        "Basic": 90,
        "MultiLLM": 20,
    }
    assert controls == {"Basic": 90, "MultiLLM": 20}
    assert all(
        set(targets) == set(FORMAL_WEATHER_TARGET_CONDITIONS)
        for targets in by_study.values())
    assert {
        study: {
            pattern: values.count(pattern)
            for pattern in FORMAL_WEATHER_PATTERNS
        }
        for study, values in patterns.items()
    } == {
        "Basic": {
            "onset": 30,
            "temporary": 30,
            "evolve": 30,
        },
        "MultiLLM": {
            "onset": 8,
            "temporary": 5,
            "evolve": 7,
        },
    }
    assert observed_changes == set(FORMAL_WEATHER_CONDITIONS)


def test_environment_treatments_form_exact_factorial_quarters():
    catalog = json.loads((CATALOG / "catalog.json").read_text())
    assert catalog["environment_factorial"] == {
        "control": 55,
        "weather_only": 55,
        "daynight_only": 55,
        "combined": 55,
    }
    expected_by_remainder = {
        1: "weather_only",
        2: "daynight_only",
        3: "combined",
        0: "control",
    }
    by_study = {"Basic": [], "MultiLLM": []}
    for entry in catalog["entries"]:
        index = int(entry["scene_id"].split("_", 2)[1])
        assert entry["environment_group"] == expected_by_remainder[
            index % 4]
        by_study[entry["experiment_id"]].append(
            entry["environment_group"])
    assert {
        study: {group: groups.count(group) for group in set(groups)}
        for study, groups in by_study.items()
    } == {
        "Basic": {
            "control": 45,
            "weather_only": 45,
            "daynight_only": 45,
            "combined": 45,
        },
        "MultiLLM": {
            "control": 10,
            "weather_only": 10,
            "daynight_only": 10,
            "combined": 10,
        },
    }


def test_environment_keyframes_are_strictly_before_task_deadlines():
    for _, scenario_path, _ in iter_catalog_scenes(CATALOG):
        scenario = json.loads(scenario_path.read_text())
        deadline_s = float(scenario["total_time_s"])
        for key in ("weather_keyframes", "daynight_keyframes"):
            event_times_s = [
                60.0 * float(frame["t"])
                for frame in scenario.get(key, [])[1:]
            ]
            assert all(time_s < deadline_s for time_s in event_times_s), (
                scenario["scenario_id"], key, deadline_s, event_times_s)


def test_half_of_each_study_has_stratified_daynight_changes():
    transitions = {"Basic": 0, "MultiLLM": 0}
    patterns = {"Basic": [], "MultiLLM": []}
    observed_periods = set()
    pinned_supplemental_seen = set()
    for entry, scenario_path, expected_path in iter_catalog_scenes(CATALOG):
        raw = json.loads(scenario_path.read_text())
        expected = json.loads(expected_path.read_text())
        profile = raw["experiment_scene"]["daynight_profile"]
        task_index = int(entry["scene_id"].split("_", 2)[1])
        is_transition = profile["mode"] == "transition"
        assert is_transition == (task_index % 4 in (2, 3))
        assert entry["daynight_profile"] == profile
        if not is_transition:
            pinned_pattern = PINNED_REBUILT_ENVIRONMENT_PATTERNS.get(
                entry["scene_id"], {}).get("daynight")
            if pinned_pattern is None:
                assert "daynight_keyframes" not in raw
            else:
                pinned_supplemental_seen.add(entry["scene_id"])
                frames = raw["daynight_keyframes"]
                assert frames[0]["t"] == 0.0
                assert all(
                    left["t"] < right["t"]
                    for left, right in zip(frames, frames[1:]))
                assert all(
                    edge in DAYNIGHT_EDGES
                    for edge in zip(
                        [frame["period"] for frame in frames],
                        [frame["period"] for frame in frames][1:]))
                assert max(60.0 * frame["t"] for frame in frames) + 10.0 \
                    <= raw["total_time_s"] + 1e-6
            assert "stable_daynight_control" in entry["tags"]
            continue

        transitions[entry["experiment_id"]] += 1
        patterns[entry["experiment_id"]].append(profile["pattern"])
        basis_s = profile["schedule_basis_s"]
        assert max(60.0 * frame["t"] for frame in raw[
            "daynight_keyframes"]) + 10.0 <= raw["total_time_s"] + 1e-6
        assert (basis_s * FORMAL_DAYNIGHT_FIRST_CHANGE_FRACTION[0] - 0.1
                <= profile["change_at_s"]
                <= basis_s * FORMAL_DAYNIGHT_FIRST_CHANGE_FRACTION[1] + 0.1)
        assert raw["daynight_keyframes"][0]["period"] == profile[
            "initial_period"]
        assert raw["daynight_keyframes"][1]["period"] == profile[
            "target_period"]
        assert [item["period"] for item in raw[
            "daynight_keyframes"]] == profile["period_sequence"]
        assert all(edge in DAYNIGHT_EDGES for edge in zip(
            profile["period_sequence"], profile["period_sequence"][1:]))
        observed_periods.update(
            keyframe["period"] for keyframe in raw["daynight_keyframes"])
        if profile["pattern"] == "single_step":
            assert len(raw["daynight_keyframes"]) == 2
            assert profile["second_period"] is None
            assert profile["second_change_at_s"] is None
        else:
            assert len(raw["daynight_keyframes"]) == 3
            assert (basis_s * FORMAL_DAYNIGHT_SECOND_CHANGE_FRACTION[0]
                    - 0.1 <= profile["second_change_at_s"]
                    <= basis_s * FORMAL_DAYNIGHT_SECOND_CHANGE_FRACTION[1]
                    + 0.1)
            assert profile["second_change_at_s"] > profile["change_at_s"]
            assert raw["daynight_keyframes"][-1]["period"] == profile[
                "second_period"]
        assert (max(60.0 * keyframe["t"]
                    for keyframe in raw["daynight_keyframes"])
                + 2.0 <= basis_s + 1e-9)
        assert any(
            assertion["kind"] == "daynight_transition"
            for assertion in expected["setup_assertions"])
        assert "daynight_change" in entry["tags"]

    assert transitions == {"Basic": 90, "MultiLLM": 20}
    assert pinned_supplemental_seen == {
        scene_id for scene_id, overrides
        in PINNED_REBUILT_ENVIRONMENT_PATTERNS.items()
        if "daynight" in overrides
    }
    assert {
        study: {
            pattern: values.count(pattern)
            for pattern in FORMAL_DAYNIGHT_PATTERNS
        }
        for study, values in patterns.items()
    } == {
        "Basic": {
            "single_step": 30,
            "forward_two_step": 30,
            "light_boundary": 30,
        },
        "MultiLLM": {
            "single_step": 8,
            "forward_two_step": 6,
            "light_boundary": 6,
        },
    }
    assert observed_periods == set(FORMAL_DAYNIGHT_PERIODS)


def test_daynight_darkness_classification_matches_scoring_policy():
    assert {
        period for period in FORMAL_DAYNIGHT_PERIODS
        if DayNight._DAYLIGHT_MAP[DayNight.TimePeriod(period)] < 30
    } == {"dawn", "night"}
    module = DayNight()
    for period in FORMAL_DAYNIGHT_PERIODS:
        status = DayNight.daynight_set_time.__wrapped__(module, period)
        assert status["is_dark"] == (period in {"dawn", "night"})


def test_basic_has_one_llm_and_sumo_owns_every_background_actor():
    for entry, scenario_path, _ in iter_catalog_scenes(CATALOG):
        if entry["experiment_id"] != "Basic":
            continue
        raw = json.loads(scenario_path.read_text())
        scenario = MultiScenario.from_dict(raw)
        llm_vehicles = [v for v in scenario.vehicles if v.agent_type == "llm"]
        assert len(llm_vehicles) == 1
        assert all(
            v.agent_type == "sumo" for v in scenario.vehicles
            if v.vehicle_id != llm_vehicles[0].vehicle_id)
        assert all(p.agent_type == "sumo" for p in scenario.pedestrians)
        for vehicle in raw["vehicles"]:
            if vehicle["agent_config"]["type"] == "sumo":
                assert not {"target_speed_kmh", "desired_speed_kmh"}.intersection(
                    vehicle.get("initial_physical_state", {}))
        assert "episode_seed" not in raw


def test_multi_llm_declares_focal_fixed_peers_and_sumo_background():
    for entry, scenario_path, _ in iter_catalog_scenes(CATALOG):
        if entry["experiment_id"] != "MultiLLM":
            continue
        raw = json.loads(scenario_path.read_text())
        scene = raw["experiment_scene"]
        llm_ids = {
            item["vehicle_id"] for item in raw["vehicles"]
            if item["agent_config"]["type"] == "llm"}
        assert scene["focal_vehicle_id"] in llm_ids
        assert set(scene["fixed_peer_vehicle_ids"]).issubset(llm_ids)
        assert scene["fixed_peer_vehicle_ids"]
        assert all(
            item["agent_config"]["type"] == "sumo"
            for item in raw["vehicles"] if item["vehicle_id"] not in llm_ids)


def test_four_way_scenes_have_strictly_staggered_approach_distances():
    for entry, scenario_path, _ in iter_catalog_scenes(CATALOG):
        if "four_way" not in entry["tags"]:
            continue
        raw = json.loads(scenario_path.read_text())
        runtime = LaneGeometryRuntime.load(
            str(
                Path(__file__).parent / ".." / "simulation"
                / "road_networks"
                / f"{raw['road_network_id']}_lane_level.json"
            )
        )
        scene = raw["experiment_scene"]
        four_way_ids = {
            scene["focal_vehicle_id"], *scene["fixed_peer_vehicle_ids"]}
        distances = []
        for vehicle in raw["vehicles"]:
            if vehicle["vehicle_id"] not in four_way_ids:
                continue
            placement = vehicle["initial_physical_state"]
            lane = runtime._lane_by_id[placement["lane_id"]]
            distances.append(
                (1.0 - float(placement["progress"]))
                * float(lane["length_m"])
            )
        assert all(
            following > leading + 1.0
            for leading, following in zip(distances, distances[1:])
        ), (entry["scene_id"], distances)


def test_catalog_contains_no_personality_or_runtime_seed_fields():
    forbidden = {
        "driver_plugin", "driver_params", "pedestrian_plugin",
        "pedestrian_params", "episode_seed", "traffic_seed",
        "pedestrian_seed", "seed",
    }

    def walk(value, parent_key=None):
        if isinstance(value, dict):
            disallowed = forbidden - (
                {"seed"} if parent_key == "npc_behavior" else set())
            assert not disallowed.intersection(value)
            for key, item in value.items():
                walk(item, key)
        elif isinstance(value, list):
            for item in value:
                walk(item, parent_key)

    walk(json.loads((CATALOG / "catalog.json").read_text()))
    for _, scenario_path, expected_path in iter_catalog_scenes(CATALOG):
        walk(json.loads(scenario_path.read_text()))
        walk(json.loads(expected_path.read_text()))


def test_manifest_pipeline_preserves_one_variant_per_task(tmp_path):
    paths = prepare_scene_manifests(tmp_path, CATALOG)
    assert set(paths) == {"Basic", "MultiLLM"}
    basic = load_manifest(paths["Basic"])
    multi = load_manifest(paths["MultiLLM"])
    assert len(basic.variants) == 180
    assert len(multi.variants) == 40
    assert all(item.requires_llm for item in basic.variants + multi.variants)
    assert basic.metadata["random_seed_configurable"] is False
    assert multi.metadata["background_authority"] == "sumo"


def test_sumo_reference_accepts_personal_agent(tmp_path):
    manifest = ExperimentManifest(
        experiment_id="Basic", description="test",
        variants=[ExperimentVariant(
            variant_id="reference", base_scenario_id="reference",
            factors={}, scenario={
                "scenario_id": "reference",
                "road_network_id": "beijing_guomao",
            }, requires_llm=True)])
    runner = ExperimentBatchRunner(
        manifest, tmp_path, sumo_reference=True,
        personal_agent_runtime_config={"enabled": True})

    assert runner.personal_agent_enabled is True
    assert runner.callback_personal_agent_config["enabled"] is True


def test_source_hash_mismatch_is_provenance_warning(tmp_path):
    manifest = ExperimentManifest(
        experiment_id="Basic", description="test", source_hash="old-source",
        variants=[ExperimentVariant(
            variant_id="reference", base_scenario_id="reference",
            factors={}, scenario={
                "scenario_id": "reference",
                "road_network_id": "beijing_guomao",
            }, requires_llm=True)])
    runner = ExperimentBatchRunner(manifest, tmp_path)

    runner._validate_runtime_integrity(["reference"])

    assert [item["type"] for item in runner.runtime_integrity_warnings] == [
        "source_hash_mismatch"]


def test_multillm_aggregate_cabin_summary_uses_focal_vehicle():
    result = {
        "evaluation": {"cabin": {
            "score": 0.45,
            "task_completion_rate": 0.25,
        }},
        "vehicles": {
            "ego": {"cabin_evaluation": {
                "score": 0.8,
                "task_completion_rate": 0.5,
            }},
            "peer": {"cabin_evaluation": {
                "score": 0.1,
                "task_completion_rate": 0.0,
            }},
        },
    }

    assert _task_cabin_summary(
        result, focal_id="ego", focal_only=True) == (0.8, 0.5)
    assert _task_cabin_summary(
        result, focal_id="ego", focal_only=False) == (0.45, 0.25)

    result["vehicles"]["ego"]["cabin_evaluation"] = {
        "score": None,
        "task_completion_rate": None,
    }
    assert _task_cabin_summary(
        result, focal_id="ego", focal_only=True) == (None, None)
    del result["vehicles"]["ego"]["cabin_evaluation"]
    assert _task_cabin_summary(
        result, focal_id="ego", focal_only=True) == (None, None)


def test_multillm_reference_requires_fixed_peers_and_selects_focal_policy(
        tmp_path):
    manifest = ExperimentManifest(
        experiment_id="MultiLLM", description="test",
        variants=[ExperimentVariant(
            variant_id="reference", base_scenario_id="reference",
            factors={}, scenario={
                "scenario_id": "reference",
                "road_network_id": "beijing_guomao",
            }, requires_llm=True)])
    try:
        ExperimentBatchRunner(manifest, tmp_path, sumo_reference=True)
    except ValueError as exc:
        assert "fixed peer model" in str(exc)
    else:
        raise AssertionError("MultiLLM reference accepted no peer model")

    runner = ExperimentBatchRunner(
        manifest, tmp_path, sumo_reference=True,
        fixed_peer_models=["peer-model"])
    assert runner.reference_policy == "focal_sumo_fixed_peers"
