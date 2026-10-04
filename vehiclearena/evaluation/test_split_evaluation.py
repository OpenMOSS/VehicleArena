"""Regression tests for split cabin/driving evaluation.

Run from the repository root with:

    PYTHONPATH=vehiclearena .venv/bin/python -m unittest \
        evaluation.test_split_evaluation
"""

from __future__ import annotations

import unittest
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.cabin_evaluator import (
    CabinYamlEvaluator,
    _candidate_action_sets,
)
from evaluation.driving_evaluator import DrivingEvaluator
from evaluation.driving_process_score import calculate_driving_process_score
from evaluation.layer_scoring import (
    cabin_layer_score_100, single_vehicle_layer_score_100,
)
from rules.rule_loader import Rule, RuleLoader
from simulation.ground_truth_rules import AcceptableAction
from simulation.multi_sim_engine import (
    MultiScenario, MultiSimEngine, MultiSimResult, VehicleResult,
)
from simulation.traffic_manager import CollisionEvent
from simulation.vehicle_state import VehicleState
from vehiclearena import VehicleWorld


class CabinYamlEvaluatorTest(unittest.TestCase):

    def test_rule_loader_does_not_silently_load_zero_rules(self):
        loader = RuleLoader()
        self.assertGreater(len(loader.env_rules), 0)
        self.assertGreater(len(loader.user_intent_rules), 0)

    def test_sumo_npc_executes_only_world_visible_equipment_rules(self):
        engine = object.__new__(MultiSimEngine)
        npc = VehicleWorld()
        engine._vw = {"npc": npc}

        applied = engine._apply_npc_external_equipment_rules("npc", [
            "vw.lowBeamHeadlight.switch('on')",
            "vw.positionLight.carcontrol_positionLight_switch(True)",
            "vw.centerInformationDisplay."
            "carcontrol_centerInformationDisplay_brightness_decrease("
            "degree='large')",
        ])

        self.assertEqual(applied, [
            "vw.lowBeamHeadlight.switch('on')",
            "vw.positionLight.carcontrol_positionLight_switch(True)",
        ])
        self.assertEqual(npc.lowBeamHeadlight.mode.value, "on")
        self.assertTrue(npc.positionLight.is_on)

    def test_sumo_npc_environment_rule_reaches_physical_lamp_state(self):
        scenario = MultiScenario.from_dict({
            "scenario_id": "npc_environment_lamp_rule",
            "road_network_id": "beijing_guomao",
            "total_time_s": 0.2,
            "tick_interval_s": 1.0,
            "daynight_keyframes": [{"t": 0, "period": "night"}],
            "vehicles": [{
                "vehicle_id": "npc",
                "initial_node": "n33399858",
                "destination_node": "n35722739",
                "is_evaluated": False,
                "agent_config": {"type": "sumo"},
            }],
        })

        def sumo_noop(*_args, **_kwargs):
            return []

        sumo_noop._agent_type = "sumo"
        engine = MultiSimEngine(scenario)
        result = engine.run({"npc": sumo_noop})

        self.assertEqual(engine._vw["npc"].lowBeamHeadlight.mode.value, "on")
        self.assertTrue(engine._vw["npc"].positionLight.is_on)
        self.assertTrue(engine.traffic_mgr.vehicles[
            "npc"].signal_state.low_beam)
        self.assertIn(
            "vw.lowBeamHeadlight.switch('on')",
            result.vehicle_results["npc"].tick_interactions[
                0]["actions_taken"],
        )

    def test_sumo_npc_lamp_rule_does_not_require_an_agent_callback(self):
        scenario = MultiScenario.from_dict({
            "scenario_id": "npc_lamp_rule_without_callback",
            "road_network_id": "beijing_guomao",
            "total_time_s": 0.2,
            "tick_interval_s": 1.0,
            "daynight_keyframes": [{"t": 0, "period": "night"}],
            "vehicles": [{
                "vehicle_id": "npc",
                "initial_node": "n33399858",
                "destination_node": "n35722739",
                "is_evaluated": False,
                "agent_config": {"type": "sumo"},
            }],
        })

        engine = MultiSimEngine(scenario)
        result = engine.run({})

        self.assertEqual(engine._vw["npc"].lowBeamHeadlight.mode.value, "on")
        self.assertTrue(engine._vw["npc"].positionLight.is_on)
        self.assertEqual(len(result.npc_equipment_rule_events), 1)
        self.assertEqual(
            result.npc_equipment_rule_events[0]["actions"],
            [
                "vw.lowBeamHeadlight.switch('on')",
                "vw.positionLight.carcontrol_positionLight_switch(True)",
            ],
        )

    def test_rule_schema_rejects_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown rule fields"):
            Rule.from_dict({
                "id": "invalid",
                "domain": "weather",
                "trigger": {"state_change": {"to": ["rainy"]}},
                "unexpected": True,
            })

    def test_rule_schema_rejects_wrong_domain_trigger(self):
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            Rule.from_dict({
                "id": "invalid",
                "domain": "weather",
                "trigger": {"map_event": {"event_types": ["accident"]}},
            })

    def test_rule_loader_rejects_unknown_document_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "invalid.yaml").write_text(
                "unsupported_rules: []\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown YAML document"):
                RuleLoader(temp_dir)

    def test_every_cabin_yaml_action_targets_a_real_api(self):
        loader = RuleLoader()
        vw = VehicleWorld()
        missing = []
        for rule in [*loader.env_rules, *loader.user_intent_rules]:
            if rule.domain in ("driving", "pedestrian"):
                continue
            actions = [
                *rule.expected_actions,
                *rule.accepted_alternatives,
                rule.expect.get("action_template", ""),
                rule.expect.get("speed_camera_template", ""),
                rule.expect.get("speed_camera_template_no_type", ""),
            ]
            for action in filter(None, actions):
                match = re.match(
                    r"vw\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(",
                    action,
                )
                if not match:
                    missing.append((rule.id, action))
                    continue
                module_name, method_name = match.groups()
                module = getattr(vw, module_name, None)
                if module is None or not callable(
                        getattr(module, method_name, None)):
                    missing.append((rule.id, action))
        self.assertEqual(missing, [])

    def test_every_cabin_rule_has_an_executable_yaml_candidate(self):
        loader = RuleLoader()
        values = {
            "dest": "the airport",
            "waypoint": "the gas station",
            "temp": 23,
            "contact": "mom",
            "category": "pop",
            "event_type": "roadwork",
            "distance": 100,
            "speed_limit": 50,
            "cam_type": "fixed",
            "closest_ahead_speed": 20,
            "density_speed": 20,
        }
        failures = []
        for rule in [*loader.env_rules, *loader.user_intent_rules]:
            if not rule.expected_actions:
                continue
            actions = [
                action.format(**values)
                for action in rule.expected_actions]
            acceptable = []
            if rule.accepted_alternatives:
                acceptable.append(AcceptableAction(
                    primary=actions[-1],
                    alternatives=[
                        action.format(**values)
                        for action in rule.accepted_alternatives
                    ],
                ))

            before = VehicleWorld()
            if rule.id in ("nav_change_dest", "nav_add_waypoint"):
                before.navigation.navigation_route_plan("current destination")
            elif rule.id == "hands_free":
                before.conversation.conversation_phone_call("mom")

            candidate_errors = []
            has_valid_candidate = False
            for candidate in _candidate_action_sets(actions, acceptable):
                _, errors = CabinYamlEvaluator._build_expected_world(
                    before, candidate)
                candidate_errors.extend(errors)
                if not errors:
                    has_valid_candidate = True
            if rule.id == "request_u_turn":
                # A cabin-only VW has no SUMO backend. It must not turn a
                # no-op U-turn into an already-satisfied physical request.
                self.assertFalse(has_valid_candidate)
                self.assertTrue(all("driving_backend_unavailable" in e["error"]
                                    for e in candidate_errors))
                continue
            if not has_valid_candidate:
                failures.append((rule.id, candidate_errors))
        self.assertEqual(failures, [])

    def test_empty_heartbeat_is_not_scored(self):
        before = VehicleWorld()
        after = VehicleWorld()
        result = CabinYamlEvaluator().evaluate(
            "ego", 0.0, before, after, [])
        self.assertIsNone(result)

    def test_any_of_action_is_a_real_executable_alternative(self):
        before = VehicleWorld()
        after = VehicleWorld()
        after.sunroof.carcontrol_sunroof_switch("open")
        result = CabinYamlEvaluator().evaluate(
            vehicle_id="ego",
            time_s=0.0,
            pre_agent_vw=before,
            post_agent_vw=after,
            ground_truth_lines=[
                "vw.window.carcontrol_window_switch("
                "[\"driver's seat\"], True)"
            ],
            acceptable_actions=[AcceptableAction(
                primary=(
                    "vw.window.carcontrol_window_switch("
                    "[\"driver's seat\"], True)"
                ),
                alternatives=[
                    "vw.sunroof.carcontrol_sunroof_switch('open')",
                ],
            )],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(
            result["accepted_actions"],
            ["vw.sunroof.carcontrol_sunroof_switch('open')"])

    def test_unrelated_driving_delta_does_not_dilute_cabin_rule(self):
        before = VehicleWorld()
        after = VehicleWorld()
        after.sunroof.carcontrol_sunroof_switch("open")
        after.navigation.navigation_route_plan("destination")

        result = CabinYamlEvaluator().evaluate(
            vehicle_id="ego",
            time_s=0.0,
            pre_agent_vw=before,
            post_agent_vw=after,
            ground_truth_lines=[
                "vw.sunroof.carcontrol_sunroof_switch('open')"
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertGreater(result["total_fields"], 0)
        self.assertFalse(any(
            detail["field"].startswith("navigation.")
            for detail in result["details"]
        ))

    def test_invalid_yaml_action_is_an_explicit_failure(self):
        before = VehicleWorld()
        after = VehicleWorld()
        result = CabinYamlEvaluator().evaluate(
            "ego", 0.0, before, after,
            ["vw.navigation.navigation_stop()"])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertTrue(result["gt_execution_errors"])


    def test_idempotent_yaml_target_is_counted_as_satisfied(self):
        before = VehicleWorld()
        after = VehicleWorld()
        result = CabinYamlEvaluator().evaluate(
            "ego", 0.0, before, after,
            ["vw.navigation.navigation_exit()"])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["total_fields"], 1)
        self.assertEqual(
            result["details"][0]["field"], "(already_satisfied)")

    def test_unsatisfied_yaml_precondition_is_an_explicit_failure(self):
        before = VehicleWorld()
        after = VehicleWorld()
        result = CabinYamlEvaluator().evaluate(
            "ego", 0.0, before, after,
            [
                "vw.navigation.navigation_destination_change("
                "'the airport')"
            ])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertTrue(result["gt_execution_errors"])


class LayerScoringTest(unittest.TestCase):

    def test_cabin_score_is_0_to_100_and_non_applicable_is_none(self):
        self.assertIsNone(cabin_layer_score_100(None))
        self.assertEqual(cabin_layer_score_100(0.73), 73.0)

    def test_single_vehicle_score_uses_persisted_process_report(self):
        report = {
            "applicable": True,
            "hard_safety_passed": False,
            "task_completed": False,
            "rational_episode_rate": 0.84,
            "metrics": {"rational_episode_count": 4},
            "driving_process": {
                "driving_process_score_100": 84.0,
            },
        }
        self.assertEqual(single_vehicle_layer_score_100(report), 84.0)
        self.assertIsNone(single_vehicle_layer_score_100({
            **report, "driving_process": None,
        }))

class SplitEvaluationIntegrationTest(unittest.TestCase):

    @staticmethod
    def _cabin_vehicle(vehicle_id: str, correct: int, total: int) -> VehicleResult:
        return VehicleResult(
            vehicle_id=vehicle_id,
            is_evaluated=True,
            checkpoints=[{
                "applicable": True,
                "correct_fields": correct,
                "total_fields": total,
            }],
        )

    def test_multillm_top_level_cabin_score_uses_focal_only(self):
        result = MultiSimResult(
            scenario_id="multillm_cabin_scope",
            vehicle_results={
                "ego": self._cabin_vehicle("ego", 4, 5),
                "peer": self._cabin_vehicle("peer", 1, 5),
            },
        )
        result._scenario_config = {
            "experiment_scene": {"focal_vehicle_id": "ego"},
            "vehicles": [
                {"vehicle_id": "ego", "is_evaluated": True,
                 "agent_config": {"type": "llm"}},
                {"vehicle_id": "peer", "is_evaluated": True,
                 "agent_config": {"type": "llm"}},
            ],
        }

        self.assertEqual(result.overall_cabin_score, 0.8)
        self.assertEqual(
            result.to_dict()["evaluation"]["cabin"]["score"], 0.8)

    def test_multillm_focal_none_is_not_filled_by_peer_cabin_score(self):
        result = MultiSimResult(
            scenario_id="multillm_cabin_none_scope",
            vehicle_results={
                "ego": VehicleResult(vehicle_id="ego", is_evaluated=True),
                "peer": self._cabin_vehicle("peer", 1, 1),
            },
        )
        result._scenario_config = {
            "experiment_scene": {"focal_vehicle_id": "ego"},
            "vehicles": [
                {"vehicle_id": "ego", "is_evaluated": True,
                 "agent_config": {"type": "llm"}},
                {"vehicle_id": "peer", "is_evaluated": True,
                 "agent_config": {"type": "llm"}},
            ],
        }

        self.assertIsNone(result.overall_cabin_score)
        self.assertIsNone(result.to_dict()["evaluation"]["cabin"]["score"])

    @staticmethod
    def _scenario(weather: str = "rainy") -> MultiScenario:
        return MultiScenario.from_dict({
            "scenario_id": "split_eval_test",
            "road_network_id": "beijing_zhongguancun",
            "total_time_s": 0.0,
            "tick_interval_s": 1.0,
            "vehicles": [{
                "vehicle_id": "ego",
                "initial_node": "n286387102",
                "destination_node": "n240420386",
                "is_evaluated": True,
                "agent_config": {"type": "llm"},
            }],
            "weather_keyframes": [{
                "t": 0,
                "condition": weather,
                "temperature": 18,
                "humidity": 80,
                "wind_speed": 2,
            }],
            "daynight_keyframes": [{"t": 0, "period": "afternoon"}],
        })

    def test_yaml_weather_rule_passes_and_is_reported_separately(self):
        def correct(vw, *_args, **_kwargs):
            vw.wiper.carcontrol_wiperBlade_switch(True, "front")
            vw.fogLight.carcontrol_fogLight_switch(True, "front")
            vw.broadcast.broadcast_warning("rain_detected")
            return []

        # Cabin scoring does not depend on browser rendering. Stub the
        # optional camera so this unit test remains valid in a network
        # sandbox where the local Web3D capture server cannot bind a port.
        with patch(
                "visualization.web3d_camera.Web3DCameraRenderer.render",
                return_value=None):
            result = MultiSimEngine(self._scenario()).run({"ego": correct})
        vehicle_result = result.vehicle_results["ego"]
        self.assertEqual(vehicle_result.cabin_score, 1.0)
        self.assertIsNotNone(vehicle_result.driving_evaluation)
        serialised = result.to_dict()
        self.assertEqual(serialised["evaluation"]["cabin"]["score"], 1.0)
        self.assertEqual(
            serialised["evaluation"]["cabin"]["layer_score_100"],
            100.0)
        self.assertEqual(
            serialised["evaluation"]["cabin"]["task_completion_rate"],
            1.0)
        self.assertIn(
            "dimension_scores",
            serialised["evaluation"] and
            serialised["vehicles"]["ego"]["driving_evaluation"],
        )
        self.assertIn(
            "single_vehicle_layer_score_100",
            serialised["vehicles"]["ego"]["driving_evaluation"],
        )

    def test_missing_yaml_actions_fail_instead_of_no_change_pass(self):
        result = MultiSimEngine(self._scenario()).run(
            {"ego": lambda *_args, **_kwargs: []})
        vehicle_result = result.vehicle_results["ego"]
        self.assertEqual(vehicle_result.cabin_score, 0.0)
        self.assertEqual(len(vehicle_result.checkpoints), 1)
        self.assertEqual(vehicle_result.cabin_task_completion_rate, 0.0)

    def test_physical_route_plan_without_yaml_request_is_not_scored(self):
        scenario = MultiScenario.from_dict({
            "scenario_id": "route_plan_without_cabin_rule",
            "road_network_id": "beijing_guomao",
            "total_time_s": 0.0,
            "vehicles": [{
                "vehicle_id": "ego",
                "initial_node": "n33399858",
                "destination_node": "n35722739",
                "is_evaluated": True,
                "agent_config": {"type": "llm"},
            }],
        })

        def plan_route(vw, *_args, **_kwargs):
            vw.navigation.navigation_route_plan(
                "n35722739", "n33399858")
            return []

        result = MultiSimEngine(scenario).run({"ego": plan_route})
        vehicle_result = result.vehicle_results["ego"]
        self.assertIsNone(vehicle_result.cabin_score)
        self.assertEqual(vehicle_result.checkpoints, [])
        self.assertFalse(result.to_dict()["vehicles"]["ego"][
            "cabin_evaluation"]["applicable"])

    def test_driving_report_exposes_rational_episode_contract(self):
        result = MultiSimEngine(self._scenario()).run({})
        driving = result.to_dict()["vehicles"]["ego"][
            "driving_evaluation"]
        self.assertIn("reasonable_driving_pass", driving)
        self.assertIn("rational_episode_rate", driving)
        self.assertIn("decision_episodes", driving)


class DrivingEvaluatorTest(unittest.TestCase):

    def test_pedestrian_risk_uses_physical_lifecycle_and_not_ttc_gaps(self):
        vehicle = SimpleNamespace(
            vehicle_id="ego", current_speed_kmh=18.0,
            target_speed_kmh=18.0, acceleration_mps2=0.0,
            pose_x_m=0.0, pose_y_m=0.0, yaw_rad=0.0,
            length_m=4.6, width_m=1.9)
        active = SimpleNamespace(
            is_spawned=True, has_arrived=False, pending_arrival=False,
            is_crashed=False, physical_pose_xy=(1.0, 0.0),
            collision_radius_m=0.4)
        arrived = SimpleNamespace(
            is_spawned=True, has_arrived=True, pending_arrival=False,
            is_crashed=False, physical_pose_xy=(1.0, 0.0),
            collision_radius_m=0.4)
        retired = SimpleNamespace(
            is_spawned=True, has_arrived=False, pending_arrival=False,
            is_crashed=False, physical_pose_xy=(1.0, 0.0),
            collision_radius_m=0.4)
        manager = SimpleNamespace(
            get_state=lambda vehicle_id: vehicle,
            _collision_log=[],
            pedestrians={
                "active": active, "arrived": arrived, "retired": retired,
            },
            _sumo_pedestrian_proxy={"active": "proxy-active"},
        )
        evaluator = DrivingEvaluator(manager, None, ["ego"])
        acc = evaluator._vehicles["ego"]

        def awareness(ttc):
            return {
                "leader": None,
                "connector_conflicts": [],
                "pedestrian_hazards": [
                    {"pedestrian_id": pedestrian_id,
                     "distance_to_crosswalk_m": 2.0,
                     "vehicle_ttc_s": ttc}
                    for pedestrian_id in ("active", "arrived", "retired")
                ],
            }

        evaluator._observe_near_misses(
            acc, vehicle, awareness(1.0), time_s=1.0, dt=0.1)
        # Only the active, physically materialised pedestrian is evidence.
        self.assertEqual(
            [event["hazard"] for event in acc.events],
            ["pedestrian:active"],
        )
        self.assertEqual(acc.events[0]["body_clearance_m"], 0.6)
        self.assertEqual(acc.events[0]["vehicle_speed_kmh"], 18.0)
        first_encounter_id = acc.events[0]["pedestrian_encounter_id"]

        # Stopping clears TTC, but not the physical encounter lifecycle.
        vehicle.current_speed_kmh = 0.0
        evaluator._observe_near_misses(
            acc, vehicle, awareness(float("inf")), time_s=3.0, dt=2.0)
        vehicle.current_speed_kmh = 18.0
        evaluator._observe_near_misses(
            acc, vehicle, awareness(1.0), time_s=9.5, dt=6.5)
        self.assertEqual(len(acc.events), 1)

        # Once the person arrives, even a stale awareness record is rejected.
        active.has_arrived = True
        manager._sumo_pedestrian_proxy.clear()
        evaluator._observe_near_misses(
            acc, vehicle, awareness(1.0), time_s=9.9, dt=0.4)
        self.assertNotIn(
            "pedestrian:active", acc.recorded_pedestrian_near_miss_keys)

        # Reusing an entity ID for a later physical appearance is a genuinely
        # new encounter, not a global per-pedestrian cap.
        active.has_arrived = False
        manager._sumo_pedestrian_proxy["active"] = "proxy-active-2"
        evaluator._observe_near_misses(
            acc, vehicle, awareness(1.0), time_s=20.0, dt=10.1)
        self.assertEqual(len(acc.events), 2)
        self.assertNotEqual(
            first_encounter_id, acc.events[1]["pedestrian_encounter_id"])

    def test_pedestrian_near_miss_requires_moving_body_proximity(self):
        vehicle = SimpleNamespace(
            vehicle_id="ego", current_speed_kmh=30.0,
            target_speed_kmh=30.0, acceleration_mps2=0.0,
            pose_x_m=0.0, pose_y_m=0.0, yaw_rad=0.0,
            length_m=4.6, width_m=1.9)
        pedestrian = SimpleNamespace(
            is_spawned=True, has_arrived=False, pending_arrival=False,
            is_crashed=False, physical_pose_xy=(16.0, 0.0),
            collision_radius_m=0.4)
        manager = SimpleNamespace(
            pedestrians={"ped": pedestrian},
            _sumo_pedestrian_proxy={"ped": "proxy-ped"},
        )
        evaluator = DrivingEvaluator(manager, None, ["ego"])
        acc = evaluator._vehicles["ego"]
        awareness = {
            "leader": None, "connector_conflicts": [],
            "pedestrian_hazards": [{
                "pedestrian_id": "ped",
                "distance_to_crosswalk_m": 15.39,
                "vehicle_ttc_s": 1.85,
            }],
        }

        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=2.6, dt=0.1)
        self.assertEqual(acc.events, [])
        self.assertIn("pedestrian:ped", acc.active_risk_episodes)

        pedestrian.physical_pose_xy = None
        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=2.65, dt=0.05)
        self.assertEqual(acc.events, [])

        # The SUMO pose is the front bumper, so 2.5 m ahead leaves 2.1 m
        # after the pedestrian radius. A centred rectangle would mis-score it.
        pedestrian.physical_pose_xy = (2.5, 0.0)
        self.assertAlmostEqual(
            evaluator._pedestrian_body_clearance_m(vehicle, "ped"), 2.1)
        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=2.7, dt=0.1)
        self.assertEqual(acc.events, [])

        pedestrian.physical_pose_xy = (1.5, 0.0)
        vehicle.current_speed_kmh = 0.0
        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=2.8, dt=0.1)
        self.assertEqual(acc.events, [])

        vehicle.current_speed_kmh = 10.0
        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=2.9, dt=0.1)
        self.assertEqual(len(acc.events), 1)
        self.assertEqual(acc.events[0]["body_clearance_m"], 1.1)
        self.assertEqual(acc.events[0]["vehicle_speed_kmh"], 10.0)
        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=3.0, dt=0.1)
        self.assertEqual(len(acc.events), 1)

    def test_route_progress_keeps_furthest_point_after_reroute(self):
        vehicle = SimpleNamespace(
            arrived=False,
            lane_route_actions=[],
            lane_route_action_index=0,
            edge_progress=0.0,
        )
        manager = SimpleNamespace(
            get_state=lambda vehicle_id: vehicle,
            _collision_log=[],
        )
        evaluator = DrivingEvaluator(manager, None, ["ego"])
        acc = evaluator._vehicles["ego"]
        acc.initial_remaining_distance_m = 100.0
        acc.last_remaining_distance_m = 20.0
        acc.max_route_progress = 0.8

        # A legal reroute can temporarily increase remaining distance.  The
        # final score must retain the furthest physical progress already made.
        acc.last_remaining_distance_m = 130.0

        self.assertEqual(evaluator._route_progress(acc, vehicle), 0.8)

    def test_startup_closing_leader_response_gets_a_scored_episode(self):
        vehicle = SimpleNamespace(
            vehicle_id="ego", current_speed_kmh=38.0,
            target_speed_kmh=38.0, acceleration_mps2=0.0)
        manager = SimpleNamespace(
            get_state=lambda vehicle_id: vehicle,
            _collision_log=[],
        )
        evaluator = DrivingEvaluator(manager, None, ["ego"])
        acc = evaluator._vehicles["ego"]
        acc.first_time_s = 0.0
        awareness = {
            "leader": {
                "vehicle_id": "lead", "gap_m": 23.4,
                "speed_kmh": 22.0,
            },
            "connector_conflicts": [],
            "pedestrian_hazards": [],
        }

        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=0.0, dt=0.0)
        self.assertEqual(acc.decision_episodes, [])
        self.assertIn("leader:lead", acc.active_risk_episodes)

        vehicle.current_speed_kmh = 37.96
        vehicle.target_speed_kmh = 30.0
        vehicle.acceleration_mps2 = -0.4
        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=0.1, dt=0.1)

        self.assertEqual(len(acc.decision_episodes), 1)
        episode = acc.decision_episodes[0]
        self.assertEqual(episode["type"], "leader_response")
        self.assertTrue(episode["reasonable"])
        self.assertEqual(
            episode["details"]["trigger"], "startup_closing_leader")
        self.assertGreater(episode["details"]["entry_ttc_s"], 4.5)

    def test_proactive_leader_response_is_scored_before_near_miss(self):
        vehicle = SimpleNamespace(
            vehicle_id="ego", current_speed_kmh=50.0,
            target_speed_kmh=50.0, acceleration_mps2=0.0)
        manager = SimpleNamespace(
            get_state=lambda vehicle_id: vehicle,
            _collision_log=[],
        )
        evaluator = DrivingEvaluator(manager, None, ["ego"])
        acc = evaluator._vehicles["ego"]
        awareness = {
            "leader": {
                "vehicle_id": "lead", "gap_m": 24.0,
                "speed_kmh": 30.0,
            },
            "connector_conflicts": [],
            "pedestrian_hazards": [],
        }

        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=1.0, dt=0.1)
        self.assertEqual(acc.near_miss_episodes, 0)
        self.assertEqual(acc.decision_episodes, [])

        vehicle.current_speed_kmh = 49.0
        vehicle.target_speed_kmh = 30.0
        vehicle.acceleration_mps2 = -1.0
        evaluator._observe_near_misses(
            acc, vehicle, awareness, time_s=1.3, dt=0.1)

        self.assertEqual(acc.near_miss_episodes, 0)
        self.assertEqual(len(acc.decision_episodes), 1)
        self.assertEqual(
            acc.decision_episodes[0]["type"], "leader_response")
        self.assertTrue(acc.decision_episodes[0]["reasonable"])
        self.assertEqual(
            acc.decision_episodes[0]["reason"], "timely_risk_response")

    def test_collision_is_a_hard_failure(self):
        vehicle = VehicleState(
            current_node="A",
            vehicle_id="ego",
            distance_traveled_m=10.0,
        )
        collision = SimpleNamespace(
            entity_a="ego",
            entity_b="other",
            entity_a_type="vehicle",
            entity_b_type="vehicle",
            location="lane",
            collision_type="rear_end",
            time_s=1.0,
        )
        manager = SimpleNamespace(
            get_state=lambda vehicle_id: (
                vehicle if vehicle_id == "ego" else None),
            _collision_log=[collision],
        )
        evaluator = DrivingEvaluator(manager, None, ["ego"])
        evaluator._vehicles["ego"].sample_count = 1
        report = evaluator.finalize()["ego"]
        self.assertFalse(report["hard_safety_passed"])
        self.assertEqual(
            report["dimension_scores"]["safety"], 0.0)
        self.assertEqual(report["metrics"]["collision_count"], 1)
        process = calculate_driving_process_score(report)
        self.assertTrue(process["hard_gate_triggered"])
        self.assertEqual(process["driving_process_score_100"], 0.0)

    def test_sumo_collision_victim_is_not_penalized(self):
        # basic_038 trace, 42.5 s: SUMO collider map_flow_03 -> victim ego.
        # basic_111 trace, 62.7 s: SUMO collider map_cross_01 -> victim ego.
        for collider, kind, time_s, location in (
                ("map_flow_03", "sumo_collision", 42.5,
                 "n5623750808_n5623750819"),
                ("map_cross_01", "sumo_junction", 62.7,
                 "connector::n472382793::9")):
            collision = CollisionEvent(
                entity_a=collider, entity_b="ego", location=location,
                collision_type=kind, time_s=time_s, physics_source="sumo")
            vehicles = {
                vid: VehicleState(current_node="A", vehicle_id=vid)
                for vid in ("ego", collider)
            }
            manager = SimpleNamespace(
                get_state=vehicles.get, _collision_log=[collision, collision])
            evaluator = DrivingEvaluator(manager, None, list(vehicles))
            for acc in evaluator._vehicles.values():
                acc.sample_count = 1
            reports = evaluator.finalize()
            for vid, at_fault in (("ego", False), (collider, True)):
                with self.subTest(collision_type=kind, vehicle=vid):
                    report = reports[vid]
                    self.assertEqual(report["metrics"]["collision_count"], 1)
                    self.assertEqual(
                        report["metrics"]["at_fault_collision_count"], int(at_fault))
                    self.assertEqual(report["hard_safety_passed"], not at_fault)
                    self.assertEqual(
                        report["dimension_scores"]["safety"], 0.0 if at_fault else 1.0)
                    self.assertEqual(len(report["hard_violations"]), int(at_fault))
                    self.assertEqual(len(report["decision_episodes"]), int(at_fault))
                    process = calculate_driving_process_score(report)
                    self.assertEqual(process["hard_gate_triggered"], at_fault)
                    self.assertEqual(
                        process["driving_process_score_100"], 0.0 if at_fault else 100.0)

    def test_unattributed_and_pedestrian_contacts_keep_collision_penalty(self):
        vehicle = VehicleState(current_node="A", vehicle_id="ego")
        for collision_type, other_type, source in (
                ("rear_end", "vehicle", "sumo"),
                ("sumo_collision", "vehicle", "unknown"),
                ("vehicle_pedestrian", "pedestrian", "sumo")):
            with self.subTest(collision_type=collision_type, source=source):
                collision = CollisionEvent(
                    entity_a="other", entity_b="ego",
                    entity_a_type=other_type, collision_type=collision_type,
                    physics_source=source, time_s=1.0)
                manager = SimpleNamespace(
                    get_state=lambda vid: vehicle, _collision_log=[collision])
                evaluator = DrivingEvaluator(manager, None, ["ego"])
                evaluator._vehicles["ego"].sample_count = 1
                report = evaluator.finalize()["ego"]
                self.assertEqual(report["metrics"]["at_fault_collision_count"], 1)
                self.assertTrue(
                    calculate_driving_process_score(report)["hard_gate_triggered"])


if __name__ == "__main__":
    unittest.main()
