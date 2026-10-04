"""Seeded, initialization-only SUMO driver diversity (never actuator scripts)."""
import copy
import hashlib
import random

SCHEMA = "native-npc-mixed-v1"
# Conservative first-pass ranges, not a calibrated human-driver population.
# All speed factors respect posted limits; differential speeds still encourage
# native overtaking. No relaxed collision checks or reduced braking capability.
RANGES = {
    "cautious": ((0.75, 0.9), (1.4, 1.9), (2.5, 3.5), (0.2, 0.4), (0.3, 0.7), (0.8, 1.0)),
    "ordinary": ((0.88, 1.0), (1.0, 1.5), (2.0, 3.0), (0.3, 0.5), (0.7, 1.2), (0.5, 0.9)),
    "urgent": ((0.97, 1.0), (0.8, 1.1), (1.5, 2.2), (0.2, 0.4), (1.3, 2.0), (0.1, 0.4)),
}
PARAMETERS = ("speedFactor", "tau", "minGap", "sigma", "lcSpeedGain", "lcCooperative")


def validate_behavior(config):
    if not isinstance(config, dict):
        raise ValueError("npc_behavior must be an object")
    if not config:
        return
    if set(config) != {"schema", "seed", "vehicle_ids"} or config["schema"] != SCHEMA:
        raise ValueError("invalid npc_behavior schema or fields")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**31:
        raise ValueError("npc_behavior seed must be an integer in [0, 2**31)")
    ids = config["vehicle_ids"]
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(v, str) or not v for v in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("npc_behavior vehicle_ids must be unique nonempty strings")


def sample_driver(seed, vehicle_id):
    digest = hashlib.sha256(f"{SCHEMA}:{seed}:{vehicle_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    profile = rng.choices(list(RANGES), weights=(0.25, 0.5, 0.25))[0]
    return {"profile": profile, **{
        name: round(rng.uniform(*bounds), 6)
        for name, bounds in zip(PARAMETERS, RANGES[profile])}}


def with_native_npc_behavior(scenario, seed):
    """Copy an authored scene BEFORE reference-agent substitution.

    Explicit actor IDs keep focal/peer reference drivers out of the treatment.
    Caller must recalibrate the returned scene before using it for evaluation.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    result = copy.deepcopy(scenario)
    ids = sorted(v["vehicle_id"] for v in result["vehicles"]
                 if v.get("agent_config", {}).get("type", "sumo") == "sumo"
                 and not v.get("initial_physical_state", {}).get("is_crashed", False))
    if not ids:
        raise ValueError("scenario has no native NPC vehicles")
    result.setdefault("sumo_config", {})["npc_behavior"] = {
        "schema": SCHEMA, "seed": seed, "vehicle_ids": ids}
    result.pop("time_window_calibration", None)
    return result
