# Evaluation protocol

[中文实验管线](../../vehiclearena/evaluation/experiments/README.md) · [English index](README.md)

## Task sets

The published benchmark contains 100 training/development tasks and 112 held-out evaluation tasks: 80 Basic tasks with one evaluated LLM driver and 32 MultiLLM tasks with fixed LLM peers. Other vehicles and pedestrians use native SUMO traffic behavior. Scene files contain `scenario.json` for executable initial conditions and `expected.json` for machine-checkable setup conditions. The test selection is validated by `scripts/run_test_suite.py`.

## Why run a SUMO reference?

The same scene may be hard even without an LLM driver. To separate that background difficulty from effects of the evaluated driver, VehicleArena runs a matched reference in which SUMO controls that driver. In Basic, the reference is all-SUMO. In MultiLLM, only the evaluated vehicle changes controller; the other LLM drivers retain their models, personality prompts, and passenger-agent/Judge settings. The scene, map, initial conditions, scheduled events, and native background-traffic configuration are matched. Their resulting trajectories are allowed to diverge because road users react to one another.

Separate calibration rollouts set the frozen time window. The matched reference run measures traffic effects; it is not another evaluated model. The reference runner is `scripts/run_reference_suite.py`; it can reuse a treatment suite's manifests and role configuration through `--source-output`.

## Frozen time windows

The registry `vehiclearena/evaluation/experiments/time_window_calibration.json` stores the fixed `total_time_s` used by evaluation. For all 112 published test tasks, the saved deadline is the last required SUMO trip vehicle's terminal time plus 10 *simulation* seconds. Terminal means arrival or collision; none of the selected calibration runs records a collision. Basic calibration is all-SUMO. MultiLLM calibration keeps fixed LLM peers and their passenger agents; it does not wait for those peers to finish. Declared persistent obstacles are excluded from the required trip-vehicle set.

The code that *recalibrates* time windows, `vehiclearena/evaluation/experiments/time_window_calibration.py`, additionally takes the maximum of the SUMO terminal time and the last included scheduled event before adding 10 seconds. Thirteen frozen test tasks would receive a different deadline if recalibrated with this formula. Published results therefore use the saved registry, not newly generated windows. These deadlines are unrelated to wall-clock worker timeouts.

## Independent metrics

VehicleArena does not combine the following into a single compensating score:

| Dimension | Interpretation |
|---|---|
| Cabin device score | Correctness of applicable equipment-rule outcomes; not applicable remains missing. |
| Passenger request score | Persisted PA/Judge grade for the evaluated vehicle; Judge failures remain unscored. |
| Driving score | Evaluated vehicle's process-rule score from recorded behavior; at-fault collision or red-light entry triggers a zero-score gate. |
| Arrival success | The evaluated vehicle arrives within the frozen window without collision or route failure. |
| Net NPC queue-time change | Signed treatment minus reference queue exposure, in vehicle-seconds. |
| Net NPC completion-time change | Signed treatment minus reference completion time for the defined interacting NPC set. |
| NPC collision-count change | Treatment minus reference non-evaluated NPC collisions. |
| Input tokens | Actual prompt/input tokens consumed. |
| Output tokens | Actual completion/output tokens consumed. |

Positive traffic-time changes indicate added delay; negative values indicate improvement. Keep NPC non-arrivals and MultiLLM interaction validity as separate diagnostics. Paired traffic metrics require matching `variant_id` and a valid `protocol_hash`; do not replace missing or invalid reference values with zero. Infrastructure-invalid episodes do not enter model averages.

The current driving-score implementation is in [`driving_process_score.py`](../../vehiclearena/evaluation/driving_process_score.py).
