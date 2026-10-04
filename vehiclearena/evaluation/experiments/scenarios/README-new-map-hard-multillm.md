# New-map hard MultiLLM extension

This historical extension added 20 MultiLLM tasks to the earlier 200-task catalog, producing the current 220-task catalog (180 Basic and 40 MultiLLM). The added tasks were generated directly from lane-level authors on new road maps. They use unsignalized four-way, unprotected left-turn, and four-agent merge geometries, with related native SUMO traffic coupled to the evaluated vehicle's route. No scene was copied from the old MultiLLM map set.

Current time-window sources and exceptions are recorded in `../time_window_calibration.json`; the fixed evaluation split contains 80 Basic and 32 MultiLLM tasks.
