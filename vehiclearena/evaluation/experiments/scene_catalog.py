"""Topology-backed task catalog for Basic and Multi-LLM experiments.

The catalog separates a physical interaction scene from the factor matrix
applied to it.  Every generated scene pins entities to stable HD-map lane,
connector, or crosswalk IDs and ships machine-readable acceptance criteria.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from simulation.lane_level_runtime import (
    LaneGeometryRuntime, _polyline_intersection_distances,
    _polyline_nearest_distances, pose_at,
)
from evaluation.experiments.experiment_parameters import (
    PARAMETER_SCHEMA, load_experiment_parameters, parameter_fingerprint,
)
from module.daynight import DayNight
from evaluation.experiments.task_compatibility import (
    adapt_scene, crosswalk_signal_relationship,
)
from evaluation.experiments.environment_design import (
    DAYNIGHT_EDGES,
    DAYNIGHT_DESCRIPTION,
    DAYNIGHT_PATTERNS,
    DAYNIGHT_PERIODS,
    WEATHER_CONDITIONS,
    WEATHER_PATTERNS,
    WEATHER_TARGET_CONDITIONS,
    WEATHER_TRANSITION_EDGES,
    climate_month,
    daynight_sequence,
    location_weather_profile,
    load_location_weather_profiles,
    select_weather_primary,
    weather_parameters,
    weather_sequence,
)
from evaluation.experiments.required_turn_sites import SITES as REQUIRED_TURN_SITES


CATALOG_SCHEMA = "vehiclearena-experiment-scenes-v2"
RETIRED_PASSING_TEMPLATES = frozenset({
    "baseline_lane_change", "chassis_target_lane_gap",
    "traffic_lane_change_pressure", "chassis_narrow_road",
})
# These templates remain available to infrastructure smoke tests, but no
# longer count as formal benchmark tasks.  Their frozen study instances are
# replaced below by interaction-local scenes with machine-checkable geometry.
RETIRED_BASIC_STUDY_TEMPLATES = frozenset({
    "cabin_available_feature_task",
    "cabin_capability_probe",
    "cabin_common_climate_task",
    "concurrent_crosswalk_dual_task",
    "concurrent_intersection_dual_task",
    "pedestrian_signalized_crosswalk",
    "pedestrian_turning_crosswalk",
    "pedestrian_unmarked_jaywalk",
    "pedestrian_unsignalized_crossing",
    "traffic_following_pressure",
    "traffic_intersection_conflict",
    "traffic_merge_gap",
})
RETIRED_MULTI_STUDY_TEMPLATES = frozenset({
    "multi_crash_obstacle_bypass",
    "multi_signal_queue_start",
    "multi_two_lane_merge",
    "multi_unsignalized_four_way",
    "multi_whole_map_traffic",
})

# NPC traffic-impact scoring needs non-focal trip vehicles whose routes can
# actually be affected by the focal driver. Every formal task retains the
# original deterministic floor of three route-aligned actors and two
# connector-conflict actors. It additionally receives a native-SUMO cohort
# that starts beside or behind the focal vehicle and shares its downstream
# destination. The causal-chain extension allows up to eight cohort vehicles;
# it remains a frozen physical input and uses no runtime policy or scripted
# speed intervention.
RELATED_BACKGROUND_BASE_MINIMUM = 5
LONGITUDINAL_COHORT_MINIMUM = 3
LONGITUDINAL_COHORT_BASE_MAXIMUM = 5
CAUSAL_CHAIN_EXTRA_VEHICLE_COUNT = 3
LONGITUDINAL_COHORT_MAXIMUM = (
    LONGITUDINAL_COHORT_BASE_MAXIMUM + CAUSAL_CHAIN_EXTRA_VEHICLE_COUNT)
LONGITUDINAL_COHORT_CLEARANCE_M = 11.0
# Most base cohort vehicles are within 48 m on the focal start segment. A
# compact pedestrian approach may use the first 4/16 m of the post-crossing
# lane; this 72 m ceiling keeps the base cohort close to the focal route.
LONGITUDINAL_COHORT_MAX_OFFSET_M = 72.0
# The optional frozen causal-chain extension records its own 180 m ceiling in
# the scenario metadata because its three rear vehicles use -24/-36/-48 m
# offsets on the same start lane.
CAUSAL_CHAIN_MAX_OFFSET_M = 180.0
POST_CROSSING_ONLY_COHORT_SCENES = frozenset({
    # Dense followers change the pedestrian release/arrival timing enough for
    # the SUMO reference ego to overlap the authored pedestrian.  Keep these
    # cohorts close but downstream of the crossing instead.
    "basic_097_crosswalk__guangzhou_tianhe",
    "basic_161_crosswalk__expansion__beijing_guomao__site_01",
})
ADJACENT_FIRST_COHORT_SCENES = frozenset({
    # Followers in ego's lane alter the deliberately tight merge arrival
    # order and make the SUMO reference collide with stream_2.  The adjacent
    # through lane has the same long downstream corridor and keeps the
    # original merge contract intact.
    "basic_240_merge_stream__expansion__xiamen_huli",
})
LONGITUDINAL_COHORT_TARGET_OVERRIDES = {
    "basic_240_merge_stream__expansion__xiamen_huli": 3,
}
RELATED_BACKGROUND_MINIMUM = (
    RELATED_BACKGROUND_BASE_MINIMUM + LONGITUDINAL_COHORT_MINIMUM)
RELATED_BACKGROUND_ROUTE_TARGET = 3
RELATED_BACKGROUND_CROSSING_TARGET = 2
RELATED_BACKGROUND_CLEARANCE_M = 14.0

# Rebuilt Basic tasks occupy new IDs so historical result directories can
# never be mistaken for the new physical contracts.  The 28 replacements,
# together with the 72 retained tasks, form the original 100-task pool; the
# separately declared expansion below extends it to 180.
BASIC_REBUILT_SCENES = (
    *(("oncoming_stream", network) for network in (
        "nanjing_xinjiekou", "qingdao_shinan", "chengdu_jinjiang",
        "amsterdam_centrum")),
    *(("unprotected_left_turn", network) for network in (
        "beijing_guomao", "guangzhou_tianhe", "wuhan_hankou",
        "shanghai_lujiazui")),
    *(("obstacle_gap_change", network) for network in (
        "wuxi_taihu", "guangzhou_tianhe", "chengdu_chunxi",
        "nanjing_hexi")),
    *(("crowded_unsignalized_crosswalk", network) for network in (
        "beijing_guomao", "guangzhou_tianhe", "wuhan_hankou",
        "amsterdam_centrum")),
    *(("platoon_pressure", network) for network in (
        "beijing_guomao", "hongkong_central", "shanghai_lujiazui",
        "tokyo_shinjuku")),
    *(("crossing_stream", network) for network in (
        "hangzhou_xihu", "suzhou_guanqian", "nanjing_xinjiekou",
        "qingdao_shinan")),
    *(("merge_stream", network) for network in (
        "chengdu_jinjiang", "amsterdam_centrum", "wuhan_hankou",
        "xian_zhonglou")),
)

# The 80-scene Basic expansion is sized together with the original 100 tasks,
# not as an independent convenience batch.  In the final 180-task pool the
# three most fundamental high-risk capabilities below have 18 scenes each and
# every other capability has 9.  This admits an exact 100:80 stratified split:
# 10:8 for the three weighted capabilities and 5:4 for every other one.
BASIC_WEIGHTED_CAPABILITIES = frozenset({
    "baseline_crosswalk",
    "baseline_unsignalized_intersection",
    "baseline_required_turn",
})

BASIC_CROSSWALK_EXPANSION_SITES = (
    ("beijing_guomao", "connector::n36704944::0"),
    ("hongkong_central", "connector::n1685959562::0"),
    ("hongkong_central", "connector::n1685959562::3"),
    ("shanghai_lujiazui", "connector::n495644095::3"),
    ("shanghai_lujiazui", "connector::n495644095::6"),
    ("tokyo_shinjuku", "connector::n475770654::0"),
    ("tokyo_shinjuku", "connector::n475770654::6"),
    ("wuhan_hankou", "connector::n1419140537::3"),
    ("amsterdam_centrum", "connector::n9844170832::0"),
    ("newyork_manhattan_mid", "connector::n9140654137::2"),
)
BASIC_CROWDED_CROSSWALK_EXPANSION_SITES = (
    ("guangzhou_tianhe", "connector::n320769078::1"),
    ("hongkong_central", "connector::n1685959562::7"),
    ("guangzhou_tianhe", "connector::n320769078::2"),
    ("guangzhou_tianhe", "connector::n320769078::3"),
    ("amsterdam_centrum", "connector::n9844170832::1"),
)
BASIC_UNSIGNALIZED_EXPANSION_SITES = (
    ("beijing_sanlitun", "connector::n35551157::3", "connector::n35551157::9"),
    ("berlin_mitte", "connector::n3378459427::0", "connector::n3378459427::3"),
    ("cairo_downtown", "connector::n315745059::0", "connector::n315745059::7"),
    ("chicago_loop", "connector::n2779938525::0", "connector::n2779938525::3"),
    ("dubai_downtown", "connector::n1739389250::4", "connector::n1739389250::8"),
    ("hanoi_hoankiem", "connector::n98011280::3", "connector::n98011280::9"),
    ("london_westend", "connector::n108236::0", "connector::n108236::8"),
    ("madrid_centro", "connector::n26066642::7", "connector::n26066642::17"),
    ("beijing_xidan", "connector::n9377983844::11", "connector::n9377983844::18"),
    ("singapore_orchard", "connector::n1834285286::3", "connector::n1834285286::6"),
)
BASIC_UNSIGNALIZED_EXPANSION_ENVIRONMENT_BASIS_S = {
    "beijing_xidan": 33.8,
}
BASIC_REQUIRED_TURN_EXPANSION_SITES = (
    ("beijing_tiananmen", "n2424906811_n9033666117::lane_1", "connector::n9033666117::5"),
    ("beijing_wangjing", "n12323738788_n1667135540::lane_1", "connector::n12323738788::1"),
    ("beijing_wudaokou", "n1360093876_n292591259::lane_1", "connector::n292591259::6"),
    ("beijing_yizhuang", "n2461496819_n2461496820::lane_2", "connector::intersection::n2115201358::5"),
    ("guangzhou_panyu", "n4547623530_n4547665925::lane_2", "connector::n4547623530::6"),
    ("hangzhou_binjiang", "n1286462578_n314160089::lane_2", "connector::intersection::n1667167449::5"),
    ("hefei_shushan", "n5476950005_n5476950007::lane_2", "connector::n5476950007::4"),
    ("hefei_zhengwu", "n2684741366_n8597500760::lane_1", "connector::n8597500760::1"),
    ("hohhot_saihan", "n11514404039_n3988926359::lane_1", "connector::n3988926359::3"),
    ("jinan_quancheng", "n355817266_n356093878::lane_3", "connector::intersection::n355193176::16"),
)
BASIC_COMPACT_TURN_EXPANSION_SITES = (
    ("dalian_zhongshan", "n2273551381_n2274080000::lane_1", (
        "connector::n2274080000::2",
        "connector::n2283350018::1",
        "connector::n5453733001::1",
    ), 70.0),
)
BASIC_FULL_NETWORK_EXPANSION_SITES = (
    ("haikou_longhua", "n11507266982_n6008753714::lane_0", 272.1, 302.1),
    ("lasa_chengguan", "n2244703998_n5055683362::lane_1", 402.4, 432.4),
    ("losangeles_downtown", "n2163516474_n671847453::lane_2", 201.5, 231.5),
    ("nantong_chongchuan", "n7007460547_n7007460715::lane_0", 302.3, 332.3),
    ("shenzhen_nanshan", "n2400176757_n2400181200::lane_0", 167.0, 197.0),
)

_BASIC_EXPANSION_SIMPLE_MAPS = {
    "baseline_signalized_intersection": ("bangkok_silom",),
    "baseline_straight_following": (
        "changchun_chaoyang", "changsha_furong", "changsha_wuyi",
        "changzhou_tianning", "chengdu_gaoxin",
    ),
    "chassis_lead_vehicle_braking": ("dongguan_nancheng",),
    "chassis_narrow_road": ("helsinki_keskusta",),
    "chassis_red_light_stop": ("jakarta_central",),
    "traffic_oncoming_stream": (
        "kualalumpur_bukit", "kunming_cuihu", "melbourne_cbd",
        "rome_centro", "sydney_cbd",
    ),
    "traffic_unprotected_left_turn": (
        "foshan_chancheng", "fuzhou_wuyi", "harbin_zhongyang",
        "istanbul_beyoglu", "paris_champs",
    ),
    "traffic_obstacle_gap_change": (
        "guiyang_guanshanhu", "jiaxing_nanhu", "lanzhou_chengguan",
        "nanchang_honggutan", "ningbo_tianyi",
    ),
    "traffic_platoon_pressure": (
        "osaka_namba", "qingdao_wusi", "seoul_gangnam",
        "shanghai_hongkou", "shanghai_pudong_zhangjiang",
    ),
    "traffic_crossing_stream": (
        "shenyang_heping", "shenyang_zhongjie", "shenzhen_futian",
        "stockholm_norrmalm", "taipei_xinyi",
    ),
    "traffic_merge_stream": (
        "taiyuan_yingze", "tianjin_heping", "toronto_downtown",
        "warsaw_srodmiescie", "xiamen_huli",
    ),
}

# Every item is (capability/template id, map id, site-specific parameters).
# Explicit lane/connector IDs freeze the geometry-sensitive sites; a null
# payload is safe only for selectors whose acceptance contract fully defines
# the chosen topology.
BASIC_EXPANSION_SCENES = (
    *(("baseline_crosswalk", network, {"connector_id": connector_id,
       "site_ordinal": ordinal})
      for ordinal, (network, connector_id) in enumerate(
          BASIC_CROSSWALK_EXPANSION_SITES, 1)),
    *((kind, network, None)
      for kind in ("baseline_signalized_intersection",
                   "baseline_straight_following")
      for network in _BASIC_EXPANSION_SIMPLE_MAPS[kind]),
    *(("baseline_unsignalized_intersection", network, {
        "ego_connector_id": ego_connector_id,
        "cross_connector_id": cross_connector_id,
        "environment_basis_s":
            BASIC_UNSIGNALIZED_EXPANSION_ENVIRONMENT_BASIS_S.get(
                network, 45.0),
      }) for network, ego_connector_id, cross_connector_id
      in BASIC_UNSIGNALIZED_EXPANSION_SITES),
    *(("chassis_continuous_turns", network, {
        "initial_lane_id": initial_lane_id,
        "connector_ids": connector_ids,
        "environment_basis_s": environment_basis_s,
      }) for network, initial_lane_id, connector_ids, environment_basis_s
      in BASIC_COMPACT_TURN_EXPANSION_SITES),
    *(("chassis_full_network_route", network, {
        "lane_id": lane_id, "progresses": (0.20, 0.45, 0.70),
        "release_speed_kmh": 26.0,
        "environment_basis_s": environment_basis_s,
        "duration_s": duration_s,
      }) for network, lane_id, environment_basis_s, duration_s
      in BASIC_FULL_NETWORK_EXPANSION_SITES),
    *((kind, network, None)
      for kind in ("chassis_lead_vehicle_braking",
                   "chassis_narrow_road", "chassis_red_light_stop")
      for network in _BASIC_EXPANSION_SIMPLE_MAPS[kind]),
    *(("baseline_required_turn", network, {
        "initial_lane_id": initial_lane_id,
        "connector_id": connector_id,
      }) for network, initial_lane_id, connector_id
      in BASIC_REQUIRED_TURN_EXPANSION_SITES),
    *((kind, network, None)
      for kind in ("traffic_oncoming_stream",
                   "traffic_unprotected_left_turn",
                   "traffic_obstacle_gap_change")
      for network in _BASIC_EXPANSION_SIMPLE_MAPS[kind]),
    *(("traffic_crowded_unsignalized_crosswalk", network, {
        "connector_id": connector_id, "site_ordinal": ordinal,
      }) for ordinal, (network, connector_id) in enumerate(
          BASIC_CROWDED_CROSSWALK_EXPANSION_SITES, 1)),
    *((kind, network, None)
      for kind in ("traffic_platoon_pressure", "traffic_crossing_stream",
                   "traffic_merge_stream")
      for network in _BASIC_EXPANSION_SIMPLE_MAPS[kind]),
)
if len(BASIC_EXPANSION_SCENES) != 80:
    raise AssertionError("the frozen Basic expansion must contain 80 scenes")

MIDDLE_SIGNAL_SITES = (
    ("beijing_guomao", "connector::n3564848114::3"),
    ("shanghai_lujiazui", "connector::intersection::n554691049::3"),
    ("tokyo_shinjuku", "connector::n472382793::0"),
    ("wuhan_hankou", "connector::intersection::n1880421483::3"),
)
# New junctions, not the four legacy two-car conflict baselines. The last
# number is the measured dry/noon ego completion rounded down to whole seconds;
# environmental events use this basis, not the more generous task deadline.
UNSIGNALIZED_CHALLENGE_SITES = (
    ("guangzhou_tianhe", "connector::n3017910562::12", "connector::n3017910562::8", 15),
    ("wuhan_hankou", "connector::n1880477884::0", "connector::n1880477884::3", 43),
    ("chengdu_chunxi", "connector::n3562028073::4", "connector::n3562028073::0", 32),
    ("beijing_guomao", "connector::n2207734183::12", "connector::n2207734183::4", 29),
)
COMPACT_TURN_SITES = (
    ("hangzhou_xihu", "n5290871899_n5290871904::lane_0",
     ("connector::n5290871899::4", "connector::n5290871902::1", "connector::n5290871901::3"), 48),
    ("suzhou_guanqian", "n3721939227_n3721939229::lane_1",
     ("connector::n3721939229::3", "connector::n5398296841::1", "connector::n4403743519::7"), 32),
    ("nanjing_xinjiekou", "n2327838702_n4405415756::lane_1",
     ("connector::n4405415756::4", "connector::n1038851730::13", "connector::n2519403621::1"), 42),
    ("qingdao_shinan", "n5614586346_n5614586347::lane_0",
     ("connector::n5614586346::2", "connector::n5614586345::6", "connector::n1294924658::4"), 44),
)
# Four additional long, multi-lane road segments for the same persistent
# lead-vehicle braking interaction used by Basic 041--044.  The measured
# dry/noon NPC completion time is also the environmental schedule basis; the
# final value is the initial publication deadline (completion plus 10 s).
LEAD_BRAKING_EXTRA_SITES = (
    ("chengdu_jinjiang", "n5213238476_n7334966710::lane_1", 43.7, 53.7),
    ("guangzhou_tianhe", "n4397029810_n5200099117::lane_3", 42.8, 52.8),
    ("chengdu_chunxi", "n13168697680_n4903112572::lane_3", 36.0, 46.0),
    ("nanjing_hexi", "n5306854186_n6151399070::lane_0", 23.0, 33.0),
)
REBUILT_OBSTACLE_GAP_LANES = {
    # Pin a moderate-length multi-lane road instead of letting the generic
    # selector choose Wuxi's longest arterial.
    "wuxi_taihu": "n12140200733_n5450284639::lane_2",
}
# Four additional red-light approaches selected by the residual red phase,
# rather than the legacy ``longest_wait=True`` policy.  Each movement has
# 8--15 seconds of red remaining at t=0, at least 38 m of usable approach,
# and a short exit lane.  Together they cover right, straight and left turns.
RED_LIGHT_STOP_EXTRA_SITES = (
    ("sanfrancisco_soma", "connector::intersection::n6319217960::11"),
    ("moscow_tverskaya", "connector::intersection::n438050857::8"),
    ("vienna_innere", "connector::intersection::n199751::0"),
    ("shanghai_jinganbei", "connector::n555654415::3"),
)
# Vodičkova in Prague contains a real 3.5 m physical corridor shared by
# both travel directions.  It is bounded by ordinary two-way roads and has no
# signal control.  The eight frozen variants alternate ego direction and move
# the already-committed oncoming vehicle progressively deeper into the
# bottleneck.  That gives the oncoming actor an unambiguous right to clear the
# corridor and makes the evaluated behavior (wait, then enter) verifiable.
NARROW_CORRIDOR_NETWORK = "prague_staremesto"
NARROW_CORRIDOR_ROUTES = {
    "forward": {
        "approach_lane_id": "n25655303_n4747985386::lane_0",
        "shared_lane_ids": (
            "n25655303_n25655304::lane_shared_forward",
            "n25655304_n415004426::lane_shared_forward",
        ),
        "connector_ids": (
            "connector::n25655303::3",
            "connector::n25655304::0",
            "connector::n415004426::0",
        ),
        "exit_lane_id": "n415004426_n9062547291::lane_1",
    },
    "backward": {
        "approach_lane_id": "n415004426_n9062547291::lane_0",
        "shared_lane_ids": (
            "n25655304_n415004426::lane_shared_backward",
            "n25655303_n25655304::lane_shared_backward",
        ),
        "connector_ids": (
            "connector::n415004426::4",
            "connector::n25655304::2",
            "connector::n25655303::0",
        ),
        "exit_lane_id": "n25655303_n4747985386::lane_1",
    },
}
NARROW_CORRIDOR_VARIANTS = (
    ("forward_peer_early", "forward", 40.0, 20.0, 0, 0.12, 18.0),
    ("backward_peer_early", "backward", 40.0, 20.0, 0, 0.35, 18.0),
    ("forward_peer_mid", "forward", 34.0, 22.0, 0, 0.50, 20.0),
    ("backward_peer_mid", "backward", 34.0, 22.0, 1, 0.12, 20.0),
    ("forward_peer_deep", "forward", 28.0, 24.0, 0, 0.82, 22.0),
    ("backward_peer_deep", "backward", 28.0, 24.0, 1, 0.48, 22.0),
    ("forward_peer_near_exit", "forward", 22.0, 20.0, 1, 0.45, 20.0),
    ("backward_peer_near_exit", "backward", 22.0, 20.0, 1, 0.82, 20.0),
)
# All MultiLLM interaction contracts receive new IDs.  Even the previously
# valid two-car merge changes to a three-agent gap, so reusing 009--012 would
# make historical results look comparable when they are not.
MULTI_REBUILT_SCENES = (
    *(("narrow_meeting", NARROW_CORRIDOR_NETWORK, variant)
      for variant in NARROW_CORRIDOR_VARIANTS),
    *(("unprotected_left_turn", network, None) for network in (
        "beijing_guomao", "hongkong_central", "wuhan_hankou",
        "xian_zhonglou")),
    *(("synchronized_four_way", network, None) for network in (
        "beijing_guomao", "guangzhou_tianhe", "wuhan_hankou",
        "xian_zhonglou")),
    *(("three_agent_merge", network, None) for network in (
        "beijing_guomao", "shanghai_lujiazui", "wuhan_hankou",
        "xian_zhonglou")),
)
# Three staged SUMO vehicles are released when ego enters a 70 m radius.
# This makes ordinary oncoming traffic an observed part of the task instead
# of merely distributing unrelated vehicles elsewhere on the map.
FULL_NETWORK_ONCOMING_SITES = {
    "shanghai_lujiazui": {
        "lane_id": "n1800806199_n602392070::lane_0",
        "progresses": (0.76, 0.59, 0.42),
        "release_speed_kmh": 26.0,
        "environment_basis_s": 399.2,
        "duration_s": 418.3,
    },
    "chongqing_jiefangbei": {
        "lane_id": "n733200818_n733201080::lane_1",
        "progresses": (0.40, 0.25, 0.10),
        "release_speed_kmh": 26.0,
        "environment_basis_s": 387.9,
        "duration_s": 508.2,
    },
    "wuhan_hankou": {
        "lane_id": "n1880395961_n1880395997::lane_3",
        "progresses": (0.73, 0.54, 0.35),
        "release_speed_kmh": 26.0,
        "environment_basis_s": 62.9,
        "duration_s": 397.6,
    },
    "xian_zhonglou": {
        "lane_id": "n699662630_n733110598::lane_3",
        "progresses": (0.75, 0.58, 0.41),
        "release_speed_kmh": 26.0,
        "environment_basis_s": 105.5,
        "duration_s": 208.8,
    },
}
PEDESTRIAN_YIELD_CONNECTORS = {
    "beijing_guomao": "connector::n36704944::5",
    "hongkong_central": "connector::n1685959562::6",
    "shanghai_lujiazui": "connector::n495644095::0",
    "tokyo_shinjuku": "connector::n475770654::3",
    "guangzhou_tianhe": "connector::n320769078::0",
    "wuhan_hankou": "connector::n1419140537::6",
    "amsterdam_centrum": "connector::n9844170832::2",
    "newyork_manhattan_mid": "connector::n9140654137::0",
}
PEDESTRIAN_YIELD_EXTRA_NETWORKS = (
    "guangzhou_tianhe", "wuhan_hankou", "amsterdam_centrum", "newyork_manhattan_mid")
FORMAL_WEATHER_CONDITIONS = WEATHER_CONDITIONS
FORMAL_WEATHER_TARGET_CONDITIONS = WEATHER_TARGET_CONDITIONS
FORMAL_WEATHER_FIRST_CHANGE_FRACTION = (0.20, 0.40)
FORMAL_WEATHER_SECOND_CHANGE_FRACTION = (0.65, 0.85)
FORMAL_WEATHER_PATTERNS = WEATHER_PATTERNS
FORMAL_DAYNIGHT_PERIODS = DAYNIGHT_PERIODS
FORMAL_DAYNIGHT_DARK_PERIODS = tuple(
    period for period in FORMAL_DAYNIGHT_PERIODS
    if DayNight._DAYLIGHT_MAP[DayNight.TimePeriod(period)] < 30)
FORMAL_DAYNIGHT_BRIGHT_PERIODS = tuple(
    period for period in FORMAL_DAYNIGHT_PERIODS
    if period not in FORMAL_DAYNIGHT_DARK_PERIODS)
FORMAL_DAYNIGHT_FIRST_CHANGE_FRACTION = (0.20, 0.40)
FORMAL_DAYNIGHT_SECOND_CHANGE_FRACTION = (0.65, 0.85)
FORMAL_DAYNIGHT_PATTERNS = DAYNIGHT_PATTERNS
PINNED_REBUILT_ENVIRONMENT_PATTERNS: dict[str, dict] = {
    # These are the seven tasks whose frozen weather treatment has no cabin
    # rule.  Give them varied, forward-only day/night schedules.  The one
    # remaining task (multi_029) uses an evolving rainy weather treatment;
    # other tasks keep their existing environment mix, even when a change has
    # no direct cabin action.
    "basic_004_crosswalk__tokyo_shinjuku": {"daynight": "light_boundary"},
    "basic_012_signalized_intersection__tokyo_shinjuku": {"daynight": "forward_two_step"},
    "basic_040_full_network_route__xian_zhonglou": {"daynight": "light_boundary"},
    "multi_024_narrow_meeting__prague_vodickova__backward_peer_mid": {"daynight": "forward_two_step"},
    "multi_028_narrow_meeting__prague_vodickova__backward_peer_near_exit": {"daynight": "light_boundary"},
    "multi_029_unprotected_left_turn__beijing_guomao": {"weather": "evolve"},
    "multi_032_unprotected_left_turn__xian_zhonglou": {"daynight": "forward_two_step"},
    "multi_036_synchronized_four_way__xian_zhonglou": {"daynight": "light_boundary"},
}
DEFAULT_NETWORKS = (
    "beijing_guomao",
    "beijing_zhongguancun",
    "shanghai_lujiazui",
)

# Three additional maps per scene family expand the hand-curated physical
# templates without turning the expensive LLM study into a copy of
# the 116-map infrastructure suite.  The union is a 12-map stratified pilot
# portfolio; every entry below is exercised through real lane-level geometry.
SCENE_FAMILY_MAP_ADDITIONS = {
    "baseline": ("shanghai_lujiazui", "hongkong_central", "tokyo_shinjuku"),
    "traffic": ("shanghai_lujiazui", "guangzhou_tianhe",
           "newyork_manhattan_mid"),
    "pedestrian": ("wuhan_hankou", "hongkong_central", "tokyo_shinjuku"),
    "cabin": ("amsterdam_centrum", "hongkong_central",
           "newyork_manhattan_mid"),
    "chassis": ("chongqing_jiefangbei", "wuhan_hankou", "xian_zhonglou"),
    "multi": ("wuhan_hankou", "xian_zhonglou", "hongkong_central"),
    "concurrent": ("chengdu_jinjiang", "tokyo_shinjuku", "hongkong_central"),
}
BENCHMARK_NETWORKS = tuple(dict.fromkeys(
    (*DEFAULT_NETWORKS, *(
        network
        for family_maps in SCENE_FAMILY_MAP_ADDITIONS.values()
        for network in family_maps))))


def _stable_number(scene_id: str) -> int:
    return int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:8], 16)


def _minimum_two_change_gap_s(duration_s: float) -> float:
    """Keep changes well separated without pushing short-task frames out."""
    return round(min(15.0, max(6.0, 0.3 * float(duration_s))), 1)


def _weather_change_time_s(
    scene_id: str, salt: str, duration_s: float,
    fraction_range: tuple[float, float],
    *, not_before_s: float | None = None,
    avoid_s: Sequence[float] = (),
) -> float:
    """Return a reproducible, response-safe time on the 0.1 s grid.

    Weather and day/night share this scheduler. Non-joint environment events
    stay eight seconds apart when the interval has room; otherwise the caller
    can explicitly identify an exact-time joint event.
    """
    duration_s = float(duration_s)
    # Very short traffic tasks cannot reserve eight seconds on both sides of
    # a wake. Preserve 25% on each side there; 32s+ tasks use the full 8s.
    boundary_margin_s = min(8.0, max(2.0, duration_s * 0.25))
    lower = max(boundary_margin_s, duration_s * fraction_range[0])
    if not_before_s is not None:
        lower = max(lower, float(not_before_s))
    upper = min(
        duration_s - boundary_margin_s,
        duration_s * fraction_range[1])
    if upper + 1e-9 < lower:
        raise ValueError(
            f"environment schedule {duration_s}s has no valid {salt} window")
    lower_tick = int(math.ceil(lower * 10.0 - 1e-9))
    upper_tick = int(math.floor(upper * 10.0 + 1e-9))
    candidates = [tick for tick in range(lower_tick, upper_tick + 1)
                  if all(abs(tick / 10.0 - other) >= 8.0 - 1e-9
                         for other in avoid_s)]
    if not candidates:
        # Short combined tasks cannot hold two independent events in the same
        # window. Align exactly with a weather event and mark it as joint in
        # the returned profile instead of producing an accidental near-wake.
        joint = [int(round(other * 10.0)) for other in avoid_s
                 if lower_tick <= int(round(other * 10.0)) <= upper_tick]
        if joint:
            candidates = joint
        else:
            raise ValueError(
                f"environment schedule {duration_s}s cannot separate {salt}")
    slot = _stable_number(f"{scene_id}:{salt}") % len(candidates)
    return round(candidates[slot] / 10.0, 1)


def _daynight_change_times_s(
    scene_id: str, duration_s: float, change_count: int,
    *, avoid_s: Sequence[float] = (),
) -> list[float]:
    """Jointly schedule one or two day/night changes.

    Selecting the first change greedily can leave no legal window for a
    second change on short scenes.  Search the complete 0.1 s candidate grid
    instead, prefer schedules separated from every weather event by at least
    eight seconds, and use exact-time joint events only when separation is
    impossible.
    """
    duration_s = float(duration_s)
    boundary_margin_s = min(8.0, max(2.0, duration_s * 0.25))

    def window(fraction_range: tuple[float, float]) -> list[int]:
        lower = max(boundary_margin_s, duration_s * fraction_range[0])
        upper = min(
            duration_s - boundary_margin_s,
            duration_s * fraction_range[1])
        return list(range(
            int(math.ceil(lower * 10.0 - 1e-9)),
            int(math.floor(upper * 10.0 + 1e-9)) + 1,
        ))

    def weather_compatible(tick: int) -> bool:
        value = tick / 10.0
        return all(
            abs(value - other) < 1e-9
            or abs(value - other) >= 8.0 - 1e-9
            for other in avoid_s)

    def joint_count(ticks: Sequence[int]) -> int:
        return sum(
            any(abs(tick / 10.0 - other) < 1e-9 for other in avoid_s)
            for tick in ticks)

    first_ticks = [
        tick for tick in window(FORMAL_DAYNIGHT_FIRST_CHANGE_FRACTION)
        if weather_compatible(tick)
    ]
    candidates: list[tuple[int, ...]]
    if change_count == 1:
        candidates = [(tick,) for tick in first_ticks]
    elif change_count == 2:
        second_ticks = [
            tick for tick in window(FORMAL_DAYNIGHT_SECOND_CHANGE_FRACTION)
            if weather_compatible(tick)
        ]
        candidates = [
            (first_tick, second_tick)
            for first_tick in first_ticks
            for second_tick in second_ticks
            if second_tick - first_tick >= int(math.ceil(
                10.0 * _minimum_two_change_gap_s(duration_s) - 1e-9))
        ]
    else:
        raise ValueError(f"unsupported day/night change count {change_count}")
    if not candidates:
        raise ValueError(
            f"environment schedule {duration_s}s has no valid day/night "
            f"sequence with {change_count} changes")

    minimum_joint_count = min(joint_count(item) for item in candidates)
    preferred = [
        item for item in candidates
        if joint_count(item) == minimum_joint_count
    ]
    slot = _stable_number(f"{scene_id}:daynight-schedule") % len(preferred)
    return [round(tick / 10.0, 1) for tick in preferred[slot]]


def _formal_weather_profile(
    task_index: int, scene_id: str, duration_s: float,
    pattern: str | None, network: str,
) -> tuple[list[dict], dict]:
    """Build the deterministic weather treatment for a formal task."""
    climate_profile_id, _ = location_weather_profile(network)
    if pattern is None:
        return [], {
            "mode": "stable_control",
            "initial_condition": "sunny",
            "target_condition": None,
            "change_at_s": None,
            "primary_condition": None,
            "recovers_to_sunny": False,
            "recovery_at_s": None,
            "second_condition": None,
            "second_change_at_s": None,
            "conditions": ["sunny"],
            "climate_profile_id": (
                f"{network}@1991-2020-v1:{climate_profile_id}"),
            "climate_month": None,
        }

    selected_index = (task_index - 1) // 2
    if pattern not in FORMAL_WEATHER_PATTERNS:
        raise ValueError(f"unsupported formal weather pattern {pattern!r}")
    primary = select_weather_primary(network, selected_index, scene_id)
    conditions = weather_sequence(network, primary, pattern)
    selected_month = climate_month(network, conditions, scene_id)
    first_change_s = _weather_change_time_s(
        scene_id, "first", duration_s,
        FORMAL_WEATHER_FIRST_CHANGE_FRACTION)
    second_condition = conditions[2] if len(conditions) == 3 else None
    second_change_s = None
    if second_condition is not None:
        second_change_s = _weather_change_time_s(
            scene_id, "second", duration_s,
            FORMAL_WEATHER_SECOND_CHANGE_FRACTION,
            not_before_s=(
                first_change_s + _minimum_two_change_gap_s(duration_s)))

    change_times = [0.0, first_change_s]
    if second_change_s is not None:
        change_times.append(second_change_s)
    keyframes = [{
        "t": time_s / 60.0,
        "condition": condition,
        **weather_parameters(
            network, condition, conditions, scene_id),
    } for time_s, condition in zip(change_times, conditions)]
    return keyframes, {
        "mode": "transition",
        "pattern": pattern,
        "initial_condition": conditions[0],
        "target_condition": conditions[1],
        "change_at_s": first_change_s,
        "primary_condition": primary,
        "second_condition": second_condition,
        "second_change_at_s": second_change_s,
        "conditions": list(conditions),
        "climate_profile_id": (
            f"{network}@1991-2020-v1:{climate_profile_id}"),
        "climate_month": selected_month,
        "schedule_basis_s": float(duration_s),
    }


def _formal_daynight_profile(
    task_index: int, scene_id: str, duration_s: float,
    pattern: str | None, *, climate_month_value: int | None = None,
    avoid_change_times_s: Sequence[float] = (),
) -> tuple[list[dict], dict]:
    """Build one deterministic day/night treatment."""
    if pattern is None:
        return [], {
            "mode": "stable_control",
            "initial_period": "noon",
            "target_period": None,
            "change_at_s": None,
            "second_period": None,
            "second_change_at_s": None,
            "period_sequence": ["noon"],
            "climate_month": climate_month_value,
        }
    if pattern not in FORMAL_DAYNIGHT_PATTERNS:
        raise ValueError(
            f"unsupported formal day/night pattern {pattern!r}")

    group_index = (task_index - 1) // 4
    selected_index = group_index * 2 + int(task_index % 4 == 3)
    periods = daynight_sequence(selected_index, pattern)
    second_period = periods[2] if len(periods) == 3 else None
    change_schedule = _daynight_change_times_s(
        scene_id, duration_s, 2 if second_period is not None else 1,
        avoid_s=avoid_change_times_s)
    first_change_s = change_schedule[0]
    second_change_s = (
        change_schedule[1] if second_period is not None else None)

    change_times = [0.0, first_change_s]
    if second_change_s is not None:
        change_times.append(second_change_s)
    keyframes = [{
        "t": time_s / 60.0,
        "period": period,
        "description": DAYNIGHT_DESCRIPTION[period],
    } for time_s, period in zip(change_times, periods)]
    joint_weather_times = [
        time_s for time_s in change_times[1:]
        if any(abs(time_s - weather_time) < 1e-9
               for weather_time in avoid_change_times_s)]
    return keyframes, {
        "mode": "transition",
        "pattern": pattern,
        "initial_period": periods[0],
        "local_start_period": periods[0],
        "target_period": periods[1],
        "change_at_s": first_change_s,
        "second_period": second_period,
        "second_change_at_s": second_change_s,
        "period_sequence": list(periods),
        "climate_month": (
            climate_month_value
            if climate_month_value is not None
            else 1 + _stable_number(
                f"{scene_id}:daynight-climate-month") % 12),
        "joint_weather_change_times_s": joint_weather_times,
        "schedule_basis_s": float(duration_s),
    }


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _text_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SceneAsset:
    experiment_id: str
    scene_id: str
    scenario: dict
    expected: dict


class Topology:
    """Convenience index over one generated lane-level companion map."""

    def __init__(self, network: str, network_dir: Path):
        self.network = network
        self.path = network_dir / f"{network}_lane_level.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.runtime = LaneGeometryRuntime(self.data)
        self.lanes = self.data["lanes"]
        self.connectors = self.data["connectors"]
        self.crosswalks = self.data["crosswalks"]
        self.lane = {item["id"]: item for item in self.lanes}
        self.connector = {item["id"]: item for item in self.connectors}

    def long_lane(self, *, min_length: float = 100.0,
                  min_siblings: int = 1, ordinal: int = 0) -> dict:
        groups = defaultdict(list)
        for lane in self.lanes:
            groups[(lane["segment_id"], lane["direction"])].append(lane)
        candidates = [
            lane for lane in self.lanes
            if lane["length_m"] >= min_length
            and len(groups[(lane["segment_id"], lane["direction"])])
            >= min_siblings
        ]
        candidates.sort(key=lambda item: (-item["length_m"], item["id"]))
        if not candidates:
            raise ValueError(
                f"{self.network} has no lane matching length={min_length}, "
                f"siblings={min_siblings}")
        return candidates[ordinal % len(candidates)]

    def sibling_lanes(self, lane: Mapping[str, Any]) -> List[dict]:
        return sorted([
            item for item in self.lanes
            if item["segment_id"] == lane["segment_id"]
            and item["direction"] == lane["direction"]
        ], key=lambda item: (item["index"], item["id"]))

    @staticmethod
    def _point_segment_distance(point, start, end) -> float:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return math.dist(tuple(point), tuple(start))
        t = max(0.0, min(1.0, (
            (point[0] - start[0]) * dx
            + (point[1] - start[1]) * dy) / length_sq))
        projection = (start[0] + t * dx, start[1] + t * dy)
        return math.dist(tuple(point), projection)

    def lanes_have_opposing_geometric_overlap(
        self, first_lane_id: str, second_lane_id: str,
    ) -> bool:
        """Return true for different IDs sharing one opposing corridor."""
        first = self.lane[first_lane_id]
        second = self.lane[second_lane_id]
        if first_lane_id == second_lane_id:
            return False
        if int(first.get("z_level", 0)) != int(second.get("z_level", 0)):
            return False
        first_line = first.get("centerline_xy", [])
        second_line = second.get("centerline_xy", [])
        for a0, a1 in zip(first_line, first_line[1:]):
            adx, ady = a1[0] - a0[0], a1[1] - a0[1]
            alen = math.hypot(adx, ady)
            if alen < 2.0:
                continue
            for b0, b1 in zip(second_line, second_line[1:]):
                bdx, bdy = b1[0] - b0[0], b1[1] - b0[1]
                blen = math.hypot(bdx, bdy)
                if blen < 2.0:
                    continue
                alignment = (adx * bdx + ady * bdy) / (alen * blen)
                if alignment > -0.75:
                    continue
                distance = min(
                    self._point_segment_distance(a0, b0, b1),
                    self._point_segment_distance(a1, b0, b1),
                    self._point_segment_distance(b0, a0, a1),
                    self._point_segment_distance(b1, a0, a1),
                )
                if distance <= 1.2:
                    return True
        return False

    @staticmethod
    def route_lane_ids(lane: Mapping[str, Any], actions: Sequence[dict]):
        return [lane["id"], *[
            action["to_lane"] for action in actions
            if action.get("to_lane")]]

    def route_has_opposing_overlap(
        self, route_lane_ids: Sequence[str],
        accepted_lane_ids: Sequence[str],
    ) -> bool:
        return any(
            self.lanes_have_opposing_geometric_overlap(first, second)
            for first in route_lane_ids for second in accepted_lane_ids)

    def lanes_have_unmodeled_geometric_crossing(
        self, first_lane_id: str, second_lane_id: str,
        *, endpoint_clearance_m: float = 8.0,
    ) -> bool:
        """Detect an at-grade mid-lane crossing absent from lane topology.

        Valid junction interactions occur on connector paths after approach
        lanes reach their endpoints.  If two unrelated lane centerlines cross
        well inside both lanes, their vehicles cannot observe a connector
        conflict and the pairing is unsuitable for a controlled scenario.
        """
        if first_lane_id == second_lane_id:
            return False
        first = self.lane[first_lane_id]
        second = self.lane[second_lane_id]
        if first["segment_id"] == second["segment_id"]:
            return False
        if int(first.get("z_level", 0)) != int(second.get("z_level", 0)):
            return False
        hit = _polyline_intersection_distances(
            [tuple(point) for point in first.get("centerline_xy", [])],
            [tuple(point) for point in second.get("centerline_xy", [])],
        )
        if not hit:
            return False
        _, first_s, second_s = hit
        return (
            endpoint_clearance_m < first_s
            < float(first["length_m"]) - endpoint_clearance_m
            and endpoint_clearance_m < second_s
            < float(second["length_m"]) - endpoint_clearance_m)

    def route_has_unmodeled_crossing(
        self, route_lane_ids: Sequence[str],
        accepted_lane_ids: Sequence[str],
    ) -> bool:
        return any(
            self.lanes_have_unmodeled_geometric_crossing(first, second)
            for first in route_lane_ids for second in accepted_lane_ids)

    def red_connector(self, *, longest_wait: bool = False) -> dict:
        phase_zero_green = {
            connector_id
            for plan in self.data["signal_plans"]
            for connector_id in plan["phases"][0]["connector_ids"]
        }
        candidates = [
            item for item in self.connectors
            if item.get("signal_controlled")
            and item["id"] not in phase_zero_green
            and self.lane[item["from_lane"]]["length_m"] >= 50.0
        ]
        if not candidates:
            raise ValueError(f"{self.network} has no initially-red connector")
        def key(item):
            remaining = float(self.runtime.signal_state(
                item["id"], 0.0).remaining_seconds)
            return (
                -remaining if longest_wait else remaining,
                -self.lane[item["from_lane"]]["length_m"],
                item["id"],
            )
        return min(candidates, key=key)

    def green_connector(self) -> dict:
        phase_zero_green = {
            connector_id
            for plan in self.data["signal_plans"]
            for connector_id in plan["phases"][0]["connector_ids"]
        }
        candidates = sorted([
            self.connector[item] for item in phase_zero_green
            if item in self.connector
        ], key=lambda item: (-item["length_m"], item["id"]))
        if not candidates:
            raise ValueError(f"{self.network} has no initially-green connector")
        return candidates[0]

    def conflict_pair(self, *, signalized: bool | None = None,
                      turning: bool = False,
                      distinct_source_segments: bool = False,
                      minimum_source_length_m: float = 0.0,
                      minimum_crossing_angle_deg: float = 0.0,
                      prefer_crossing_angle: bool = False,
                      ) -> tuple[dict, dict]:
        candidates = []
        for first_id, others in self.runtime._connector_conflicts.items():
            first = self.connector[first_id]
            for second_id in others:
                if first_id >= second_id:
                    continue
                second = self.connector[second_id]
                controlled = bool(first.get("signal_controlled")
                                  or second.get("signal_controlled"))
                if signalized is not None and controlled != signalized:
                    continue
                if first["from_lane"] == second["from_lane"]:
                    continue
                first_lane = self.lane[first["from_lane"]]
                second_lane = self.lane[second["from_lane"]]
                if (distinct_source_segments
                        and first_lane["segment_id"]
                        == second_lane["segment_id"]):
                    continue
                if min(first_lane["length_m"], second_lane["length_m"]) \
                        < minimum_source_length_m:
                    continue
                if (turning
                        and first["turn"] == "straight"
                        and second["turn"] == "straight"):
                    continue
                points = self.runtime.connector_conflict_points(first_id)
                hit = next((item for item in points
                            if item["other_connector_id"] == second_id), None)
                if not hit:
                    continue
                first_progress = hit["self_distance_s_m"] / max(
                    0.1, first["length_m"])
                second_progress = hit["other_distance_s_m"] / max(
                    0.1, second["length_m"])
                if 0.12 <= first_progress <= 0.88 and 0.12 <= second_progress <= 0.88:
                    crossing_angle_deg = self.connector_conflict_angle_deg(
                        first, second)
                    if crossing_angle_deg < minimum_crossing_angle_deg:
                        continue
                    score = (
                        -crossing_angle_deg if prefer_crossing_angle else 0.0,
                        0 if first["turn"] != second["turn"] else 1,
                        abs(first_progress - 0.5)
                        + abs(second_progress - 0.5),
                        first_id, second_id,
                    )
                    candidates.append((score, first, second))
        if not candidates:
            raise ValueError(
                f"{self.network} has no requested connector conflict pair")
        _, first, second = min(candidates, key=lambda item: item[0])
        return first, second

    def connector_conflict_angle_deg(
        self, first: dict, second: dict,
    ) -> float:
        """Return the acute path angle at a modeled connector conflict."""
        hit = next(
            item for item in self.runtime.connector_conflict_points(first["id"])
            if item["other_connector_id"] == second["id"])
        _, _, first_heading = pose_at(
            [tuple(point) for point in first["centerline_xy"]],
            hit["self_distance_s_m"])
        _, _, second_heading = pose_at(
            [tuple(point) for point in second["centerline_xy"]],
            hit["other_distance_s_m"])
        difference = abs(
            (first_heading - second_heading + math.pi)
            % (2.0 * math.pi) - math.pi)
        acute = min(difference, math.pi - difference)
        return math.degrees(max(0.0, acute))

    def conflict_progress(self, first: dict, second: dict) -> tuple[float, float]:
        hit = next(
            item for item in self.runtime.connector_conflict_points(first["id"])
            if item["other_connector_id"] == second["id"])
        return (
            hit["self_distance_s_m"] / max(0.1, first["length_m"]),
            hit["other_distance_s_m"] / max(0.1, second["length_m"]),
        )

    def merge_pair(self, *, minimum_source_length_m: float = 0.0,
                   distinct_source_segments: bool = False,
                   ) -> tuple[dict, dict]:
        by_target = defaultdict(list)
        for connector in self.connectors:
            if not connector.get("signal_controlled"):
                by_target[connector["to_lane"]].append(connector)
        candidates = []
        for connectors in by_target.values():
            for index, first in enumerate(connectors):
                for second in connectors[index + 1:]:
                    if first["from_lane"] == second["from_lane"]:
                        continue
                    first_lane = self.lane[first["from_lane"]]
                    second_lane = self.lane[second["from_lane"]]
                    if (distinct_source_segments
                            and first_lane["segment_id"]
                            == second_lane["segment_id"]):
                        continue
                    if min(first_lane["length_m"], second_lane["length_m"]) \
                            < minimum_source_length_m:
                        continue
                    candidates.append((
                        -min(first["length_m"], second["length_m"]),
                        first["id"], second["id"], first, second))
        if not candidates:
            raise ValueError(f"{self.network} has no unsignalized merge")
        *_, first, second = min(candidates)
        return first, second

    def shared_opposing_lanes(self) -> tuple[dict, dict]:
        groups = defaultdict(list)
        for lane in self.lanes:
            if lane.get("shared_bidirectional"):
                groups[lane["segment_id"]].append(lane)
        candidates = []
        for lanes in groups.values():
            directions = {lane["direction"]: lane for lane in lanes}
            if "forward" in directions and "backward" in directions:
                candidates.append((
                    -directions["forward"]["length_m"],
                    directions["forward"]["id"],
                    directions["forward"], directions["backward"]))
        if not candidates:
            raise ValueError(f"{self.network} has no shared opposing lanes")
        *_, first, second = min(candidates)
        return first, second

    def ordinary_opposing_lanes(
        self, *, minimum_length_m: float = 180.0,
    ) -> tuple[dict, dict]:
        """Select an ordinary two-way road suitable for observed meetings."""
        groups = defaultdict(list)
        for lane in self.lanes:
            if (not lane.get("shared_bidirectional")
                    and float(lane["length_m"]) >= minimum_length_m):
                groups[lane["segment_id"]].append(lane)
        candidates = []
        for lanes in groups.values():
            by_direction = {
                direction: sorted(
                    (lane for lane in lanes
                     if lane["direction"] == direction),
                    key=lambda lane: (lane["index"], lane["id"]),
                )
                for direction in ("forward", "backward")
            }
            if not all(by_direction.values()):
                continue
            first = by_direction["forward"][0]
            second = by_direction["backward"][0]
            usable = min(float(first["length_m"]), float(second["length_m"]))
            # Prefer a clean one-lane-per-direction road around 300 m.  It is
            # long enough for three meetings without turning the scene into a
            # long-route navigation test.
            candidates.append((
                int(len(by_direction["forward"]) != 1
                    or len(by_direction["backward"]) != 1),
                abs(usable - 300.0), first["segment_id"], first, second,
            ))
        if not candidates:
            raise ValueError(
                f"{self.network} has no ordinary opposing road of "
                f"{minimum_length_m:g} m")
        *_, first, second = min(candidates, key=lambda item: item[:3])
        return first, second

    @staticmethod
    def approach_heading(lane: Mapping[str, Any]) -> float:
        points = list(lane.get("centerline_xy", []))
        for start, end in zip(reversed(points[:-1]), reversed(points[1:])):
            dx, dy = end[0] - start[0], end[1] - start[1]
            if math.hypot(dx, dy) > 1.0:
                return math.atan2(dy, dx)
        raise ValueError(f"lane {lane.get('id')} has no usable approach tangent")

    @staticmethod
    def heading_difference_deg(first: float, second: float) -> float:
        return math.degrees(abs(
            (first - second + math.pi) % (2.0 * math.pi) - math.pi))

    def unprotected_left_turn_pair(self) -> tuple[dict, dict]:
        """Return an unsignalized left turn conflicting with opposing through traffic."""
        candidates = []
        for left in self.connectors:
            if left["turn"] != "left" or left.get("signal_controlled"):
                continue
            left_lane = self.lane[left["from_lane"]]
            if float(left_lane["length_m"]) < 90.0:
                continue
            left_points = {
                item["other_connector_id"]: item
                for item in self.runtime.connector_conflict_points(left["id"])
            }
            for straight_id in self.runtime._connector_conflicts.get(
                    left["id"], set()):
                straight = self.connector[straight_id]
                if (straight["turn"] != "straight"
                        or straight.get("signal_controlled")
                        or straight["node_id"] != left["node_id"]):
                    continue
                straight_lane = self.lane[straight["from_lane"]]
                if (float(straight_lane["length_m"]) < 90.0
                        or straight_lane["segment_id"]
                        == left_lane["segment_id"]):
                    continue
                heading_difference = self.heading_difference_deg(
                    self.approach_heading(left_lane),
                    self.approach_heading(straight_lane))
                if heading_difference < 135.0:
                    continue
                hit = left_points.get(straight_id)
                if not hit:
                    continue
                # Keep the actual conflict point near the mouth of both
                # movements so approach TTC remains interpretable.
                left_conflict_s = float(hit["self_distance_s_m"])
                straight_conflict_s = float(hit["other_distance_s_m"])
                if max(left_conflict_s, straight_conflict_s) > 35.0:
                    continue
                candidates.append((
                    -heading_difference,
                    max(left_conflict_s, straight_conflict_s),
                    -min(float(left_lane["length_m"]),
                         float(straight_lane["length_m"])),
                    left["id"], straight["id"], left, straight,
                ))
        if not candidates:
            raise ValueError(
                f"{self.network} has no unprotected left-turn/across-path pair")
        return min(candidates, key=lambda item: item[:5])[-2:]

    def four_way_connectors(self) -> List[dict]:
        by_node = defaultdict(list)
        for item in self.connectors:
            if not item.get("signal_controlled"):
                by_node[item["node_id"]].append(item)
        candidates = []
        for node_id, connectors in by_node.items():
            connector_ids = {item["id"] for item in connectors}
            ranked = sorted(connectors, key=lambda item: (
                -len(self.runtime._connector_conflicts.get(
                    item["id"], set()) & connector_ids),
                item["id"],
            ))
            selected = []
            source_segments = set()
            for item in ranked:
                source_segment = self.lane[item["from_lane"]]["segment_id"]
                if source_segment in source_segments:
                    continue
                selected.append(item)
                source_segments.add(source_segment)
                if len(selected) == 4:
                    break
            if len(selected) < 4:
                continue
            pair_conflicts = sum(
                second["id"] in self.runtime._connector_conflicts.get(
                    first["id"], set())
                for index, first in enumerate(selected)
                for second in selected[index + 1:])
            if pair_conflicts >= 3:
                source_lengths = [
                    float(self.lane[item["from_lane"]]["length_m"])
                    for item in selected
                ]
                candidates.append((
                    -min(source_lengths), -sum(source_lengths),
                    -pair_conflicts, node_id, selected))
        if not candidates:
            raise ValueError(f"{self.network} has no four-way conflict set")
        selected = min(candidates, key=lambda item: item[:4])[4]
        # A short approach cannot honour a large requested starting distance.
        # Assign shorter approaches to earlier arrivals so clamping never
        # reverses the intended arrival order at the conflict area.
        return sorted(selected, key=lambda item: (
            self.lane[item["from_lane"]]["length_m"], item["id"]))

    def crosswalk_conflict(
        self, *, turning: bool = False,
        initial_signal: str | None = None,
        unsignalized: bool = False,
        required_connector_id: str | None = None,
    ) -> tuple[dict, dict]:
        if initial_signal not in (None, "red", "yellow", "green"):
            raise ValueError(
                f"unsupported initial crosswalk signal {initial_signal!r}")
        candidates = []
        for crosswalk in self.crosswalks:
            authored = crosswalk.get("source") == "authored_evaluation_unsignalized_crossing"
            if authored != unsignalized:
                continue
            for connector_id in crosswalk.get("conflicting_connectors", []):
                if required_connector_id and connector_id != required_connector_id:
                    continue
                connector = self.connector.get(connector_id)
                if not connector:
                    continue
                if unsignalized and crosswalk["road_segment_id"] != self.lane[connector["from_lane"]]["segment_id"]:
                    continue
                if turning and connector["turn"] not in ("left", "right"):
                    continue
                signal = self.runtime.signal_state(connector_id, 0.0)
                if unsignalized and (connector.get("signal_controlled") or signal is not None
                        or crosswalk_signal_relationship(self.runtime, crosswalk["id"], connector_id) != "unsignalized"):
                    continue
                if (initial_signal is not None
                        and (signal is None
                             or signal.signal != initial_signal)):
                    continue
                point = next((
                    item for item in self.runtime.connector_crosswalk_points(
                        connector_id)
                    if item["crosswalk_id"] == crosswalk["id"]
                ), None)
                if not point:
                    continue
                progress = point["connector_distance_s_m"] / max(
                    0.1, connector["length_m"])
                if (0.0 <= progress <= 1.0) if unsignalized else (0.05 <= progress <= 0.95):
                    connector_line = [tuple(value)
                                      for value in connector["centerline_xy"]]
                    crosswalk_line = [tuple(value)
                                      for value in crosswalk["centerline_xy"]]
                    near = (_polyline_intersection_distances(
                        connector_line, crosswalk_line)
                        or _polyline_nearest_distances(
                            connector_line, crosswalk_line))
                    if not near:
                        continue
                    _, connector_s, crosswalk_s = near
                    cx, cy, _ = pose_at(connector_line, connector_s)
                    px, py, _ = pose_at(crosswalk_line, crosswalk_s)
                    separation = math.hypot(cx - px, cy - py)
                    candidates.append((
                        (-float(signal.remaining_seconds)
                         if initial_signal is not None and signal else 0.0),
                        separation,
                        0 if connector["turn"] in ("left", "right") else 1,
                        abs(progress - 0.5), crosswalk["id"],
                        crosswalk, connector))
        if not candidates:
            qualifier = (f" with initial {initial_signal} signal"
                         if initial_signal is not None else "")
            raise ValueError(
                f"{self.network} has no crosswalk conflict{qualifier}")
        *_, crosswalk, connector = min(candidates)
        return crosswalk, connector

    def crosswalk_progress(self, crosswalk: dict, connector: dict) -> float:
        near = (_polyline_intersection_distances(
            [tuple(point) for point in connector["centerline_xy"]],
            [tuple(point) for point in crosswalk["centerline_xy"]],
        ) or _polyline_nearest_distances(
            [tuple(point) for point in connector["centerline_xy"]],
            [tuple(point) for point in crosswalk["centerline_xy"]],
        ))
        if not near:
            raise ValueError("crosswalk and connector have no nearest points")
        return near[1] / max(0.1, connector["length_m"])

    def pedestrian_crosswalk_progress(
        self, crosswalk: dict, connector: dict,
    ) -> float:
        hit = (_polyline_intersection_distances(
            [tuple(point) for point in connector["centerline_xy"]],
            [tuple(point) for point in crosswalk["centerline_xy"]],
        ) or _polyline_nearest_distances(
            [tuple(point) for point in connector["centerline_xy"]],
            [tuple(point) for point in crosswalk["centerline_xy"]],
        ))
        if not hit:
            return 0.5
        return hit[2] / max(0.1, float(crosswalk["length_m"]))

    def connector_chains(
        self, *, count: int, minimum_connectors: int = 4,
        minimum_turns: int = 0,
    ) -> List[tuple[dict, List[dict]]]:
        """Return deterministic non-looping connector paths."""
        by_source = defaultdict(list)
        for item in self.connectors:
            if item["turn"] != "uturn":
                by_source[item["from_lane"]].append(item)
        for values in by_source.values():
            values.sort(key=lambda item: (
                0 if item["turn"] == "straight" else 1, item["id"]))
        chains = []
        used_start_segments = set()
        for lane in sorted(
                self.lanes, key=lambda item: (-item["length_m"], item["id"])):
            cursor = lane["id"]
            actions = []
            visited = {cursor}
            for step in range(max(minimum_connectors + 4, 8)):
                choices = [
                    item for item in by_source.get(cursor, [])
                    if item["to_lane"] not in visited]
                if not choices:
                    break
                if (sum(action["turn"] != "straight" for action in actions)
                        < minimum_turns):
                    choices.sort(key=lambda item: (
                        0 if item["turn"] != "straight" else 1,
                        item["id"]))
                chosen = choices[step % len(choices)]
                actions.append(chosen)
                cursor = chosen["to_lane"]
                visited.add(cursor)
            # Lane IDs can be unique while the route still closes a node-level
            # cycle. Such a chain gives the evaluated vehicle identical start
            # and destination nodes, so remove only the cyclic tail.
            while (actions and self.lane[
                    actions[-1]["to_lane"]]["end_node"]
                    == lane["start_node"]):
                actions.pop()
            if (len(actions) >= minimum_connectors
                    and sum(item["turn"] != "straight" for item in actions)
                    >= minimum_turns
                    and lane["segment_id"] not in used_start_segments):
                chains.append((lane, actions))
                used_start_segments.add(lane["segment_id"])
                if len(chains) >= count:
                    return chains
        raise ValueError(
            f"{self.network} has only {len(chains)} suitable route chains; "
            f"needed {count}")


def _vehicle(vid: str, lane: dict, *, destination_node: str | None = None,
             progress: float = 0.0, speed: float = 0.0,
             connector: dict | None = None,
             crashed: bool = False, evaluated: bool = True,
             chassis: str = "sedan", equipment: str = "executive") -> dict:
    target = destination_node or (
        lane["end_node"] if connector is None else destination_node)
    placement = {
        "progress": round(float(progress), 6),
        "speed_kmh": float(speed),
        "target_speed_kmh": float(speed),
        "desired_speed_kmh": max(30.0, float(speed)),
    }
    if connector is None:
        placement["lane_id"] = lane["id"]
    else:
        placement["connector_id"] = connector["id"]
    if crashed:
        placement["crashed"] = True
    return {
        "vehicle_id": vid,
        "initial_node": lane["start_node"],
        "destination_node": target or lane["end_node"],
        "initial_lane": int(lane["index"]),
        "is_evaluated": evaluated,
        "equipment_profile": equipment,
        "chassis_profile": chassis,
        "agent_config": {"type": "sumo"},
        "initial_physical_state": placement,
    }


def _connector_vehicle(
    topology: Topology, vid: str, connector: dict, *, progress: float,
    speed: float, evaluated: bool = True,
) -> dict:
    source = topology.lane[connector["from_lane"]]
    target = topology.lane[connector["to_lane"]]
    return _vehicle(
        vid, source, destination_node=target["end_node"],
        progress=progress, speed=speed,
        connector=connector, evaluated=evaluated)


def _approach_connector_vehicle(
    topology: Topology, vid: str, connector: dict, *, distance_m: float,
    speed: float, evaluated: bool = True,
) -> dict:
    source = topology.lane[connector["from_lane"]]
    target = topology.lane[connector["to_lane"]]
    vehicle = _vehicle(
        vid, source, destination_node=target["end_node"],
        progress=max(0.03, 1.0 - distance_m / source["length_m"]),
        speed=speed, evaluated=evaluated)
    vehicle["initial_physical_state"]["lane_route_actions"] = [{
        "type": "connector", "connector_id": connector["id"],
    }]
    return vehicle


def _pedestrian(
    pid: str, *, initial_node: str, destination_node: str,
    speed: float = 1.4,
    crosswalk: dict | None = None,
    progress: float = 0.0,
) -> dict:
    placement: dict = {
        "progress": round(float(progress), 6),
        "speed_mps": float(speed),
        "spawned": True,
    }
    if crosswalk:
        placement["crosswalk_id"] = crosswalk["id"]
    return {
        "ped_id": pid,
        "initial_node": initial_node,
        "destination_node": destination_node,
        "agent_config": {"type": "sumo"},
        "speed": speed,
        "start_time": 0.0,
        "is_evaluated": True,
        "collision_radius_m": 0.4,
        "initial_physical_state": placement,
    }


def _scene(
    experiment_id: str, scene_id: str, topology: Topology,
    *, title: str, purpose: str, vehicles: Sequence[dict],
    pedestrians: Sequence[dict] = (), duration_s: float = 20.0,
    tags: Sequence[str] = (), factors: Sequence[str] = (),
    acceptance: Sequence[dict] = (),
    runtime_acceptance: Sequence[dict] = (),
) -> SceneAsset:
    setup_assertions = [
        {"kind": "entity_count", "vehicles": len(vehicles),
         "pedestrians": len(pedestrians)},
        *copy.deepcopy(list(acceptance)),
    ]
    geometry = {
        "lane_ids": sorted({
            state["initial_physical_state"]["lane_id"]
            for state in vehicles
            if "lane_id" in state.get("initial_physical_state", {})}),
        "connector_ids": sorted({
            state["initial_physical_state"]["connector_id"]
            for state in vehicles
            if "connector_id" in state.get("initial_physical_state", {})}),
        "crosswalk_ids": sorted({
            state["initial_physical_state"]["crosswalk_id"]
            for state in pedestrians
            if "crosswalk_id" in state.get("initial_physical_state", {})}),
    }
    geometry["connector_ids"] = sorted(set(geometry["connector_ids"]) | {
        action["connector_id"]
        for state in vehicles
        for action in state.get(
            "initial_physical_state", {}).get("lane_route_actions", [])
        if action.get("type") == "connector"
    })
    scenario = {
        "scenario_id": scene_id,
        "name": title,
        "road_network_id": topology.network,
        "difficulty": "controlled",
        "total_time_s": float(duration_s),
        "tick_interval_s": 2.0,
        "physics_step_s": 0.1,
        "vehicles": copy.deepcopy(list(vehicles)),
        "pedestrians": copy.deepcopy(list(pedestrians)),
        "experiment_scene": {
            "schema": CATALOG_SCHEMA,
            "experiment_id": experiment_id,
            "purpose": purpose,
            "interaction_tags": list(tags),
            "factor_axes": list(factors),
            "geometry": geometry,
            # Runtime-critical setup conditions travel with the executable
            # scene. A frozen manifest must not silently lose the assertions
            # that made the authored experiment meaningful.
            "setup_assertions": copy.deepcopy(setup_assertions),
        },
    }
    expected = {
        "schema": CATALOG_SCHEMA,
        "scene_id": scene_id,
        "setup_assertions": copy.deepcopy(setup_assertions),
        "smoke": {
            "duration_s": min(8.0, float(duration_s)),
            "all_agents": "sumo",
            "forbid_initialization_error": True,
            "allow_initial_collision": False,
            "minimum_initial_collisions": 0,
        },
    }
    if runtime_acceptance:
        expected["runtime_acceptance"] = copy.deepcopy(
            list(runtime_acceptance))
    return SceneAsset(experiment_id, scene_id, scenario, expected)


def _following(topology: Topology, experiment_id: str, scene_id: str,
               *, collision: bool = False, title: str,
               factors: Sequence[str] = (),
               lane_id: str | None = None) -> SceneAsset:
    lane = (topology.long_lane(min_length=140.0, min_siblings=2)
            if lane_id is None else topology.lane[lane_id])
    if lane["length_m"] < 140.0 or len(topology.sibling_lanes(lane)) < 2:
        raise ValueError(
            f"{topology.network} lane {lane['id']} is not a valid "
            "multi-lane following site")
    gap_m = 3.0 if collision else 28.0
    lead_progress = 0.58
    follower_progress = lead_progress - gap_m / lane["length_m"]
    vehicles = [
        _vehicle("ego", lane, progress=follower_progress, speed=38.0,
                 evaluated=True),
        _vehicle("lead", lane, progress=lead_progress,
                 speed=0.0 if collision else 22.0, evaluated=False),
    ]
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="Controlled same-lane car-following interaction.",
        vehicles=vehicles, duration_s=18.0,
        tags=("following", "rear_end" if collision else "car_following"),
        factors=factors,
        acceptance=(
            {"kind": "same_lane", "entities": ["ego", "lead"]},
            {"kind": "initial_longitudinal_gap_m", "rear": "ego",
             "front": "lead", "maximum": gap_m + 0.5},
        ))


def _lead_vehicle_braking(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
    factors: Sequence[str] = (), lane_id: str | None = None,
    environment_basis_s: float | None = None,
    duration_s: float | None = None,
) -> SceneAsset:
    """Build a SUMO-native following task under the legacy task family ID."""
    asset = _following(
        topology, experiment_id, scene_id, collision=False, title=title,
        factors=factors, lane_id=lane_id)
    asset.scenario["experiment_scene"]["purpose"] = (
        "SUMO-native same-lane following; the lead vehicle is never "
        "scripted after initialization.")
    asset.scenario["experiment_scene"]["interaction_tags"] = [
        tag for tag in asset.scenario["experiment_scene"]["interaction_tags"]
        if tag != "car_following"
    ] + ["native_sumo_following"]
    if environment_basis_s is not None:
        asset.scenario["experiment_scene"][
            "environment_schedule_basis_s"] = float(environment_basis_s)
    if duration_s is not None:
        asset.scenario["total_time_s"] = float(duration_s)
    return asset


def _oncoming_stream(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
) -> SceneAsset:
    """Three ordinary oncoming vehicles must physically pass the ego."""
    ego_lane, opposing_lane = topology.ordinary_opposing_lanes()
    ego = _vehicle("ego", ego_lane, progress=0.10, speed=28.0)
    peers = []
    progresses = (0.10, 0.35, 0.60)
    for index, progress in enumerate(progresses, 1):
        peer = _vehicle(
            f"oncoming_{index}", opposing_lane, progress=progress,
            speed=26.0, evaluated=False)
        peer["initial_physical_state"]["native_lane_change_enabled"] = False
        peers.append(peer)
    duration_s = max(
        45.0,
        min(150.0, (1.0 - 0.10) * float(ego_lane["length_m"])
            / (28.0 / 3.6) + 15.0),
    )
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "Observe and safely pass three moving vehicles on the opposing "
            "lane of the ego road."),
        vehicles=[ego, *peers], duration_s=duration_s,
        tags=("oncoming", "meeting_traffic", "opposite_direction",
              "three_encounters"),
        factors=("encounter_count",),
        acceptance=({
            "kind": "ordinary_oncoming_stream",
            "entity": "ego",
            "peer_ids": [peer["vehicle_id"] for peer in peers],
            "ego_lane_id": ego_lane["id"],
            "opposing_lane_id": opposing_lane["id"],
            "peer_progresses": list(progresses),
            "minimum_road_length_m": 180.0,
        },),
        runtime_acceptance=(
            {"kind": "minimum_opposing_encounters", "entity": "ego",
             "peer_ids": [peer["vehicle_id"] for peer in peers],
             "distance_m": 30.0, "minimum_distinct_peers": 3,
             "minimum_heading_difference_deg": 120.0},
            {"kind": "collision_free", "entities": [
                "ego", *[peer["vehicle_id"] for peer in peers]]},
            {"kind": "route_arrival", "entities": ["ego"]},
        ),
    )


def _conflict_distance_m(
    topology: Topology, connector: Mapping[str, Any],
    other: Mapping[str, Any],
) -> float:
    hit = next(
        item for item in topology.runtime.connector_conflict_points(
            str(connector["id"]))
        if item["other_connector_id"] == other["id"])
    return float(hit["self_distance_s_m"])


def _unprotected_left_turn(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
    focal_turning: bool = True, through_vehicle_count: int = 1,
) -> SceneAsset:
    """Build a left-turn-across-opposing-path encounter with explicit TTC."""
    if through_vehicle_count not in (1, 2, 3):
        raise ValueError(
            "left-turn task supports one, two, or three through vehicles")
    if not focal_turning and through_vehicle_count != 1:
        raise ValueError("a through-vehicle focal task supports one ego only")
    left, straight = topology.unprotected_left_turn_pair()
    # The through vehicle reaches the conflict 0.8 s before the left turn.
    # This gives a deterministic priority relation without separating the
    # vehicles enough to remove the interaction.
    left_ttc_s = 9.4
    through_ttc_s = [8.6, 10.4, 12.2][:through_vehicle_count]
    speeds = {"left": 20.0, "straight": 28.0}
    left_setback = (left_ttc_s * speeds["left"] / 3.6
                    - _conflict_distance_m(topology, left, straight))
    through_setbacks = [
        ttc * speeds["straight"] / 3.6
        - _conflict_distance_m(topology, straight, left)
        for ttc in through_ttc_s]
    if min([left_setback, *through_setbacks]) < 10.0:
        raise ValueError(
            f"{topology.network} left-turn conflict has insufficient approach")
    left_id = "ego" if focal_turning else "turning_peer"
    through_ids = (
        [f"oncoming_through_{index}" for index in range(
            1, through_vehicle_count + 1)]
        if focal_turning else ["ego"])
    left_vehicle = _approach_connector_vehicle(
        topology, left_id, left, distance_m=left_setback,
        speed=speeds["left"], evaluated=focal_turning)
    through_vehicles = [
        _approach_connector_vehicle(
            topology, entity_id, straight, distance_m=setback,
            speed=speeds["straight"], evaluated=not focal_turning)
        for entity_id, setback in zip(through_ids, through_setbacks)]
    vehicles = (
        [left_vehicle, *through_vehicles]
        if focal_turning else [*through_vehicles, left_vehicle])
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "An unsignalized left turn yields to opposing through traffic "
            "arriving as a one- or two-vehicle stream around the turn window."),
        vehicles=vehicles, duration_s=45.0,
        tags=("intersection", "unsignalized", "unprotected_left_turn",
              "left_turn_across_path", "opposing_traffic",
              "required_yield"),
        factors=("conflict_role", "arrival_time_gap"),
        acceptance=({
            "kind": "unprotected_left_turn_across_path",
            "left_turn_entity": left_id,
            "through_entities": through_ids,
            "left_connector_id": left["id"],
            "through_connector_id": straight["id"],
            "left_initial_ttc_s": left_ttc_s,
            "through_initial_ttc_s": through_ttc_s,
            "minimum_opposing_heading_deg": 135.0,
            "maximum_ttc_gap_s": 1.0,
            "maximum_through_headway_s": (
                1.9 * max(1, through_vehicle_count - 1)),
        },),
        runtime_acceptance=(
            {"kind": "connector_conflict_observed", "entities": [
                left_id, *through_ids], "maximum_ttc_gap_s": 2.0},
            {"kind": "connector_entry_order", "order": [
                *through_ids, left_id]},
            {"kind": "collision_free", "entities": [
                left_id, *through_ids]},
            {"kind": "route_arrival", "entities": [
                left_id, *through_ids]},
        ),
    )


def _platoon_pressure(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
) -> SceneAsset:
    """SUMO-native traffic surrounds ego; no scripted lead speed override."""
    lane = topology.long_lane(min_length=160.0, min_siblings=3)
    siblings = topology.sibling_lanes(lane)
    adjacent = min(
        (item for item in siblings if item["id"] != lane["id"]),
        key=lambda item: abs(item["index"] - lane["index"]))
    lead_progress = 0.64
    ego_progress = lead_progress - 26.0 / lane["length_m"]
    rear_progress = ego_progress - 24.0 / lane["length_m"]
    vehicles = [
        _vehicle("ego", lane, progress=ego_progress, speed=36.0),
        _vehicle("lead", lane, progress=lead_progress, speed=24.0,
                 evaluated=False),
        _vehicle("rear", lane, progress=rear_progress, speed=34.0,
                 evaluated=False),
        _vehicle("side_front", adjacent,
                 progress=ego_progress + 18.0 / lane["length_m"],
                 speed=30.0, evaluated=False),
        _vehicle("side_rear", adjacent,
                 progress=ego_progress - 16.0 / lane["length_m"],
                 speed=32.0, evaluated=False),
    ]
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "Ego follows SUMO-native traffic while a close rear follower and "
            "an occupied adjacent lane require observation before maneuvering."),
        vehicles=vehicles, duration_s=35.0,
        tags=("following", "platoon", "native_sumo_following", "rear_pressure",
              "adjacent_lane_occupied"),
        factors=("longitudinal_pressure", "adjacent_occupancy"),
        acceptance=({
            "kind": "platoon_pressure_setup", "entity": "ego",
            "lead": "lead", "rear": "rear",
            "side_entities": ["side_front", "side_rear"],
            "lead_gap_m": 26.0, "rear_gap_m": 24.0,
            "adjacent_lane_id": adjacent["id"],
        },),
        runtime_acceptance=(
            {"kind": "minimum_same_lane_encounter", "entity": "ego",
             "peer": "lead", "distance_m": 30.0},
            {"kind": "collision_free", "entities": [
                vehicle["vehicle_id"] for vehicle in vehicles]},
            {"kind": "route_arrival", "entities": ["ego"]},
        ),
    )


def _obstacle_gap_change(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
    lane_id: str | None = None,
) -> SceneAsset:
    """A fixed obstacle forces ego to enter a genuinely occupied target lane."""
    lane = (topology.long_lane(min_length=180.0, min_siblings=3, ordinal=2)
            if lane_id is None else topology.lane[lane_id])
    siblings = topology.sibling_lanes(lane)
    adjacent = min(
        (item for item in siblings if item["id"] != lane["id"]),
        key=lambda item: abs(item["index"] - lane["index"]))
    ego_progress = 0.28
    vehicles = [
        _vehicle("ego", lane, progress=ego_progress, speed=32.0),
        _vehicle("crashed_vehicle", lane,
                 progress=ego_progress + 55.0 / lane["length_m"],
                 speed=0.0, crashed=True, evaluated=False),
        _vehicle("gap_front", adjacent,
                 progress=ego_progress + 24.0 / lane["length_m"],
                 speed=27.0, evaluated=False),
        _vehicle("gap_rear", adjacent,
                 progress=ego_progress - 18.0 / lane["length_m"],
                 speed=31.0, evaluated=False),
    ]
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "A lane obstacle makes a lane change mandatory while moving "
            "vehicles bound the only usable target-lane gap."),
        vehicles=vehicles, duration_s=45.0,
        tags=("crash_obstacle", "required_lane_change", "gap_acceptance",
              "target_lane_occupied"),
        factors=("mandatory_lane_change", "moving_gap"),
        acceptance=(
            {"kind": "crashed_lane_obstacle", "entity": "crashed_vehicle"},
            {"kind": "obstacle_gap_change_setup", "entity": "ego",
             "obstacle": "crashed_vehicle", "front": "gap_front",
             "rear": "gap_rear", "target_lane_id": adjacent["id"],
             "obstacle_gap_m": 55.0, "front_offset_m": 24.0,
             "rear_offset_m": 18.0},
        ),
        runtime_acceptance=(
            {"kind": "lane_change_completed", "entity": "ego",
             "target_lane_id": adjacent["id"], "before_entity":
             "crashed_vehicle"},
            {"kind": "minimum_adjacent_encounters", "entity": "ego",
             "peer_ids": ["gap_front", "gap_rear"],
             "distance_m": 35.0, "minimum_distinct_peers": 2},
            {"kind": "collision_free", "entities": [
                "ego", "crashed_vehicle", "gap_front", "gap_rear"]},
            {"kind": "route_arrival", "entities": ["ego"]},
        ),
    )


def _crossing_stream(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
) -> SceneAsset:
    ego_connector, cross_connector = topology.conflict_pair(
        signalized=False, distinct_source_segments=True,
        minimum_source_length_m=90.0,
        minimum_crossing_angle_deg=55.0, prefer_crossing_angle=True)
    ego_setback = 48.0
    cross_setbacks = (30.0, 55.0, 80.0)
    vehicles = [_approach_connector_vehicle(
        topology, "ego", ego_connector, distance_m=ego_setback,
        speed=25.0)]
    for index, setback in enumerate(cross_setbacks, 1):
        vehicles.append(_approach_connector_vehicle(
            topology, f"cross_{index}", cross_connector,
            distance_m=setback, speed=28.0, evaluated=False))
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "Ego selects a safe gap in a three-vehicle priority stream at "
            "one unsignalized crossing conflict."),
        vehicles=vehicles, duration_s=50.0,
        tags=("intersection", "unsignalized", "crossing_stream",
              "gap_acceptance", "three_conflicts"),
        factors=("crossing_stream", "gap_acceptance"),
        acceptance=({
            "kind": "localized_crossing_stream", "entity": "ego",
            "cross_entities": [f"cross_{index}" for index in range(1, 4)],
            "ego_connector_id": ego_connector["id"],
            "cross_connector_id": cross_connector["id"],
            "ego_setback_m": ego_setback,
            "cross_setbacks_m": list(cross_setbacks),
            "minimum_crossing_angle_deg": 55.0,
        },),
        runtime_acceptance=(
            {"kind": "minimum_connector_conflicts", "entity": "ego",
             "peer_ids": [f"cross_{index}" for index in range(1, 4)],
             "minimum_distinct_peers": 3},
            {"kind": "collision_free", "entities": [
                vehicle["vehicle_id"] for vehicle in vehicles]},
            {"kind": "route_arrival", "entities": ["ego"]},
        ),
    )


def _merge_stream(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
) -> SceneAsset:
    ego_connector, stream_connector = topology.merge_pair(
        minimum_source_length_m=90.0, distinct_source_segments=True)
    ego_setback = 45.0
    stream_setbacks = (25.0, 50.0, 75.0)
    vehicles = [_approach_connector_vehicle(
        topology, "ego", ego_connector, distance_m=ego_setback,
        speed=24.0)]
    for index, setback in enumerate(stream_setbacks, 1):
        vehicles.append(_approach_connector_vehicle(
            topology, f"stream_{index}", stream_connector,
            distance_m=setback, speed=26.0, evaluated=False))
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "Ego merges between successive vehicles already approaching the "
            "same target lane."),
        vehicles=vehicles, duration_s=50.0,
        tags=("merge", "merge_stream", "gap_acceptance",
              "three_target_lane_vehicles"),
        factors=("merge_stream", "gap_acceptance"),
        acceptance=({
            "kind": "localized_merge_stream", "entity": "ego",
            "stream_entities": [f"stream_{index}" for index in range(1, 4)],
            "ego_connector_id": ego_connector["id"],
            "stream_connector_id": stream_connector["id"],
            "ego_setback_m": ego_setback,
            "stream_setbacks_m": list(stream_setbacks),
        },),
        runtime_acceptance=(
            {"kind": "minimum_merge_encounters", "entity": "ego",
             "peer_ids": [f"stream_{index}" for index in range(1, 4)],
             "minimum_distinct_peers": 2, "distance_m": 30.0},
            {"kind": "merge_completed", "entity": "ego",
             "target_lane_id": ego_connector["to_lane"]},
            {"kind": "collision_free", "entities": [
                vehicle["vehicle_id"] for vehicle in vehicles]},
            {"kind": "route_arrival", "entities": ["ego"]},
        ),
    )


def _three_agent_merge(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
) -> SceneAsset:
    """Place ego inside a three-vehicle converging stream."""
    ego_connector, stream_connector = topology.merge_pair(
        minimum_source_length_m=90.0, distinct_source_segments=True)
    ego_setback = 45.0
    # Keep a generous physical gap on short or sharply curved approaches.
    # A 25 m authored separation was geometrically valid but Hong Kong's
    # native SUMO lane insertion reported the pair as colliding at tick 0.
    stream_setbacks = (35.0, 75.0, 115.0)
    vehicles = [_approach_connector_vehicle(
        topology, "ego", ego_connector, distance_m=ego_setback,
        speed=24.0)]
    stream_ids = ("merge_lead", "merge_follower", "merge_tail")
    for entity_id, setback in zip(stream_ids, stream_setbacks):
        vehicles.append(_approach_connector_vehicle(
            topology, entity_id, stream_connector,
            distance_m=setback, speed=26.0, evaluated=True))
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "Four LLM vehicles share one merge window: ego must enter the "
            "gap ahead of two successive followers on the converging approach."),
        vehicles=vehicles, duration_s=50.0,
        tags=("merge", "four_agent_merge", "gap_acceptance",
              "multi_agent_negotiation"),
        factors=("llm_count", "model_assignment", "merge_order"),
        acceptance=({
            "kind": "localized_merge_stream", "entity": "ego",
            "stream_entities": list(stream_ids),
            "ego_connector_id": ego_connector["id"],
            "stream_connector_id": stream_connector["id"],
            "ego_setback_m": ego_setback,
            "stream_setbacks_m": list(stream_setbacks),
            "expected_stream_count": 3,
            "all_entities_evaluated": True,
        },),
        runtime_acceptance=(
            {"kind": "minimum_merge_encounters", "entity": "ego",
             "peer_ids": list(stream_ids),
             "minimum_distinct_peers": 3, "distance_m": 45.0},
            {"kind": "connector_entry_order", "order": [
                "merge_lead", "ego", "merge_follower", "merge_tail"]},
            {"kind": "merge_completed", "entity": "ego",
             "target_lane_id": ego_connector["to_lane"]},
            {"kind": "collision_free", "entities": [
                "ego", *stream_ids]},
            {"kind": "route_arrival", "entities": [
                "ego", *stream_ids]},
        ),
    )


def _narrow_road_sequence(
    topology: Topology, variant: Sequence[Any], *,
    experiment_id: str = "chassis",
    scene_prefix: str = "chassis_narrow_road",
    peer_evaluated: bool = False,
    add_oncoming_follower: bool = False,
) -> SceneAsset:
    """Build one opposing-traffic sequence at the Prague shared corridor."""
    if topology.network != NARROW_CORRIDOR_NETWORK:
        raise ValueError(
            f"narrow corridor is authored for {NARROW_CORRIDOR_NETWORK}, "
            f"not {topology.network}")
    (suffix, ego_direction, ego_setback_m, ego_speed_kmh,
     peer_shared_index, peer_progress, peer_speed_kmh) = variant
    ego_route = NARROW_CORRIDOR_ROUTES[str(ego_direction)]
    peer_direction = (
        "backward" if ego_direction == "forward" else "forward")
    peer_route = NARROW_CORRIDOR_ROUTES[peer_direction]

    ego_lane = topology.lane[ego_route["approach_lane_id"]]
    ego_exit = topology.lane[ego_route["exit_lane_id"]]
    ego = _vehicle(
        "ego", ego_lane, destination_node=ego_exit["end_node"],
        progress=max(0.03, 1.0 - float(ego_setback_m) / ego_lane["length_m"]),
        speed=float(ego_speed_kmh), evaluated=True)
    ego["initial_physical_state"]["lane_route_actions"] = [
        {"type": "connector", "connector_id": connector_id}
        for connector_id in ego_route["connector_ids"]
    ]
    ego["initial_physical_state"]["native_lane_change_enabled"] = False

    peer_lane_id = peer_route["shared_lane_ids"][int(peer_shared_index)]
    peer_lane = topology.lane[peer_lane_id]
    peer_exit = topology.lane[peer_route["exit_lane_id"]]
    peer = _vehicle(
        "oncoming", peer_lane, destination_node=peer_exit["end_node"],
        progress=float(peer_progress), speed=float(peer_speed_kmh),
        evaluated=peer_evaluated)
    # connector_ids = entry, optional internal transition, exit.  The peer is
    # already committed inside the corridor, so retain only its remaining
    # physical route.
    peer_remaining_connectors = peer_route["connector_ids"][
        int(peer_shared_index) + 1:]
    peer["initial_physical_state"]["lane_route_actions"] = [
        {"type": "connector", "connector_id": connector_id}
        for connector_id in peer_remaining_connectors
    ]
    peer["initial_physical_state"]["native_lane_change_enabled"] = False

    followers = []
    follower_setbacks_m = (18.0, 40.0)
    if add_oncoming_follower:
        follower_lane = topology.lane[peer_route["approach_lane_id"]]
        for follower_id, follower_setback_m in zip(
                ("oncoming_follower", "oncoming_tail"),
                follower_setbacks_m):
            follower = _vehicle(
                follower_id, follower_lane,
                destination_node=peer_exit["end_node"],
                progress=max(
                    0.03,
                    1.0 - follower_setback_m / follower_lane["length_m"]),
                speed=18.0, evaluated=peer_evaluated)
            follower["initial_physical_state"]["lane_route_actions"] = [
                {"type": "connector", "connector_id": connector_id}
                for connector_id in peer_route["connector_ids"]
            ]
            follower["initial_physical_state"][
                "native_lane_change_enabled"] = False
            followers.append(follower)

    scene_id = f"{scene_prefix}__prague_vodickova__{suffix}"
    asset = _scene(
        experiment_id, scene_id, topology,
        title=(
            "Single-width opposing corridor "
            f"[{ego_direction}, {str(suffix).replace('_', ' ')}]"),
        purpose=(
            "An oncoming vehicle already occupies a single-width shared "
            "corridor; ego must wait outside, then enter after the authored "
            "oncoming vehicle or platoon clears."),
        vehicles=(ego, peer, *followers),
        duration_s=85.0 if followers else 60.0,
        tags=(
            "narrow_road", "shared_bidirectional", "oncoming",
            "mutual_exclusion", "sequence_entry", "unsignalized",
            f"ego_{ego_direction}", str(suffix),
            *(("oncoming_platoon",) if followers else ())),
        factors=("chassis_profile", "oncoming_corridor_depth",
                 *(("oncoming_count",) if followers else ())),
        acceptance=(
            {"kind": "narrow_shared_corridor_sequence",
             "entity": "ego", "peer": "oncoming",
             "ego_direction": ego_direction,
             "ego_approach_lane_id": ego_route["approach_lane_id"],
             "ego_route_connector_ids": list(ego_route["connector_ids"]),
             "peer_entry_connector_id": peer_route["connector_ids"][0],
             "peer_initial_lane_id": peer_lane_id,
             "peer_initial_progress": float(peer_progress),
             "peer_route_connector_ids": list(peer_remaining_connectors),
             "corridor_segment_ids": [
                 topology.lane[lane_id]["segment_id"]
                 for lane_id in ego_route["shared_lane_ids"]],
             "ego_setback_m": float(ego_setback_m),
             "physical_width_m": 3.5,
             "peer_evaluated": bool(peer_evaluated)},
            *(tuple({"kind": "narrow_oncoming_platoon",
                "leader": "oncoming", "follower": follower["vehicle_id"],
                "follower_approach_lane_id": peer_route["approach_lane_id"],
                "follower_route_connector_ids": list(
                    peer_route["connector_ids"]),
                "follower_setback_m": follower_setback_m,
                "peer_evaluated": bool(peer_evaluated)}
                for follower, follower_setback_m in zip(
                    followers, follower_setbacks_m))),
        ),
        runtime_acceptance=(
            {"kind": "shared_corridor_entry_order", "order": [
                "oncoming", *[item["vehicle_id"] for item in followers],
                "ego"]},
            {"kind": "collision_free", "entities": [
                "ego", "oncoming",
                *[item["vehicle_id"] for item in followers]]},
            {"kind": "route_arrival", "entities": [
                "ego", "oncoming",
                *[item["vehicle_id"] for item in followers]]},
        ),
    )
    asset.scenario["experiment_scene"][
        "environment_schedule_basis_s"] = 45.0
    return asset


def _helsinki_narrow_road_sequence(topology: Topology) -> SceneAsset:
    """A second real single-width corridor, independent of the Prague site."""
    if topology.network != "helsinki_keskusta":
        raise ValueError("the expansion corridor is authored for Helsinki")
    ego_approach_lane_id = "n1371624183_n409705474::lane_1"
    ego_shared_lane_id = (
        "n1371624183_n1371708598::lane_shared_forward")
    peer_shared_lane_id = (
        "n1371624183_n1371708598::lane_shared_backward")
    ego_route_connector_ids = (
        "connector::n1371624183::0",
        "connector::n1371708598::3",
    )
    peer_entry_connector_id = "connector::n1371708598::7"
    peer_route_connector_ids = ("connector::n1371624183::2",)
    ego_exit_lane = topology.lane[
        topology.connector[ego_route_connector_ids[-1]]["to_lane"]]
    peer_exit_lane = topology.lane[
        topology.connector[peer_route_connector_ids[-1]]["to_lane"]]
    ego_lane = topology.lane[ego_approach_lane_id]
    ego_setback_m = 16.0
    peer_progress = 0.45

    ego = _vehicle(
        "ego", ego_lane, destination_node=ego_exit_lane["end_node"],
        progress=1.0 - ego_setback_m / ego_lane["length_m"],
        speed=18.0, evaluated=True)
    ego["initial_physical_state"]["lane_route_actions"] = [
        {"type": "connector", "connector_id": connector_id}
        for connector_id in ego_route_connector_ids]
    ego["initial_physical_state"]["native_lane_change_enabled"] = False

    peer_lane = topology.lane[peer_shared_lane_id]
    peer = _vehicle(
        "oncoming", peer_lane,
        destination_node=peer_exit_lane["end_node"],
        progress=peer_progress, speed=20.0, evaluated=False)
    peer["initial_physical_state"]["lane_route_actions"] = [
        {"type": "connector", "connector_id": connector_id}
        for connector_id in peer_route_connector_ids]
    peer["initial_physical_state"]["native_lane_change_enabled"] = False

    asset = _scene(
        "chassis", "chassis_narrow_road__expansion__helsinki_keskusta",
        topology, title="Single-width opposing corridor [Helsinki]",
        purpose=(
            "An oncoming vehicle is already committed to a mapped 3.5 m "
            "shared corridor; ego must wait on the ordinary two-way approach "
            "before entering."),
        vehicles=(ego, peer), duration_s=70.0,
        tags=("narrow_road", "shared_bidirectional", "oncoming",
              "mutual_exclusion", "sequence_entry", "unsignalized",
              "helsinki_corridor"),
        factors=("chassis_profile", "corridor_map"),
        acceptance=({
            "kind": "narrow_shared_corridor_sequence",
            "entity": "ego", "peer": "oncoming",
            "ego_direction": "forward",
            "ego_approach_lane_id": ego_approach_lane_id,
            "ego_route_connector_ids": list(ego_route_connector_ids),
            "peer_entry_connector_id": peer_entry_connector_id,
            "peer_initial_lane_id": peer_shared_lane_id,
            "peer_initial_progress": peer_progress,
            "peer_route_connector_ids": list(peer_route_connector_ids),
            "corridor_segment_ids": [
                topology.lane[ego_shared_lane_id]["segment_id"]],
            "ego_setback_m": ego_setback_m,
            "physical_width_m": 3.5,
            "peer_evaluated": False,
        },),
        runtime_acceptance=(
            {"kind": "shared_corridor_entry_order",
             "order": ["oncoming", "ego"]},
            {"kind": "collision_free", "entities": ["ego", "oncoming"]},
            {"kind": "route_arrival", "entities": ["ego", "oncoming"]},
        ),
    )
    asset.scenario["experiment_scene"][
        "environment_schedule_basis_s"] = 55.0
    return asset


def _connector_conflict(
    topology: Topology, experiment_id: str, scene_id: str, *,
    signalized: bool, title: str, factors: Sequence[str] = (),
    close: bool = False, turning: bool = False,
) -> SceneAsset:
    first, second = topology.conflict_pair(
        signalized=signalized, turning=turning,
        distinct_source_segments=True,
        minimum_source_length_m=(0.0 if close else 45.0))
    first_p, second_p = topology.conflict_progress(first, second)
    offset = 0.015 if close else 0.35
    if close:
        vehicles = [
            _connector_vehicle(topology, "ego", first,
                               progress=max(0.02, first_p - offset), speed=18.0),
            _connector_vehicle(topology, "cross_traffic", second,
                               progress=max(0.02, second_p - offset), speed=18.0,
                               evaluated=False),
        ]
    else:
        vehicles = [
            _approach_connector_vehicle(
                topology, "ego", first, distance_m=32.0, speed=18.0),
            _approach_connector_vehicle(
                topology, "cross_traffic", second, distance_m=35.0,
                speed=18.0, evaluated=False),
        ]
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="Two movements approach one HD-map connector conflict point.",
        vehicles=vehicles, duration_s=18.0,
        tags=("intersection", "signalized" if signalized else "unsignalized",
              "connector_conflict", *(('turning',) if turning else ())),
        factors=factors,
        acceptance=(
            {"kind": "connector_conflict", "entities": ["ego", "cross_traffic"]},
            {"kind": "signal_control", "value": signalized,
             "entities": ["ego", "cross_traffic"]},
        ))


def _unsignalized_challenge(
    topology: Topology, ego_connector_id: str, cross_connector_id: str,
    environment_basis_s: float,
) -> SceneAsset:
    """A minor approach yields to three successive priority-road vehicles."""
    ego = topology.connector[ego_connector_id]
    cross = topology.connector[cross_connector_id]
    vehicles = [_approach_connector_vehicle(
        topology, "ego", ego, distance_m=50, speed=25)]
    for index, distance in enumerate((35, 55, 75), 1):
        vehicles.append(_approach_connector_vehicle(
            topology, f"cross_{index}", cross, distance_m=distance, speed=30,
            evaluated=False))
    asset = _scene(
        "baseline", f"baseline_unsignalized_intersection__challenge__{topology.network}",
        topology, title=f"Unsignalized crossing: yield to successive traffic [{topology.network}]",
        purpose="Observe three successive priority-road vehicles, yield, then cross in a safe gap.",
        vehicles=vehicles, duration_s=environment_basis_s + 20,
        tags=("intersection", "unsignalized", "connector_conflict", "unsignalized_challenge",
              "crossing_stream", "gap_acceptance", "required_yield"),
        acceptance=(
            {"kind": "signal_control", "value": False,
             "entities": ["ego", "cross_1", "cross_2", "cross_3"]},
            *({"kind": "connector_conflict", "entities": ["ego", f"cross_{i}"]}
              for i in range(1, 4)),
            {"kind": "unsignalized_crossing_stream", "entity": "ego",
             "cross_entities": ["cross_1", "cross_2", "cross_3"],
             "ego_connector_id": ego_connector_id, "cross_connector_id": cross_connector_id,
             "minimum_crossing_angle_deg": 50, "ego_setback_m": 50,
             "cross_setbacks_m": [35, 55, 75]},
        ))
    asset.scenario["experiment_scene"]["environment_schedule_basis_s"] = environment_basis_s
    return asset


def _merge(topology: Topology, experiment_id: str, scene_id: str, *,
           title: str, factors: Sequence[str] = ()) -> SceneAsset:
    first, second = topology.merge_pair()
    vehicles = [
        _approach_connector_vehicle(
            topology, "ego", first, distance_m=28.0, speed=22.0),
        _approach_connector_vehicle(
            topology, "merging_vehicle", second, distance_m=32.0,
            speed=24.0, evaluated=False),
    ]
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="Two inbound movements converge onto the same target lane.",
        vehicles=vehicles, duration_s=18.0, tags=("merge", "gap_acceptance"),
        factors=factors,
        acceptance=(
            {"kind": "same_target_lane",
             "entities": ["ego", "merging_vehicle"]},
            {"kind": "merge_approach_arrival_window",
             "entities": ["ego", "merging_vehicle"],
             "setbacks_m": [28.0, 32.0],
             "maximum_approach_ttc_gap_s": 0.3},
        ),
        runtime_acceptance=(
            {"kind": "minimum_merge_encounters", "entity": "ego",
             "peer_ids": ["merging_vehicle"],
             "minimum_distinct_peers": 1, "distance_m": 30.0},
            {"kind": "merge_completed", "entity": "ego",
             "target_lane_id": first["to_lane"]},
            {"kind": "collision_free", "entities": [
                "ego", "merging_vehicle"]},
            {"kind": "route_arrival", "entities": [
                "ego", "merging_vehicle"]},
        ))


def _four_way(topology: Topology, experiment_id: str, scene_id: str, *,
    title: str, factors: Sequence[str] = (),
    synchronized: bool = False) -> SceneAsset:
    connectors = topology.four_way_connectors()
    vehicles = []
    setbacks = (
        (34.0, 37.0, 40.0, 43.0) if synchronized
        else (24.0, 42.0, 60.0, 78.0))
    for index, connector in enumerate(connectors):
        source = topology.lane[connector["from_lane"]]
        target = topology.lane[connector["to_lane"]]
        vehicle = _vehicle(
            "ego" if index == 0 else f"approach_{index + 1}",
            source, destination_node=target["end_node"],
            # Keep four genuine conflicting approaches, but give the SUMO
            # reference policy distinct arrival times. LLM-controlled runs
            # receive the same physical starting state and may still choose
            # unsafe commands; this only avoids an artificial simultaneous
            # spawn inside a short junction approach.
            progress=max(
                0.05,
                1.0 - setbacks[index] / source["length_m"]),
            speed=18.0, evaluated=index == 0)
        vehicle["initial_physical_state"]["lane_route_actions"] = [{
            "type": "connector", "connector_id": connector["id"],
        }]
        vehicles.append(vehicle)
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="Four autonomous approaches negotiate one unsignalized junction.",
        vehicles=vehicles, duration_s=28.0,
        tags=("unsignalized", "four_way", "connector_conflict",
              *(("synchronized_arrival", "multi_agent_negotiation")
                if synchronized else ())),
        factors=factors,
        acceptance=(
            {"kind": "same_junction_minimum_conflicts",
             "entities": [item["vehicle_id"] for item in vehicles],
             "minimum_conflicting_pairs": 3},
            *(({"kind": "synchronized_unsignalized_arrivals",
                "entities": [item["vehicle_id"] for item in vehicles],
                "setbacks_m": list(setbacks), "speed_kmh": 18.0,
                "maximum_arrival_spread_s": 2.0},)
              if synchronized else ()),
        ),
        runtime_acceptance=(
            *(({"kind": "minimum_connector_conflicts",
                "entity": "ego",
                "peer_ids": [item["vehicle_id"] for item in vehicles[1:]],
                "minimum_distinct_peers": 2},)
              if synchronized else ()),
            {"kind": "collision_free", "entities": [
                item["vehicle_id"] for item in vehicles]},
            {"kind": "route_arrival", "entities": [
                item["vehicle_id"] for item in vehicles]},
        ) if synchronized else ())


def _crosswalk(topology: Topology, experiment_id: str, scene_id: str, *,
               title: str, turning: bool = False,
               factors: Sequence[str] = (),
               vehicle_progress_offset: float = 0.40,
               initial_signal: str | None = None,
               required_connector_id: str | None = None) -> SceneAsset:
    yield_baseline = scene_id.split("__")[0] == "baseline_crosswalk"
    crosswalk, connector = topology.crosswalk_conflict(
        turning=turning, initial_signal=initial_signal,
        unsignalized=yield_baseline,
        required_connector_id=(
            required_connector_id
            if required_connector_id is not None
            else PEDESTRIAN_YIELD_CONNECTORS[topology.network]
            if yield_baseline else None))
    progress = topology.crosswalk_progress(crosswalk, connector)
    source = topology.lane[connector["from_lane"]]
    target = topology.lane[connector["to_lane"]]
    vehicles = [(
        _connector_vehicle(
            topology, "ego", connector, progress=progress, speed=18.0)
        if vehicle_progress_offset <= 0.0
        else _approach_connector_vehicle(
            topology, "ego", connector, distance_m=35.0 if yield_baseline else 24.0, speed=18.0)
    )]
    pedestrian_start = source["end_node"]
    pedestrian_destination = target["start_node"]
    if pedestrian_destination == pedestrian_start:
        # Some clustered intersections expose the same routing node on both
        # sides of a connector.  The physical crosswalk is still valid, but
        # a one-node pedestrian route is treated as already arrived and the
        # policy is never exercised.  Use the distinct downstream node as the
        # logical completion token; motion still follows the authored path.
        pedestrian_destination = target["end_node"]
    pedestrians = [_pedestrian(
        "pedestrian", initial_node=pedestrian_start,
        destination_node=pedestrian_destination,
        speed=1.4, crosswalk=crosswalk,
        progress=topology.pedestrian_crosswalk_progress(
            crosswalk, connector))]
    if yield_baseline:
        pedestrians[0]["initial_physical_state"].update({
            "progress": 0.0, "waiting": True,
            "crossing_trigger": {"vehicle_id": "ego", "ttc_s": 4.0},
        })
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=("Yield to a pedestrian released at ego route TTC < 4 s, without traffic signals."
                 if yield_baseline else "Vehicle trajectory intersects an occupied mapped crosswalk."),
        vehicles=vehicles, pedestrians=pedestrians,
        duration_s=((90.0 if topology.network == "newyork_manhattan_mid" else 60.0)
                    if yield_baseline else 18.0),
        tags=("crosswalk", "vehicle_pedestrian",
              *(("unsignalized", "pedestrian_yield", "ttc_triggered") if yield_baseline else ()),
              "turning" if turning else "through",
              *((f"initial_{initial_signal}",)
                if initial_signal is not None else ())),
        factors=factors,
        acceptance=(
            {"kind": "connector_crosswalk_conflict",
             "vehicle": "ego", "pedestrian": "pedestrian"},
            *(({"kind": "unsignalized_pedestrian_ttc",
                "vehicle": "ego", "pedestrian": "pedestrian", "ttc_s": 4.0},)
              if yield_baseline else ()),
            *(({"kind": "initial_signal", "connector_id": connector["id"],
                "value": initial_signal},)
              if initial_signal is not None else ()),
        ))


def _crowded_unsignalized_crosswalk(
    topology: Topology, experiment_id: str, scene_id: str, *, title: str,
    required_connector_id: str | None = None,
) -> SceneAsset:
    """Release a staggered eight-person stream across ego's approach road."""
    crosswalk, connector = topology.crosswalk_conflict(
        unsignalized=True,
        required_connector_id=(
            required_connector_id
            if required_connector_id is not None
            else PEDESTRIAN_YIELD_CONNECTORS[topology.network]))
    source = topology.lane[connector["from_lane"]]
    target = topology.lane[connector["to_lane"]]
    ego = _approach_connector_vehicle(
        topology, "ego", connector, distance_m=45.0, speed=20.0)
    pedestrian_start = source["end_node"]
    pedestrian_destination = target["start_node"]
    if pedestrian_destination == pedestrian_start:
        pedestrian_destination = target["end_node"]
    thresholds = (7.0, 6.4, 5.8, 5.2, 4.6, 4.0, 3.4, 2.8)
    speeds = (1.10, 1.25, 1.35, 1.20, 1.40, 1.15, 1.30, 1.20)
    pedestrians = []
    for index, (threshold, speed) in enumerate(
            zip(thresholds, speeds), 1):
        pedestrian = _pedestrian(
            f"pedestrian_{index}", initial_node=pedestrian_start,
            destination_node=pedestrian_destination, speed=speed,
            crosswalk=crosswalk, progress=0.0)
        pedestrian["initial_physical_state"].update({
            "waiting": True,
            "crossing_trigger": {
                "vehicle_id": "ego", "ttc_s": threshold,
            },
        })
        pedestrians.append(pedestrian)
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "Eight pedestrians enter one unsignalized crossing in a "
            "staggered stream as ego approaches, creating sustained occupancy."),
        vehicles=[ego], pedestrians=pedestrians,
        duration_s=(90.0 if topology.network
                    == "newyork_manhattan_mid" else 65.0),
        tags=("crosswalk", "unsignalized", "pedestrian_group",
              "crowded_crossing", "staggered_ttc_release",
              "sustained_occupancy"),
        factors=("pedestrian_count", "release_span"),
        acceptance=(
            {"kind": "connector_crosswalk_conflict",
             "vehicle": "ego", "pedestrian": "pedestrian_1"},
            {"kind": "dense_unsignalized_pedestrian_group",
             "vehicle": "ego",
             "pedestrian_ids": [p["ped_id"] for p in pedestrians],
             "crosswalk_id": crosswalk["id"],
             "connector_id": connector["id"],
             "ttc_thresholds_s": list(thresholds),
             "minimum_pedestrian_count": 8,
             "minimum_release_span_s": 4.0},
        ),
        runtime_acceptance=(
            {"kind": "minimum_pedestrian_releases",
             "pedestrian_ids": [p["ped_id"] for p in pedestrians],
             "minimum_count": 8},
            {"kind": "minimum_simultaneous_crosswalk_occupancy",
             "crosswalk_id": crosswalk["id"], "minimum_count": 4},
            {"kind": "vehicle_pedestrian_collision_free",
             "vehicle": "ego",
             "pedestrian_ids": [p["ped_id"] for p in pedestrians]},
            {"kind": "route_arrival", "entities": ["ego"]},
        ),
    )


def _mapped_unsignalized_crossing(
    topology: Topology, experiment_id: str, scene_id: str, *,
    title: str, factors: Sequence[str] = (),
    initial_signal: str | None = None,
) -> SceneAsset:
    """Build a risky crossing whose complete motion is executable by SUMO."""
    return _crosswalk(
        topology, experiment_id, scene_id,
        title=title, factors=factors,
        initial_signal=initial_signal,
    )


def _signal_queue(topology: Topology, experiment_id: str, scene_id: str, *,
                  title: str, factors: Sequence[str] = ()) -> SceneAsset:
    connector = topology.red_connector()
    lane = topology.lane[connector["from_lane"]]
    stop = topology.runtime.stop_progress(lane["id"])
    spacing = 9.0 / lane["length_m"]
    target = topology.lane[connector["to_lane"]]["end_node"]
    vehicles = [
        _vehicle(
            f"queue_{index + 1}", lane, destination_node=target,
            progress=max(0.05, stop - spacing * index), speed=0.0,
            evaluated=index == 0)
        for index in range(3)
    ]
    for vehicle in vehicles:
        vehicle["initial_physical_state"]["desired_speed_kmh"] = 32.0
        vehicle["initial_physical_state"]["lane_route_actions"] = [{
            "type": "connector",
            "connector_id": connector["id"],
        }]
    initial_signal = topology.runtime.signal_state(connector["id"], 0.0)
    red_wait_s = float(initial_signal.remaining_seconds)
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="A stopped queue waits for a connector phase and starts in order.",
        # Observe the complete three-vehicle discharge, but end before a long
        # target lane can carry the same vehicles into an unrelated second
        # signal cycle.  Twenty seconds after the first green is sufficient
        # for this queue-start protocol on the validated map portfolio.
        vehicles=vehicles, duration_s=max(35.0, red_wait_s + 20.0),
        tags=("signal", "red_stop", "queue_discharge"), factors=factors,
        acceptance=(
            {"kind": "initial_signal", "connector_id": connector["id"],
             "value": "red"},
            {"kind": "same_lane", "entities": [
                "queue_1", "queue_2", "queue_3"]},
        ))


def _signal_queue_middle(topology: Topology, connector_id: str) -> SceneAsset:
    connector = topology.connector[connector_id]
    lane = topology.lane[connector["from_lane"]]
    stop = topology.runtime.stop_progress(lane["id"])
    destination = topology.lane[connector["to_lane"]]["end_node"]
    vehicles = []
    # Array order selects the focal LLM; physical progress determines queue order.
    for vid, setback, evaluated in (("ego", 13, True), ("front", 4, False), ("rear", 22, False)):
        vehicle = _vehicle(vid, lane, destination_node=destination,
                           progress=stop - setback / lane["length_m"], speed=0,
                           evaluated=evaluated)
        vehicle["initial_physical_state"]["lane_route_actions"] = [
            {"type": "connector", "connector_id": connector_id}]
        vehicles.append(vehicle)
    return _scene(
        "baseline", f"baseline_signalized_intersection__middle__{topology.network}",
        topology, title=f"Signal queue: evaluated vehicle in the middle [{topology.network}]",
        purpose="Wait behind the front vehicle at red, then follow it through green without collision.",
        vehicles=vehicles, duration_s=90,
        tags=("signal", "red_stop", "queue_discharge", "queue_middle", "car_following"),
        acceptance=(
            {"kind": "initial_signal", "connector_id": connector_id, "value": "red"},
            {"kind": "same_lane", "entities": ["front", "ego", "rear"]},
            {"kind": "middle_signal_queue", "entity": "ego", "front": "front", "rear": "rear",
             "connector_id": connector_id, "spacing_m": 9.0},
        ))


def _signal_approach_stop(
    topology: Topology, experiment_id: str, scene_id: str, *,
    title: str, factors: Sequence[str] = (),
    connector_id: str | None = None,
    approach_distance_m: float | None = None,
    red_wait_range_s: tuple[float, float] | None = None,
    environment_basis_s: float | None = None,
    duration_s: float | None = None,
) -> SceneAsset:
    """A moving vehicle must brake for a red phase, then depart on green."""
    connector = (
        topology.red_connector(longest_wait=True)
        if connector_id is None else topology.connector[connector_id]
    )
    lane = topology.lane[connector["from_lane"]]
    stop = topology.runtime.stop_progress(lane["id"])
    target = topology.lane[connector["to_lane"]]["end_node"]
    selected_approach_distance_m = (
        min(38.0, max(24.0, lane["length_m"] * 0.45))
        if approach_distance_m is None else float(approach_distance_m)
    )
    vehicle = _vehicle(
        "ego", lane, destination_node=target,
        progress=max(
            0.05,
            stop - selected_approach_distance_m / lane["length_m"],
        ),
        speed=32.0, evaluated=True)
    vehicle["initial_physical_state"]["desired_speed_kmh"] = 32.0
    vehicle["initial_physical_state"]["lane_route_actions"] = [{
        "type": "connector", "connector_id": connector["id"],
    }]
    initial_signal = topology.runtime.signal_state(connector["id"], 0.0)
    if initial_signal is None or initial_signal.signal != "red":
        raise ValueError(
            f"{topology.network} connector {connector['id']} is not red at t=0")
    red_wait_s = float(initial_signal.remaining_seconds)
    if red_wait_range_s is not None and not (
            float(red_wait_range_s[0]) <= red_wait_s
            <= float(red_wait_range_s[1])):
        raise ValueError(
            f"{topology.network} connector {connector['id']} has "
            f"{red_wait_s:.3f} s of red remaining, outside "
            f"{red_wait_range_s}")
    assertions = [{
        "kind": "initial_signal", "connector_id": connector["id"],
        "value": "red",
    }]
    if red_wait_range_s is not None:
        assertions.append({
            "kind": "bounded_red_light_approach",
            "entity": "ego",
            "connector_id": connector["id"],
            "turn": connector["turn"],
            "minimum_red_remaining_s": float(red_wait_range_s[0]),
            "maximum_red_remaining_s": float(red_wait_range_s[1]),
            "minimum_approach_distance_m": 30.0,
            "maximum_approach_distance_m": 40.0,
            "minimum_initial_speed_kmh": 30.0,
            "maximum_initial_speed_kmh": 35.0,
            "maximum_exit_lane_length_m": 100.0,
        })
    asset = _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=(
            "A vehicle approaches an active red phase at road speed, brakes "
            "before the stop line, and departs after green."),
        vehicles=[vehicle], duration_s=(
            max(40.0, red_wait_s + 25.0)
            if duration_s is None else float(duration_s)),
        tags=("signal", "red_stop", "moving_approach", "braking"),
        factors=factors,
        acceptance=assertions)
    if environment_basis_s is not None:
        asset.scenario["experiment_scene"][
            "environment_schedule_basis_s"] = float(environment_basis_s)
    return asset


def _lane_change(topology: Topology, experiment_id: str, scene_id: str, *,
                 title: str, factors: Sequence[str] = ()) -> SceneAsset:
    lane = topology.long_lane(min_length=150.0, min_siblings=3, ordinal=1)
    siblings = topology.sibling_lanes(lane)
    middle = siblings[len(siblings) // 2]
    adjacent = min(
        (item for item in siblings if item["id"] != middle["id"]),
        key=lambda item: abs(item["index"] - middle["index"]))
    vehicles = [
        _vehicle("ego", middle, progress=0.30, speed=34.0),
        _vehicle("slow_lead", middle, progress=0.47, speed=12.0,
                 evaluated=False),
        _vehicle("adjacent_vehicle", adjacent, progress=0.39, speed=29.0,
                 evaluated=False),
    ]
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="A slower lead creates a lane-change decision with an occupied gap.",
        vehicles=vehicles, duration_s=22.0,
        tags=("lane_change", "gap_acceptance", "side_collision_risk"),
        factors=factors,
        acceptance=(
            {"kind": "same_lane", "entities": ["ego", "slow_lead"]},
            {"kind": "adjacent_lane", "entities": [
                "ego", "adjacent_vehicle"]},
        ))


def _required_turn(topology: Topology, initial_id: str, connector_id: str) -> SceneAsset:
    lane = topology.lane[initial_id]
    connector = topology.connector[connector_id]
    target = topology.lane[connector["from_lane"]]
    exit_lane = topology.lane[connector["to_lane"]]
    vehicle = _vehicle("ego", lane, destination_node=exit_lane["end_node"],
                       progress=1 - 100 / lane["length_m"], speed=30)
    vehicle["initial_physical_state"]["lane_route_actions"] = [
        {"type": "lane_change", "from_lane_id": lane["id"],
         "to_lane_id": target["id"], "target_lane_index": target["index"]},
        {"type": "connector", "from_lane_id": target["id"],
         "to_lane_id": exit_lane["id"], "connector_id": connector_id,
         "turn": connector["turn"]},
    ]
    return _scene(
        "baseline", f"baseline_required_turn__{topology.network}", topology,
        title=f"Lane change before {connector['turn']} turn [{topology.network}]",
        purpose=("Start in a straight-only lane. Change into the adjacent turn-capable "
                 "lane and take the specified turn to reach the destination."),
        vehicles=[vehicle], duration_s=120,
        tags=("lane_change", "required_turn", connector["turn"]),
        acceptance=({"kind": "required_lane_change_turn", "entity": "ego",
                     "target_lane_id": target["id"], "connector_id": connector_id},))


def _crash_obstacle(topology: Topology, experiment_id: str, scene_id: str, *,
                    title: str, factors: Sequence[str] = ()) -> SceneAsset:
    lane = topology.long_lane(min_length=150.0, min_siblings=3, ordinal=3)
    vehicles = [
        _vehicle("ego", lane, progress=0.28, speed=34.0),
        _vehicle("crashed_vehicle", lane, progress=0.58, speed=0.0,
                 crashed=True, evaluated=False),
    ]
    if experiment_id == "multi":
        adjacent = min(
            (item for item in topology.sibling_lanes(lane)
             if item["id"] != lane["id"]),
            key=lambda item: abs(item["index"] - lane["index"]))
        vehicles.insert(1, _vehicle(
            "adjacent_peer", adjacent, progress=0.38, speed=27.0,
            evaluated=False))
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="A crashed vehicle blocks one lane while adjacent lanes remain usable.",
        vehicles=vehicles, duration_s=24.0,
        tags=("crash_obstacle", "lane_change", "reroute"), factors=factors,
        acceptance=(
            {"kind": "crashed_lane_obstacle", "entity": "crashed_vehicle"},
            {"kind": "minimum_parallel_lanes", "lane_id": lane["id"],
             "count": 3},
        ))


def _multi_vehicle(topology: Topology, experiment_id: str, scene_id: str, *,
                   title: str, factors: Sequence[str] = (),
                   count: int = 12,
                   oncoming_stream: Mapping[str, Any] | None = None) -> SceneAsset:
    candidates = topology.connector_chains(
        count=max(count * 4, 40), minimum_connectors=4)
    candidates.sort(key=lambda item: hashlib.sha256(
        f"{scene_id}:{item[0]['id']}".encode()).hexdigest())
    vehicles = []
    accepted_route_lanes = []
    for lane, actions in candidates:
        if len(vehicles) >= count:
            break
        route_lanes = topology.route_lane_ids(lane, actions)
        if (topology.route_has_opposing_overlap(
                route_lanes, accepted_route_lanes)
                or topology.route_has_unmodeled_crossing(
                    route_lanes, accepted_route_lanes)):
            continue
        index = len(vehicles)
        vehicle = _vehicle(
            "ego" if index == 0 else f"traffic_{index:02d}", lane,
            destination_node=topology.lane[
                actions[-1]["to_lane"]]["end_node"],
            progress=0.08 + (index % 4) * 0.08,
            speed=22.0 + (index % 3) * 4.0,
            evaluated=index == 0)
        vehicle["initial_physical_state"]["lane_route_actions"] = [{
            "type": "connector",
            "connector_id": action["id"],
        } for action in actions]
        vehicles.append(vehicle)
        accepted_route_lanes.extend(route_lanes)
    if len(vehicles) < count:
        raise ValueError(
            f"{topology.network} supplies only {len(vehicles)}/{count} "
            f"non-overlapping routes for {scene_id}")
    tags = ["full_network", "multi_vehicle", "traffic_load"]
    acceptance = [{"kind": "minimum_distinct_segments", "count": 8}]
    duration_s = 60.0
    purpose = "Distributed routes exercise the full map under concurrent traffic."
    if oncoming_stream is not None:
        lane = topology.lane[str(oncoming_stream["lane_id"])]
        progresses = [float(value) for value in oncoming_stream["progresses"]]
        release_speed = float(oncoming_stream["release_speed_kmh"])
        trigger_distance = 70.0
        vehicles = vehicles[:count - len(progresses)]
        peer_ids = []
        for index, progress in enumerate(progresses, start=1):
            peer_id = f"oncoming_{index:02d}"
            peer_ids.append(peer_id)
            vehicle = _vehicle(
                peer_id, lane, progress=progress, speed=release_speed,
                evaluated=False)
            vehicles.append(vehicle)
        tags.extend(("ego_centric_oncoming", "sumo_native_oncoming"))
        purpose = (
            "A long ego route includes three freely moving SUMO oncoming "
            "vehicles while remaining traffic covers the wider map.")
        duration_s = float(oncoming_stream["duration_s"])
        acceptance.append({
            "kind": "native_full_network_oncoming_stream",
            "entity": "ego",
            "peer_ids": peer_ids,
            "lane_id": lane["id"],
            "progresses": progresses,
            "trigger_distance_m": trigger_distance,
            "release_speed_kmh": release_speed,
            "minimum_runtime_encounters": len(peer_ids),
            "minimum_initial_spacing_m": 25.0,
            "maximum_lateral_separation_m": 20.0,
            "minimum_heading_difference_deg": 120.0,
        })
    asset = _scene(
        experiment_id, scene_id, topology, title=title,
        purpose=purpose, vehicles=vehicles, duration_s=duration_s,
        tags=tags,
        factors=factors,
        acceptance=acceptance)
    if oncoming_stream is not None:
        asset.scenario["experiment_scene"]["environment_schedule_basis_s"] = float(
            oncoming_stream["environment_basis_s"])
    return asset


def _continuous_turn_route(
    topology: Topology, experiment_id: str, scene_id: str, *,
    title: str, factors: Sequence[str] = (),
) -> SceneAsset:
    lane, actions = topology.connector_chains(
        count=1, minimum_connectors=3, minimum_turns=2)[0]
    vehicle = _vehicle(
        "ego", lane,
        destination_node=topology.lane[actions[-1]["to_lane"]]["end_node"],
        progress=0.55, speed=25.0)
    vehicle["initial_physical_state"]["lane_route_actions"] = [{
        "type": "connector", "connector_id": action["id"],
    } for action in actions]
    return _scene(
        experiment_id, scene_id, topology, title=title,
        purpose="One fixed HD-map route contains consecutive turning connectors.",
        vehicles=[vehicle], duration_s=50.0,
        tags=("continuous_turns", "route_following"), factors=factors,
        acceptance=({
            "kind": "minimum_route_connectors", "entity": "ego",
            "count": 3, "minimum_turns": 2,
        },))


def _compact_turn_route(
    topology: Topology, initial_lane_id: str, connector_ids: Sequence[str],
    environment_basis_s: float,
) -> SceneAsset:
    """Three separate turning junctions with short connecting road legs."""
    lane = topology.lane[initial_lane_id]
    last = topology.connector[connector_ids[-1]]
    vehicle = _vehicle(
        "ego", lane, destination_node=topology.lane[last["to_lane"]]["end_node"],
        progress=1 - 60 / lane["length_m"], speed=25)
    vehicle["initial_physical_state"]["lane_route_actions"] = [
        {"type": "connector", "connector_id": cid} for cid in connector_ids]
    asset = _scene(
        "chassis", f"chassis_continuous_turns__compact__{topology.network}", topology,
        title=f"Compact three-turn route [{topology.network}]",
        purpose="Complete three separate nearby turns, including a direction reversal, without a shortcut.",
        vehicles=[vehicle], duration_s=environment_basis_s + 20,
        tags=("continuous_turns", "route_following", "compact_turns", "alternating_direction"),
        acceptance=(
            {"kind": "minimum_route_connectors", "entity": "ego", "count": 3, "minimum_turns": 3},
            {"kind": "compact_turn_route", "entity": "ego", "connector_ids": list(connector_ids),
             "initial_setback_m": 60, "minimum_gap_m": 40, "maximum_gap_m": 140},
        ))
    asset.scenario["experiment_scene"]["environment_schedule_basis_s"] = environment_basis_s
    return asset


def _augment_background_vehicles(
    topology: Topology, asset: SceneAsset, *, total: int = 9,
) -> SceneAsset:
    existing = asset.scenario["vehicles"]
    used_segments = set()
    accepted_route_lanes = []
    for vehicle in existing:
        state = vehicle.get("initial_physical_state", {})
        if "lane_id" in state:
            used_segments.add(topology.lane[state["lane_id"]]["segment_id"])
            accepted_route_lanes.append(state["lane_id"])
        elif "connector_id" in state:
            connector = topology.connector[state["connector_id"]]
            used_segments.add(topology.lane[
                connector["from_lane"]]["segment_id"])
    chains = topology.connector_chains(
        count=max(total * 2, 20), minimum_connectors=3)
    for lane, actions in chains:
        if len(existing) >= total:
            break
        if lane["segment_id"] in used_segments:
            continue
        route_lanes = topology.route_lane_ids(lane, actions)
        if (topology.route_has_opposing_overlap(
                route_lanes, accepted_route_lanes)
                or topology.route_has_unmodeled_crossing(
                    route_lanes, accepted_route_lanes)):
            continue
        index = len(existing)
        vehicle = _vehicle(
            f"background_{index:02d}", lane,
            destination_node=topology.lane[
                actions[-1]["to_lane"]]["end_node"],
            progress=0.08 + (index % 3) * 0.08,
            speed=22.0 + (index % 2) * 4.0,
            evaluated=False)
        vehicle["initial_physical_state"]["lane_route_actions"] = [{
            "type": "connector", "connector_id": action["id"],
        } for action in actions]
        existing.append(vehicle)
        used_segments.add(lane["segment_id"])
        accepted_route_lanes.extend(route_lanes)
    if len(existing) < total:
        raise ValueError(
            f"could not augment {asset.scene_id} to {total} vehicles")
    asset.expected["setup_assertions"][0]["vehicles"] = len(existing)
    return asset


def _augment_background_pedestrians(
    topology: Topology, asset: SceneAsset, *, total: int = 8,
) -> SceneAsset:
    existing = asset.scenario["pedestrians"]
    used = {
        item.get("initial_physical_state", {}).get("crosswalk_id")
        for item in existing}
    for crosswalk in topology.crosswalks:
        if len(existing) >= total:
            break
        if crosswalk["id"] in used or not crosswalk.get("conflicting_connectors"):
            continue
        connector = topology.connector.get(
            crosswalk["conflicting_connectors"][0])
        if not connector:
            continue
        source = topology.lane[connector["from_lane"]]
        target = topology.lane[connector["to_lane"]]
        existing.append(_pedestrian(
            f"background_ped_{len(existing):02d}",
            initial_node=source["end_node"],
            destination_node=target["start_node"],
            speed=1.2 + (len(existing) % 3) * 0.15,
            crosswalk=crosswalk, progress=0.05 + (len(existing) % 4) * 0.12))
        used.add(crosswalk["id"])
    if len(existing) < total:
        raise ValueError(
            f"could not augment {asset.scene_id} to {total} pedestrians")
    asset.expected["setup_assertions"][0]["pedestrians"] = len(existing)
    return asset


def _refresh_scene_geometry(asset: SceneAsset) -> None:
    geometry = asset.scenario["experiment_scene"]["geometry"]
    geometry["lane_ids"] = sorted({
        state["initial_physical_state"]["lane_id"]
        for state in asset.scenario["vehicles"]
        if "lane_id" in state.get("initial_physical_state", {})})
    geometry["connector_ids"] = sorted({
        connector_id
        for state in asset.scenario["vehicles"]
        for connector_id in (
            ([state["initial_physical_state"]["connector_id"]]
             if "connector_id" in state.get("initial_physical_state", {})
             else [])
            + [action["connector_id"] for action in state.get(
                "initial_physical_state", {}).get("lane_route_actions", [])
               if action.get("type") == "connector"])
    })
    geometry["crosswalk_ids"] = sorted({
        state["initial_physical_state"]["crosswalk_id"]
        for state in asset.scenario["pedestrians"]
        if "crosswalk_id" in state.get("initial_physical_state", {})})


SCENE_FAMILY_TEMPLATES = {
    "baseline": (
        "baseline_straight_following", "baseline_signalized_intersection",
        "baseline_unsignalized_intersection", "baseline_lane_change", "baseline_crosswalk"),
    "traffic": (
        "traffic_following_pressure", "traffic_merge_gap",
        "traffic_intersection_conflict", "traffic_lane_change_pressure"),
    "pedestrian": (
        "pedestrian_signalized_crosswalk", "pedestrian_unsignalized_crossing",
        "pedestrian_turning_crosswalk", "pedestrian_unmarked_jaywalk"),
    "cabin": (
        "cabin_common_climate_task", "cabin_available_feature_task",
        "cabin_capability_probe"),
    "chassis": (
        "chassis_red_light_stop", "chassis_lead_vehicle_braking",
        "chassis_target_lane_gap", "chassis_continuous_turns", "chassis_narrow_road",
        "chassis_full_network_route"),
    "multi": (
        "multi_unsignalized_four_way", "multi_signal_queue_start",
        "multi_two_lane_merge", "multi_crosswalk_right_turn",
        "multi_crash_obstacle_bypass", "multi_whole_map_traffic"),
    "concurrent": (
        "concurrent_intersection_dual_task", "concurrent_crosswalk_dual_task",
        "concurrent_obstacle_dual_task"),
}


def _portfolio_scene_id(template_id: str, topology: Topology) -> str:
    return f"{template_id}__{topology.network}"


def _portfolio_title(title: str, topology: Topology) -> str:
    return f"{title} [{topology.network}]"


def _tag_portfolio_scene(asset: SceneAsset, template_id: str,
                         *, role: str) -> SceneAsset:
    asset.scenario["experiment_scene"].update({
        "scene_template_id": template_id,
        "portfolio_network_id": asset.scenario["road_network_id"],
        "portfolio_role": role,
    })
    return asset


def _build_portfolio_template(
    template_id: str, topology: Topology,
) -> SceneAsset:
    """Instantiate one canonical template against another real HD map."""
    experiment_id = template_id.split("_", 1)[0]
    scene_id = _portfolio_scene_id(template_id, topology)

    if template_id == "baseline_straight_following":
        asset = _following(
            topology, experiment_id, scene_id, collision=False,
            title=_portfolio_title("Straight-road baseline", topology))
    elif template_id == "baseline_signalized_intersection":
        asset = _signal_queue(
            topology, experiment_id, scene_id,
            title=_portfolio_title(
                "Signalized-intersection baseline", topology))
    elif template_id == "baseline_unsignalized_intersection":
        asset = _connector_conflict(
            topology, experiment_id, scene_id, signalized=False,
            title=_portfolio_title(
                "Unsignalized-intersection baseline", topology))
    elif template_id == "baseline_lane_change":
        asset = _lane_change(
            topology, experiment_id, scene_id,
            title=_portfolio_title("Lane-change baseline", topology))
    elif template_id == "baseline_crosswalk":
        asset = _crosswalk(
            topology, experiment_id, scene_id,
            title=_portfolio_title("Crosswalk baseline", topology))
    elif template_id.startswith("traffic_"):
        factors = ("background_density",)
        if template_id == "traffic_following_pressure":
            asset = _following(
                topology, experiment_id, scene_id, collision=False,
                title=_portfolio_title(
                    "Following under dense traffic", topology),
                factors=factors)
        elif template_id == "traffic_merge_gap":
            asset = _merge(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Merge under dense traffic", topology),
                factors=factors)
        elif template_id == "traffic_intersection_conflict":
            asset = _connector_conflict(
                topology, experiment_id, scene_id, signalized=False,
                title=_portfolio_title(
                    "Intersection under dense traffic", topology),
                factors=factors)
        else:
            asset = _lane_change(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Lane change under dense traffic", topology),
                factors=factors)
        asset = _augment_background_vehicles(topology, asset)
    elif template_id.startswith("pedestrian_"):
        factors = ("pedestrian_density",)
        if template_id == "pedestrian_signalized_crosswalk":
            asset = _crosswalk(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Signalized crosswalk behavior", topology),
                factors=factors,
                initial_signal="green")
        elif template_id == "pedestrian_unsignalized_crossing":
            asset = _mapped_unsignalized_crossing(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Unsignalized road crossing", topology),
                factors=factors)
        elif template_id == "pedestrian_turning_crosswalk":
            asset = _crosswalk(
                topology, experiment_id, scene_id, turning=True,
                title=_portfolio_title(
                    "Turning vehicle and pedestrian", topology),
                factors=factors)
        else:
            asset = _mapped_unsignalized_crossing(
                topology, experiment_id, scene_id,
                title=_portfolio_title("Unmarked jaywalking", topology),
                factors=factors)
        asset = _augment_background_pedestrians(topology, asset)
    elif template_id.startswith("cabin_"):
        factors = ("equipment_profile", "request_support")
        if template_id == "cabin_common_climate_task":
            asset = _following(
                topology, experiment_id, scene_id, collision=False,
                title=_portfolio_title("Common equipment request", topology),
                factors=factors)
        else:
            asset = _multi_vehicle(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Available equipment request in traffic"
                    if template_id == "cabin_available_feature_task"
                    else "Unsupported capability probe", topology),
                factors=factors, count=8)
    elif template_id.startswith("chassis_"):
        factors = ("chassis_profile",)
        if template_id == "chassis_red_light_stop":
            asset = _signal_approach_stop(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Chassis red-light braking", topology), factors=factors)
        elif template_id == "chassis_lead_vehicle_braking":
            asset = _lead_vehicle_braking(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Chassis emergency following", topology), factors=factors)
        elif template_id == "chassis_target_lane_gap":
            asset = _lane_change(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Chassis lane-change gap", topology), factors=factors)
        elif template_id == "chassis_continuous_turns":
            asset = _continuous_turn_route(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Chassis continuous-turn response", topology),
                factors=factors)
        elif template_id == "chassis_narrow_road":
            asset = _following(
                topology, experiment_id, scene_id, collision=False,
                title=_portfolio_title(
                    "Chassis narrow-road following", topology),
                factors=factors)
        else:
            asset = _multi_vehicle(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Chassis full-network route", topology), factors=factors,
                oncoming_stream=FULL_NETWORK_ONCOMING_SITES[topology.network])
    elif template_id.startswith("multi_"):
        factors = ("llm_count", "model_assignment")
        if template_id == "multi_unsignalized_four_way":
            asset = _four_way(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Multi-agent four-way conflict", topology), factors=factors)
        elif template_id == "multi_signal_queue_start":
            asset = _signal_queue(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Multi-agent signal queue start", topology), factors=factors)
        elif template_id == "multi_two_lane_merge":
            asset = _merge(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Multi-agent two-lane merge", topology), factors=factors)
        elif template_id == "multi_crosswalk_right_turn":
            asset = _crosswalk(
                topology, experiment_id, scene_id, turning=True,
                title=_portfolio_title(
                    "Multi-agent crosswalk right turn", topology),
                factors=factors)
        elif template_id == "multi_crash_obstacle_bypass":
            asset = _crash_obstacle(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Multi-agent crash obstacle bypass", topology),
                factors=factors)
        else:
            asset = _multi_vehicle(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Whole-map multi-agent traffic", topology),
                factors=factors, count=16)
    elif template_id.startswith("concurrent_"):
        factors = ("dual_domain_condition", "cabin_task_load", "event_anchor")
        if template_id == "concurrent_intersection_dual_task":
            asset = _connector_conflict(
                topology, experiment_id, scene_id, signalized=False,
                title=_portfolio_title(
                    "Intersection conflict with cabin task", topology),
                factors=factors)
        elif template_id == "concurrent_crosswalk_dual_task":
            asset = _crosswalk(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Crosswalk risk with cabin task", topology),
                factors=factors, initial_signal="green")
        else:
            asset = _crash_obstacle(
                topology, experiment_id, scene_id,
                title=_portfolio_title(
                    "Obstacle bypass with cabin task", topology),
                factors=factors)
    else:
        raise ValueError(f"unsupported portfolio template {template_id}")
    return _tag_portfolio_scene(
        asset, template_id, role="stratified_replica")


def build_scene_assets(network_dir: Path) -> List[SceneAsset]:
    """Build the canonical scene definitions without writing them."""
    maps = {name: Topology(name, network_dir) for name in BENCHMARK_NETWORKS}
    g = maps["beijing_guomao"]
    z = maps["beijing_zhongguancun"]
    s = maps["shanghai_lujiazui"]
    assets: List[SceneAsset] = []

    # Baseline road-interaction family.
    assets.extend([
        _following(g, "baseline", "baseline_straight_following", collision=False,
                   title="Straight-road baseline"),
        _signal_queue(g, "baseline", "baseline_signalized_intersection",
                      title="Signalized-intersection baseline"),
        _connector_conflict(g, "baseline", "baseline_unsignalized_intersection",
                            signalized=False,
                            title="Unsignalized-intersection baseline"),
        _lane_change(g, "baseline", "baseline_lane_change",
                     title="Lane-change baseline"),
        _crosswalk(g, "baseline", "baseline_crosswalk",
                   title="Crosswalk baseline"),
    ])

    # Dense SUMO traffic family.
    traffic_factor = ("background_density",)
    assets.extend([
        _augment_background_vehicles(g, asset) for asset in [
            _following(g, "traffic", "traffic_following_pressure", collision=False,
                       title="Following under dense traffic",
                       factors=traffic_factor),
            _merge(g, "traffic", "traffic_merge_gap",
                   title="Merge under dense traffic", factors=traffic_factor),
            _connector_conflict(g, "traffic", "traffic_intersection_conflict",
                                signalized=False,
                                title="Intersection under dense traffic",
                                factors=traffic_factor),
            _lane_change(g, "traffic", "traffic_lane_change_pressure",
                         title="Lane change under dense traffic",
                         factors=traffic_factor),
        ]
    ])

    # SUMO pedestrian interaction family.
    pedestrian_factor = ("pedestrian_density",)
    assets.extend([
        _augment_background_pedestrians(topology, asset)
        for topology, asset in [
            (g, _crosswalk(g, "pedestrian", "pedestrian_signalized_crosswalk",
                           title="Signalized crosswalk behavior",
                           factors=pedestrian_factor,
                           initial_signal="green")),
            (z, _mapped_unsignalized_crossing(z, "pedestrian", "pedestrian_unsignalized_crossing",
                               title="Unsignalized road crossing",
                               factors=pedestrian_factor)),
            (s, _crosswalk(s, "pedestrian", "pedestrian_turning_crosswalk", turning=True,
                           title="Turning vehicle and pedestrian",
                           factors=pedestrian_factor)),
            (g, _mapped_unsignalized_crossing(g, "pedestrian", "pedestrian_unmarked_jaywalk",
                               title="Unmarked jaywalking", factors=pedestrian_factor,
                               )),
        ]
    ])

    # Cabin/equipment interaction family.
    cabin_factor = ("equipment_profile", "request_support")
    common_equipment = _following(
        g, "cabin", "cabin_common_climate_task", collision=False,
        title="Common equipment request", factors=cabin_factor)
    assets.extend([
        common_equipment,
        _multi_vehicle(
            g, "cabin", "cabin_available_feature_task",
            title="Available equipment request in traffic",
            factors=cabin_factor, count=8),
        _multi_vehicle(
            g, "cabin", "cabin_capability_probe",
            title="Unsupported capability probe",
            factors=cabin_factor, count=8),
    ])

    # Chassis-sensitive driving family.
    chassis_factor = ("chassis_profile",)
    lead_braking = _lead_vehicle_braking(
        g, "chassis", "chassis_lead_vehicle_braking",
        title="Chassis emergency following", factors=chassis_factor)
    assets.extend([
        _signal_approach_stop(
            g, "chassis", "chassis_red_light_stop",
            title="Chassis red-light braking", factors=chassis_factor),
        lead_braking,
        _lane_change(g, "chassis", "chassis_target_lane_gap",
                     title="Chassis lane-change gap", factors=chassis_factor),
        _continuous_turn_route(
            g, "chassis", "chassis_continuous_turns",
            title="Chassis continuous-turn response", factors=chassis_factor),
        _following(z, "chassis", "chassis_narrow_road", collision=False,
                   title="Chassis narrow-road following", factors=chassis_factor),
        _multi_vehicle(
            s, "chassis", "chassis_full_network_route",
            title="Chassis full-network route", factors=chassis_factor,
            oncoming_stream=FULL_NETWORK_ONCOMING_SITES[s.network]),
    ])

    # Multi-LLM interaction family.
    multi_factor = ("llm_count", "model_assignment")
    assets.extend([
        _four_way(g, "multi", "multi_unsignalized_four_way",
                  title="Multi-agent four-way conflict", factors=multi_factor),
        _signal_queue(g, "multi", "multi_signal_queue_start",
                      title="Multi-agent signal queue start", factors=multi_factor),
        _merge(g, "multi", "multi_two_lane_merge",
               title="Multi-agent two-lane merge", factors=multi_factor),
        _crosswalk(s, "multi", "multi_crosswalk_right_turn", turning=True,
                   title="Multi-agent crosswalk right turn", factors=multi_factor),
        _crash_obstacle(g, "multi", "multi_crash_obstacle_bypass",
                        title="Multi-agent crash obstacle bypass", factors=multi_factor),
        _multi_vehicle(g, "multi", "multi_whole_map_traffic",
                       title="Whole-map multi-agent traffic", factors=multi_factor,
                       count=16),
    ])

    # Concurrent cabin/road interaction family.
    concurrent_factor = ("dual_domain_condition", "cabin_task_load", "event_anchor")
    conflict = _connector_conflict(
        g, "concurrent", "concurrent_intersection_dual_task", signalized=False,
        title="Intersection conflict with cabin task", factors=concurrent_factor)
    crosswalk = _crosswalk(
        s, "concurrent", "concurrent_crosswalk_dual_task",
        title="Crosswalk risk with cabin task", factors=concurrent_factor,
        initial_signal="green")
    obstacle = _crash_obstacle(
        g, "concurrent", "concurrent_obstacle_dual_task",
        title="Obstacle bypass with cabin task", factors=concurrent_factor)
    assets.extend([conflict, crosswalk, obstacle])

    # Preserve the original 41 scenes as named anchors, then instantiate the
    # same physical templates on the stratified map portfolio.  Each replica
    # gets new HD-map IDs; scenarios are never copied with stale geometry.
    for asset in assets:
        _tag_portfolio_scene(
            asset, asset.scene_id, role="original_anchor")
    portfolio_jobs: Dict[str, List[tuple[str, str]]] = defaultdict(list)
    for scene_family, networks in SCENE_FAMILY_MAP_ADDITIONS.items():
        for network in networks:
            portfolio_jobs[network].extend(
                (scene_family, template_id)
                for template_id in SCENE_FAMILY_TEMPLATES[scene_family])
    for network, jobs in portfolio_jobs.items():
        topology = maps[network]
        for _scene_family, template_id in jobs:
            assets.append(_build_portfolio_template(template_id, topology))

    ids = [asset.scene_id for asset in assets]
    if len(ids) != len(set(ids)):
        raise ValueError("scene IDs must be globally unique")
    for asset in assets:
        _refresh_scene_geometry(asset)
    return assets


def _build_rebuilt_basic_scene(
    network_dir: Path, kind: str, network: str,
) -> SceneAsset:
    topology = Topology(network, network_dir)
    scene_id = f"traffic_{kind}__{network}"
    title_network = network.replace("_", " ")
    if kind == "oncoming_stream":
        return _oncoming_stream(
            topology, "traffic", scene_id,
            title=f"Three observed oncoming meetings [{title_network}]")
    if kind == "unprotected_left_turn":
        return _unprotected_left_turn(
            topology, "traffic", scene_id,
            title=f"Yielding unprotected left turn [{title_network}]",
            focal_turning=True)
    if kind == "crowded_unsignalized_crosswalk":
        return _crowded_unsignalized_crosswalk(
            topology, "traffic", scene_id,
            title=f"Crowded unsignalized crossing [{title_network}]")
    if kind == "obstacle_gap_change":
        return _obstacle_gap_change(
            topology, "traffic", scene_id,
            title=f"Mandatory obstacle lane change [{title_network}]",
            lane_id=REBUILT_OBSTACLE_GAP_LANES.get(network))
    if kind == "platoon_pressure":
        return _platoon_pressure(
            topology, "traffic", scene_id,
            title=f"Following inside a SUMO-native platoon [{title_network}]")
    if kind == "crossing_stream":
        return _crossing_stream(
            topology, "traffic", scene_id,
            title=f"Gap selection across a three-car stream [{title_network}]")
    if kind == "merge_stream":
        return _merge_stream(
            topology, "traffic", scene_id,
            title=f"Merge into a three-car stream [{title_network}]")
    raise ValueError(f"unsupported rebuilt Basic scene kind {kind!r}")


def _build_basic_expansion_scene(
    network_dir: Path, kind: str, network: str,
    parameters: Mapping[str, Any] | None,
) -> SceneAsset:
    """Instantiate one of the frozen Basic 161--240 expansion sites."""
    topology = Topology(network, network_dir)
    params = dict(parameters or {})
    title_network = network.replace("_", " ")
    source_id = f"{kind}__expansion__{network}"

    if kind == "baseline_crosswalk":
        source_id += f"__site_{int(params['site_ordinal']):02d}"
        return _crosswalk(
            topology, "baseline", source_id,
            title=f"Unsignalized pedestrian yield [{title_network}]",
            required_connector_id=str(params["connector_id"]))
    if kind == "baseline_signalized_intersection":
        return _signal_queue(
            topology, "baseline", source_id,
            title=f"Signal queue discharge [{title_network}]")
    if kind == "baseline_straight_following":
        return _following(
            topology, "baseline", source_id, collision=False,
            title=f"Straight-road following [{title_network}]")
    if kind == "baseline_unsignalized_intersection":
        return _unsignalized_challenge(
            topology, str(params["ego_connector_id"]),
            str(params["cross_connector_id"]),
            float(params["environment_basis_s"]))
    if kind == "chassis_continuous_turns":
        return _compact_turn_route(
            topology, str(params["initial_lane_id"]),
            tuple(str(item) for item in params["connector_ids"]),
            float(params["environment_basis_s"]))
    if kind == "chassis_full_network_route":
        return _multi_vehicle(
            topology, "chassis", source_id,
            title=f"Full-network route with ego-centric traffic [{title_network}]",
            factors=("chassis_profile",), count=12,
            oncoming_stream=params)
    if kind == "chassis_lead_vehicle_braking":
        return _lead_vehicle_braking(
            topology, "chassis", source_id,
            title=f"Lead-vehicle braking [{title_network}]",
            factors=("chassis_profile",))
    if kind == "chassis_narrow_road":
        return _helsinki_narrow_road_sequence(topology)
    if kind == "chassis_red_light_stop":
        return _signal_approach_stop(
            topology, "chassis", source_id,
            title=f"Moving-approach red-light stop [{title_network}]",
            factors=("chassis_profile",))
    if kind == "baseline_required_turn":
        return _required_turn(
            topology, str(params["initial_lane_id"]),
            str(params["connector_id"]))
    if kind == "traffic_oncoming_stream":
        return _oncoming_stream(
            topology, "traffic", source_id,
            title=f"Three observed oncoming meetings [{title_network}]")
    if kind == "traffic_unprotected_left_turn":
        return _unprotected_left_turn(
            topology, "traffic", source_id,
            title=f"Yielding unprotected left turn [{title_network}]",
            focal_turning=True)
    if kind == "traffic_obstacle_gap_change":
        return _obstacle_gap_change(
            topology, "traffic", source_id,
            title=f"Mandatory obstacle lane change [{title_network}]")
    if kind == "traffic_crowded_unsignalized_crosswalk":
        source_id += f"__site_{int(params['site_ordinal']):02d}"
        return _crowded_unsignalized_crosswalk(
            topology, "traffic", source_id,
            title=f"Crowded unsignalized crossing [{title_network}]",
            required_connector_id=str(params["connector_id"]))
    if kind == "traffic_platoon_pressure":
        return _platoon_pressure(
            topology, "traffic", source_id,
            title=f"Following inside a SUMO-native platoon [{title_network}]")
    if kind == "traffic_crossing_stream":
        return _crossing_stream(
            topology, "traffic", source_id,
            title=f"Gap selection across a three-car stream [{title_network}]")
    if kind == "traffic_merge_stream":
        return _merge_stream(
            topology, "traffic", source_id,
            title=f"Merge into a three-car stream [{title_network}]")
    raise ValueError(f"unsupported Basic expansion kind {kind!r}")


def _build_rebuilt_multi_scene(
    network_dir: Path, kind: str, network: str,
    variant: Sequence[Any] | None,
) -> SceneAsset:
    topology = Topology(network, network_dir)
    title_network = network.replace("_", " ")
    if kind == "narrow_meeting":
        if variant is None:
            raise ValueError("narrow meeting needs one authored variant")
        return _narrow_road_sequence(
            topology, variant, experiment_id="multi",
            scene_prefix="multi_narrow_meeting", peer_evaluated=True,
            add_oncoming_follower=True)
    if kind == "unprotected_left_turn":
        return _unprotected_left_turn(
            topology, "multi", f"multi_unprotected_left_turn__{network}",
            title=f"Multi-agent unprotected left turn [{title_network}]",
            focal_turning=True, through_vehicle_count=3)
    if kind == "synchronized_four_way":
        return _four_way(
            topology, "multi", f"multi_synchronized_four_way__{network}",
            title=f"Synchronized four-way negotiation [{title_network}]",
            factors=("llm_count", "model_assignment"), synchronized=True)
    if kind == "three_agent_merge":
        return _three_agent_merge(
            topology, "multi", f"multi_three_agent_merge__{network}",
            title=f"Three-agent merge gap [{title_network}]")
    raise ValueError(f"unsupported rebuilt MultiLLM scene kind {kind!r}")


def _minimal_route_actions(actions: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Freeze only the fields consumed by deterministic scenario placement."""
    frozen = []
    for action in actions:
        if action.get("type") == "connector":
            frozen.append({
                "type": "connector",
                "connector_id": str(action["connector_id"]),
            })
        elif action.get("type") == "lane_change":
            target = action.get("to_lane_id") or action.get("to_lane")
            if target:
                frozen.append({
                    "type": "lane_change",
                    "to_lane_id": str(target),
                })
    return frozen


def _vehicle_route_context(
    topology: Topology, vehicle: Mapping[str, Any],
) -> tuple[List[str], List[str]]:
    """Return the authored route, falling back to the runtime route planner."""
    state = vehicle.get("initial_physical_state", {})
    lane_id = str(state.get("lane_id", ""))
    route_lanes: List[str] = []
    connector_ids: List[str] = []
    active_connector_id = str(state.get("connector_id", ""))
    if active_connector_id:
        active = topology.connector.get(active_connector_id)
        if active is None:
            raise ValueError(
                f"unknown initial connector {active_connector_id!r}")
        route_lanes.extend([active["from_lane"], active["to_lane"]])
        connector_ids.append(active_connector_id)
        lane_id = active["to_lane"]
    elif lane_id:
        route_lanes.append(lane_id)
    else:
        raise ValueError(
            f"vehicle {vehicle.get('vehicle_id')} has no initial lane")

    actions = list(state.get("lane_route_actions") or [])
    if not actions:
        lane = topology.lane[lane_id]
        plan = topology.runtime.plan_lane_route(
            lane["start_node"], str(vehicle.get("destination_node", "")),
            current_lane_id=lane_id)
        if plan is None:
            raise ValueError(
                f"vehicle {vehicle.get('vehicle_id')} has no route from "
                f"{lane_id} to {vehicle.get('destination_node')}")
        actions = list(plan["actions"])

    cursor = lane_id
    for action in actions:
        if action.get("type") == "connector":
            connector_id = str(action.get("connector_id", ""))
            connector = topology.connector.get(connector_id)
            if connector is None:
                raise ValueError(f"unknown route connector {connector_id!r}")
            cursor = connector["to_lane"]
            connector_ids.append(connector_id)
        elif action.get("type") == "lane_change":
            cursor = str(
                action.get("to_lane_id") or action.get("to_lane") or "")
            if cursor not in topology.lane:
                raise ValueError("route lane change has no mapped target")
        else:
            continue
        if not route_lanes or route_lanes[-1] != cursor:
            route_lanes.append(cursor)
    return list(dict.fromkeys(route_lanes)), list(dict.fromkeys(connector_ids))


def _augment_related_background_traffic(
    topology: Topology, scenario: dict, expected: dict, focal_id: str,
) -> dict:
    """Add auditable native-SUMO trips coupled to the focal route."""
    vehicles = scenario["vehicles"]
    by_id = {str(vehicle["vehicle_id"]): vehicle for vehicle in vehicles}
    focal = by_id[focal_id]
    route_lane_ids, focal_connector_ids = _vehicle_route_context(
        topology, focal)
    route_lanes = [topology.lane[lane_id] for lane_id in route_lane_ids]
    route_keys = {
        (lane["segment_id"], lane["direction"]) for lane in route_lanes}
    route_segment_ids = {lane["segment_id"] for lane in route_lanes}
    pedestrian_sensitive = bool(scenario.get("pedestrians"))

    occupied: Dict[str, List[float]] = defaultdict(list)
    for vehicle in vehicles:
        state = vehicle.get("initial_physical_state", {})
        lane_id = str(state.get("lane_id", ""))
        if lane_id in topology.lane:
            occupied[lane_id].append(float(state.get("progress", 0.0)))

    existing_ids = set(by_id)
    cohort_vehicle_ids: List[str] = []
    cohort_initial_route_offsets_m: Dict[str, float] = {}
    cohort_destination_nodes: Dict[str, str] = {}
    route_vehicle_ids: List[str] = []
    crossing_vehicle_ids: List[str] = []

    def reserve_id(kind: str, ordinal: int) -> str:
        base = f"map_{kind}_{ordinal:02d}"
        candidate = base
        suffix = 1
        while candidate in existing_ids:
            suffix += 1
            candidate = f"{base}_{suffix}"
        existing_ids.add(candidate)
        return candidate

    def is_clear(
        lane: Mapping[str, Any], progress: float,
        clearance_m: float = RELATED_BACKGROUND_CLEARANCE_M,
    ) -> bool:
        return all(
            abs(progress - other) * float(lane["length_m"])
            >= clearance_m
            for other in occupied[lane["id"]])

    # Build a local, long-overlap traffic cohort before distributing the
    # original five related vehicles.  Same-direction sibling lanes share the
    # focal road segment, and every accepted route reaches the focal
    # destination.  The stable target gives portfolio-wide 3/4/5 variation;
    # compact one-lane starts may stop early but must still fit three vehicles.
    focal_state = focal.get("initial_physical_state", {})
    focal_lane_id = str(focal_state.get("lane_id", ""))
    if not focal_lane_id:
        active_connector = topology.connector.get(
            str(focal_state.get("connector_id", "")))
        if active_connector is None:
            raise ValueError(
                f"{scenario['scenario_id']} focal vehicle has no lane anchor")
        focal_lane_id = str(active_connector["to_lane"])
        focal_progress = 0.04
    else:
        focal_progress = float(focal_state.get("progress", 0.0))
    focal_lane = topology.lane[focal_lane_id]
    focal_destination = str(focal["destination_node"])
    cohort_target = LONGITUDINAL_COHORT_TARGET_OVERRIDES.get(
        scenario["scenario_id"],
        LONGITUDINAL_COHORT_MINIMUM
        + _stable_number(
            f"{scenario['scenario_id']}:longitudinal-cohort")
        % (LONGITUDINAL_COHORT_BASE_MAXIMUM
           - LONGITUDINAL_COHORT_MINIMUM + 1),
    )
    adjacent_first = (
        scenario["scenario_id"] in ADJACENT_FIRST_COHORT_SCENES)
    cohort_lanes = sorted(
        topology.sibling_lanes(focal_lane),
        key=lambda lane: (
            (1 if lane["id"] == focal_lane_id else 0)
            if adjacent_first
            else (0 if lane["id"] == focal_lane_id else 1),
            abs(int(lane["index"]) - int(focal_lane["index"])),
            lane["id"],
        ),
    )
    # Followers are preferable around pedestrian, merge and unprotected-turn
    # contracts: leaders could reach the authored conflict before ego and
    # change the task itself.  Other scenes distribute the traffic around ego
    # so both following and leading effects remain observable.
    source_template_id = str(scenario.get(
        "experiment_scene", {}).get("source_template_id", ""))
    follower_only = pedestrian_sensitive or any(
        token in source_template_id
        for token in ("unprotected_left_turn", "merge_stream"))
    post_crossing_only = (
        scenario["scenario_id"] in POST_CROSSING_ONLY_COHORT_SCENES)
    longitudinal_offsets_m = (
        (-12.0, -24.0, -36.0, -48.0)
        if follower_only
        else (12.0, -12.0, 0.0, 24.0, -24.0,
              36.0, -36.0, 48.0, -48.0)
    )
    cohort_placements = (
        [
            (lane, offset_m)
            for lane in cohort_lanes
            for offset_m in longitudinal_offsets_m
        ]
        if follower_only
        else [
            (lane, offset_m)
            for offset_m in longitudinal_offsets_m
            for lane in cohort_lanes
        ]
    )

    def add_cohort_vehicle(
        lane: Mapping[str, Any], progress: float, route_offset_m: float,
        *, destination_node: str | None = None,
    ) -> bool:
        destination = str(destination_node or focal_destination)
        if not is_clear(
                lane, progress,
                clearance_m=LONGITUDINAL_COHORT_CLEARANCE_M):
            return False
        if destination == str(lane["start_node"]):
            return False
        plan = topology.runtime.plan_lane_route(
            lane["start_node"], destination,
            current_lane_id=lane["id"])
        if plan is None:
            return False
        speed_limit = float(lane.get("speed_limit_kmh", 30.0))
        speed = max(12.0, min(
            30.0, speed_limit * (
                0.58 + 0.04 * (len(cohort_vehicle_ids) % 3))))
        vehicle_id = reserve_id(
            "flow", len(cohort_vehicle_ids) + 1)
        vehicle = _vehicle(
            vehicle_id, lane, destination_node=destination,
            progress=progress, speed=round(speed, 3), evaluated=False)
        plan_actions = _minimal_route_actions(plan["actions"])
        if plan_actions:
            vehicle["initial_physical_state"][
                "lane_route_actions"] = plan_actions
        vehicles.append(vehicle)
        occupied[lane["id"]].append(progress)
        cohort_vehicle_ids.append(vehicle_id)
        cohort_initial_route_offsets_m[vehicle_id] = round(
            float(route_offset_m), 3)
        cohort_destination_nodes[vehicle_id] = destination
        return True

    for lane, offset_m in (() if post_crossing_only else cohort_placements):
        if len(cohort_vehicle_ids) >= cohort_target:
            break
        length_m = float(lane["length_m"])
        focal_station_m = focal_progress * length_m
        station_m = focal_station_m + offset_m
        if station_m < 4.0 or station_m > length_m - 4.0:
            continue
        add_cohort_vehicle(
            lane, station_m / length_m, route_offset_m=offset_m,
            destination_node=(
                str(lane["end_node"])
                if pedestrian_sensitive else focal_destination))

    # The shortest single-lane crosswalk approach has room for only one
    # follower.  Complete its cohort immediately after the authored crossing,
    # still on ego's route and close to the start in route distance.  These
    # vehicles cannot pre-empt or collide with the crossing pedestrian.
    if ((len(cohort_vehicle_ids) < LONGITUDINAL_COHORT_MINIMUM
            or post_crossing_only)
            and pedestrian_sensitive and focal_connector_ids
            and len(route_lane_ids) >= 2):
        downstream_lane = topology.lane[route_lane_ids[1]]
        direct_connector = next((
            topology.connector[connector_id]
            for connector_id in focal_connector_ids
            if (topology.connector[connector_id]["from_lane"]
                == focal_lane_id
                and topology.connector[connector_id]["to_lane"]
                == downstream_lane["id"])
        ), None)
        if direct_connector is not None:
            connector_line = direct_connector.get("centerline_xy", [])
            connector_length_m = sum(
                math.dist(tuple(first), tuple(second))
                for first, second in zip(
                    connector_line, connector_line[1:]))
            downstream_base_offset_m = (
                (1.0 - focal_progress) * float(focal_lane["length_m"])
                + connector_length_m)
            downstream_lanes = sorted(
                topology.sibling_lanes(downstream_lane),
                key=lambda lane: (
                    0 if lane["id"] == downstream_lane["id"] else 1,
                    abs(int(lane["index"])
                        - int(downstream_lane["index"])),
                    lane["id"],
                ))
            for station_m in (4.0, 16.0, 28.0):
                for lane in downstream_lanes:
                    if len(cohort_vehicle_ids) >= cohort_target:
                        break
                    route_offset_m = downstream_base_offset_m + station_m
                    if (route_offset_m
                            > LONGITUDINAL_COHORT_MAX_OFFSET_M + 1e-6
                            or station_m > float(lane["length_m"]) - 4.0):
                        continue
                    add_cohort_vehicle(
                        lane, station_m / float(lane["length_m"]),
                        route_offset_m=route_offset_m)
                if len(cohort_vehicle_ids) >= cohort_target:
                    break

    if len(cohort_vehicle_ids) < LONGITUDINAL_COHORT_MINIMUM:
        raise ValueError(
            f"{scenario['scenario_id']} can place only "
            f"{len(cohort_vehicle_ids)} longitudinal cohort vehicles; "
            f"needed {LONGITUDINAL_COHORT_MINIMUM}")

    candidate_lanes = []
    fallback_opposing_lanes = []
    seen_lanes = set()
    for route_lane in route_lanes:
        siblings = sorted(
            topology.sibling_lanes(route_lane),
            key=lambda lane: (
                0 if lane["id"] == route_lane["id"] else 1,
                abs(int(lane["index"]) - int(route_lane["index"])),
                lane["id"],
            ))
        for lane in siblings:
            if lane["id"] not in seen_lanes:
                seen_lanes.add(lane["id"])
                candidate_lanes.append(lane)
    # A few compact two-way streets have only one lane in the focal travel
    # direction.  Keep the normal same-direction placements first, then allow
    # the opposing lane on the same physical segment as a capacity fallback.
    # Those vehicles are still directly coupled to ego (oncoming traffic), and
    # are preferable to padding the task with unrelated network traffic.
    for route_lane in route_lanes:
        opposing_lanes = sorted(
            (
                lane for lane in topology.lanes
                if lane["segment_id"] == route_lane["segment_id"]
                and lane["direction"] != route_lane["direction"]
            ),
            key=lambda lane: (
                abs(int(lane["index"]) - int(route_lane["index"])),
                lane["id"],
            ))
        for lane in opposing_lanes:
            if lane["id"] not in seen_lanes:
                seen_lanes.add(lane["id"])
                fallback_opposing_lanes.append(lane)

    # Preserve the original five-vehicle placement order whenever capacity
    # permits.  A denser grid is appended only as a fallback for compact
    # corridors whose new cohort consumes one of the historical slots.
    if "narrow_meeting" in source_template_id:
        progress_grid = [index / 10.0 for index in range(1, 10)]
        rotation = (
            _stable_number(scenario["scenario_id"]) % len(progress_grid))
        progress_grid = progress_grid[rotation:] + progress_grid[:rotation]
    else:
        primary_progress_grid = [0.18, 0.34, 0.50, 0.66, 0.82]
        rotation = (
            _stable_number(scenario["scenario_id"])
            % len(primary_progress_grid))
        primary_progress_grid = (
            primary_progress_grid[rotation:]
            + primary_progress_grid[:rotation])
        fallback_progress_grid = [
            index / 10.0 for index in range(1, 10)
            if all(abs(index / 10.0 - primary) > 1e-6
                   for primary in primary_progress_grid)
        ]
        progress_grid = [*primary_progress_grid, *fallback_progress_grid]
    route_placements = [
        (lane, progress)
        for progress in progress_grid
        for lane in candidate_lanes
    ] + [
        (lane, progress)
        for progress in progress_grid
        for lane in fallback_opposing_lanes
    ]

    def add_route_vehicle() -> bool:
        for lane, progress in route_placements:
            if (float(lane["length_m"]) < 24.0
                    or not is_clear(lane, progress)):
                continue
            same_direction = (
                lane["segment_id"], lane["direction"]) in route_keys
            destination = str(focal["destination_node"])
            if (pedestrian_sensitive
                    or follower_only
                    or not same_direction
                    or destination == str(lane["start_node"])):
                destination = str(lane["end_node"])
            if pedestrian_sensitive or follower_only:
                plan_actions: List[dict] = []
            else:
                plan = topology.runtime.plan_lane_route(
                    lane["start_node"], destination,
                    current_lane_id=lane["id"])
                if plan is None:
                    destination = str(lane["end_node"])
                    plan_actions = []
                else:
                    plan_actions = _minimal_route_actions(plan["actions"])
            speed_limit = float(lane.get("speed_limit_kmh", 30.0))
            speed = max(12.0, min(
                28.0, speed_limit * (
                    0.55 + 0.05 * (len(route_vehicle_ids) % 3))))
            vehicle_id = reserve_id(
                "route", len(route_vehicle_ids) + 1)
            vehicle = _vehicle(
                vehicle_id, lane, destination_node=destination,
                progress=progress, speed=round(speed, 3),
                evaluated=False)
            if plan_actions:
                vehicle["initial_physical_state"][
                    "lane_route_actions"] = plan_actions
            vehicles.append(vehicle)
            occupied[lane["id"]].append(progress)
            route_vehicle_ids.append(vehicle_id)
            return True
        return False

    conflict_candidates = []
    for focal_connector_id in (() if pedestrian_sensitive
                               else focal_connector_ids):
        for crossing_id in sorted(
                topology.runtime._connector_conflicts.get(
                    focal_connector_id, set())):
            crossing = topology.connector.get(crossing_id)
            if crossing is None:
                continue
            source = topology.lane[crossing["from_lane"]]
            target = topology.lane[crossing["to_lane"]]
            if (float(source["length_m"]) < 24.0
                    or str(source["start_node"])
                    == str(target["end_node"])):
                continue
            conflict_candidates.append((
                focal_connector_id, crossing_id, crossing, source))
    conflict_candidates.sort(key=lambda item: (
        _stable_number(
            f"{scenario['scenario_id']}:{item[0]}:{item[1]}"),
        item[1],
    ))

    def add_crossing_vehicle() -> bool:
        setback_grid = (30.0, 50.0, 70.0, 90.0)
        for _, _, connector, source in conflict_candidates:
            for setback in setback_grid:
                progress = max(0.08, min(
                    0.82, 1.0 - setback / float(source["length_m"])))
                if not is_clear(source, progress):
                    continue
                speed_limit = float(source.get("speed_limit_kmh", 30.0))
                speed = max(12.0, min(
                    26.0, speed_limit * (
                        0.58 + 0.05 * (len(crossing_vehicle_ids) % 2))))
                vehicle_id = reserve_id(
                    "cross", len(crossing_vehicle_ids) + 1)
                vehicle = _approach_connector_vehicle(
                    topology, vehicle_id, connector,
                    distance_m=(1.0 - progress)
                    * float(source["length_m"]),
                    speed=round(speed, 3), evaluated=False)
                vehicles.append(vehicle)
                occupied[source["id"]].append(progress)
                crossing_vehicle_ids.append(vehicle_id)
                return True
        return False

    while len(route_vehicle_ids) < RELATED_BACKGROUND_ROUTE_TARGET:
        if not add_route_vehicle():
            break
    while len(crossing_vehicle_ids) < RELATED_BACKGROUND_CROSSING_TARGET:
        if not add_crossing_vehicle():
            break
    while (len(route_vehicle_ids) + len(crossing_vehicle_ids)
           < RELATED_BACKGROUND_BASE_MINIMUM):
        if add_route_vehicle():
            continue
        if add_crossing_vehicle():
            continue
        raise ValueError(
            f"{scenario['scenario_id']} cannot place "
            f"{RELATED_BACKGROUND_BASE_MINIMUM} base related background "
            "vehicles")

    all_route_vehicle_ids = [*cohort_vehicle_ids, *route_vehicle_ids]
    all_related_vehicle_ids = [
        *all_route_vehicle_ids, *crossing_vehicle_ids]

    for assertion in expected.get("setup_assertions", []):
        if assertion.get("kind") == "entity_count":
            assertion["vehicles"] = len(vehicles)
    assertion = {
        "kind": "related_background_traffic",
        "entity": focal_id,
        "minimum_count": RELATED_BACKGROUND_MINIMUM,
        "longitudinal_cohort_minimum_count": (
            LONGITUDINAL_COHORT_MINIMUM),
        "longitudinal_cohort_maximum_count": (
            LONGITUDINAL_COHORT_BASE_MAXIMUM),
        "longitudinal_cohort_vehicle_ids": cohort_vehicle_ids,
        "longitudinal_cohort_initial_route_offsets_m": (
            cohort_initial_route_offsets_m),
        "longitudinal_cohort_destination_nodes": cohort_destination_nodes,
        "longitudinal_cohort_anchor_lane_id": focal_lane_id,
        "longitudinal_cohort_maximum_offset_m": (
            LONGITUDINAL_COHORT_MAX_OFFSET_M),
        "route_aligned_vehicle_ids": all_route_vehicle_ids,
        "crossing_vehicle_ids": crossing_vehicle_ids,
        "focal_route_lane_ids": route_lane_ids,
        "focal_connector_ids": focal_connector_ids,
    }
    expected.setdefault("setup_assertions", []).append(assertion)
    return {
        "schema": "vehiclearena-related-background-v1",
        "minimum_count": RELATED_BACKGROUND_MINIMUM,
        "base_minimum_count": RELATED_BACKGROUND_BASE_MINIMUM,
        "longitudinal_cohort_minimum_count": (
            LONGITUDINAL_COHORT_MINIMUM),
        "longitudinal_cohort_maximum_count": (
            LONGITUDINAL_COHORT_BASE_MAXIMUM),
        "longitudinal_cohort_vehicle_ids": cohort_vehicle_ids,
        "longitudinal_cohort_initial_route_offsets_m": (
            cohort_initial_route_offsets_m),
        "longitudinal_cohort_destination_nodes": cohort_destination_nodes,
        "longitudinal_cohort_anchor_lane_id": focal_lane_id,
        "longitudinal_cohort_maximum_offset_m": (
            LONGITUDINAL_COHORT_MAX_OFFSET_M),
        "longitudinal_cohort_destination_policy": (
            "pedestrian_approach_or_focal_destination"
            if pedestrian_sensitive else "focal_destination"),
        "route_aligned_vehicle_ids": all_route_vehicle_ids,
        "supplemental_route_aligned_vehicle_ids": route_vehicle_ids,
        "crossing_vehicle_ids": crossing_vehicle_ids,
        "vehicle_ids": all_related_vehicle_ids,
        "relation_definition": {
            "longitudinal_cohort": (
                "starts near the focal vehicle on its same-direction start "
                "segment, or just beyond a compact pedestrian connector, "
                "and shares the focal downstream destination; pedestrian "
                "followers terminate at the crossing approach so native "
                "traffic never pre-empts the authored pedestrian"
            ),
            "route_aligned": (
                "same physical road segment as the focal route; "
                "same-direction lanes are preferred and opposing lanes are used "
                "only as a compact-road fallback"
            ),
            "crossing": "connector conflicts with a connector on the focal route",
        },
        "focal_route_keys": [
            {"segment_id": str(segment_id), "direction": str(direction)}
            for segment_id, direction in sorted(route_keys)
        ],
        "focal_route_segment_ids": sorted(
            str(segment_id) for segment_id in route_segment_ids),
    }


def normalize_native_npc_inputs(scenario: dict) -> None:
    """Keep only initial conditions for native SUMO background traffic."""
    npc_ids = set()
    for vehicle in scenario.get("vehicles", []):
        if vehicle.get("agent_config", {}).get("type") == "sumo":
            npc_ids.add(str(vehicle.get("vehicle_id", "")))
            state = vehicle.get("initial_physical_state", {})
            state.pop("target_speed_kmh", None)
            state.pop("desired_speed_kmh", None)
    npc_ids.update(
        str(pedestrian.get("ped_id", ""))
        for pedestrian in scenario.get("pedestrians", [])
        if pedestrian.get("agent_config", {}).get("type", "sumo") == "sumo")
    events = [
        event for event in scenario.get("experiment_world_events", [])
        if str(event.get("entity_id", "")) not in npc_ids
    ]
    if events:
        scenario["experiment_world_events"] = events
    else:
        scenario.pop("experiment_world_events", None)


def build_study_assets(
    network_dir: Path,
    environment_schedule_duration_s: Mapping[str, float] | None = None,
    environment_pattern_overrides: Mapping[str, dict] | None = None,
) -> List[SceneAsset]:
    """Project the physical template library into the two paper studies.

    Basic uses 180 single-LLM tasks. Multi-LLM uses 20 interaction tasks
    with one focal LLM, up to three fixed peer LLMs, and SUMO background
    traffic. Physical templates are grouped by interaction family rather than
    by a legacy experiment number.
    """
    source_assets = build_scene_assets(network_dir)
    basic_sources = [
        asset for asset in source_assets
        if asset.experiment_id in {"baseline", "traffic", "pedestrian", "cabin", "chassis"}
        or (
            asset.experiment_id == "concurrent"
            and "obstacle_dual_task" not in asset.scene_id
        )
    ]
    multi_sources = [
        asset for asset in source_assets
        if asset.experiment_id == "multi"
        and "crosswalk_right_turn" not in asset.scene_id
    ]
    if len(basic_sources) != 96 or len(multi_sources) != 20:
        raise ValueError(
            "study task contract changed: expected 96 Basic and 20 "
            f"Multi-LLM tasks, got {len(basic_sources)} and "
            f"{len(multi_sources)}")

    schedule_durations = dict(environment_schedule_duration_s or {})
    pattern_overrides = dict(environment_pattern_overrides or {})
    projected: List[SceneAsset] = []
    for study, assets in (
            ("Basic", basic_sources), ("MultiLLM", multi_sources)):
        ordered_assets = sorted(assets, key=lambda item: item.scene_id)
        if study == "Basic":
            # Append four additional sites; never renumber unrelated frozen tasks.
            ordered_assets.extend(_crosswalk(
                Topology(network, network_dir), "baseline", f"baseline_crosswalk__{network}",
                title=f"Unsignalized pedestrian-yield baseline [{network}]")
                for network in PEDESTRIAN_YIELD_EXTRA_NETWORKS)
        indexed_assets = list(enumerate(ordered_assets, start=1))
        if study == "Basic":
            # Preserve all surviving IDs; retire the 12 optional-passing
            # templates and the four old same-lane "narrow road" following
            # tasks. Replace those four slots with real shared-corridor
            # interactions and add four more without renumbering any other
            # frozen task.
            indexed_assets = [(i, asset) for i, asset in indexed_assets
                              if asset.scene_id.split("__", 1)[0]
                              not in (RETIRED_PASSING_TEMPLATES
                                      | RETIRED_BASIC_STUDY_TEMPLATES)]
            indexed_assets.extend((101 + i, _required_turn(
                Topology(network, network_dir), lane_id, connector_id))
                for i, (network, lane_id, connector_id) in enumerate(REQUIRED_TURN_SITES))
            indexed_assets.extend((109 + i, _signal_queue_middle(
                Topology(network, network_dir), connector_id))
                for i, (network, connector_id) in enumerate(MIDDLE_SIGNAL_SITES))
            indexed_assets.extend((113 + i, _unsignalized_challenge(
                Topology(network, network_dir), ego_id, cross_id, basis))
                for i, (network, ego_id, cross_id, basis)
                in enumerate(UNSIGNALIZED_CHALLENGE_SITES))
            indexed_assets.extend((117 + i, _compact_turn_route(
                Topology(network, network_dir), lane_id, connector_ids, basis))
                for i, (network, lane_id, connector_ids, basis)
                in enumerate(COMPACT_TURN_SITES))
            indexed_assets.extend((121 + i, _lead_vehicle_braking(
                Topology(network, network_dir), "chassis",
                f"chassis_lead_vehicle_braking__extension__{network}",
                title=f"Chassis emergency following [{network}]",
                factors=("chassis_profile",), lane_id=lane_id,
                environment_basis_s=basis, duration_s=duration))
                for i, (network, lane_id, basis, duration)
                in enumerate(LEAD_BRAKING_EXTRA_SITES))
            indexed_assets.extend((129 + i, _signal_approach_stop(
                Topology(network, network_dir), "chassis",
                f"chassis_red_light_stop__extension__{network}",
                title=f"Bounded red-light braking [{network}]",
                factors=("chassis_profile",),
                connector_id=connector_id,
                approach_distance_m=38.0,
                red_wait_range_s=(8.0, 15.0),
                environment_basis_s=20.0,
                duration_s=35.0))
                for i, (network, connector_id)
                in enumerate(RED_LIGHT_STOP_EXTRA_SITES))
            narrow_topology = Topology(
                NARROW_CORRIDOR_NETWORK, network_dir)
            indexed_assets.extend((
                (45 + i if i < 4 else 121 + i),
                _narrow_road_sequence(narrow_topology, variant),
            ) for i, variant in enumerate(NARROW_CORRIDOR_VARIANTS))
            indexed_assets.extend((
                133 + i,
                _build_rebuilt_basic_scene(network_dir, kind, network),
            ) for i, (kind, network) in enumerate(BASIC_REBUILT_SCENES))
            indexed_assets.extend((
                161 + i,
                _build_basic_expansion_scene(
                    network_dir, kind, network, parameters),
            ) for i, (kind, network, parameters)
              in enumerate(BASIC_EXPANSION_SCENES))
            indexed_assets.sort(key=lambda item: item[0])
        else:
            indexed_assets = [
                (i, asset) for i, asset in indexed_assets
                if asset.scene_id.split("__", 1)[0]
                not in RETIRED_MULTI_STUDY_TEMPLATES]
            indexed_assets.extend((
                21 + i,
                _build_rebuilt_multi_scene(
                    network_dir, kind, network, variant),
            ) for i, (kind, network, variant)
              in enumerate(MULTI_REBUILT_SCENES))
            indexed_assets.sort(key=lambda item: item[0])
        expected_count = 180 if study == "Basic" else 20
        if len(indexed_assets) != expected_count:
            raise ValueError(
                f"formal {study} contract changed: expected "
                f"{expected_count} valid tasks, got {len(indexed_assets)}")
        weather_candidates = []
        daynight_candidates = []
        for index, source in indexed_assets:
            prefix = "basic" if study == "Basic" else "multi"
            scene_id = (
                f"{prefix}_{index:03d}_"
                f"{source.scene_id.split('_', 1)[-1]}")
            duration_s = float(schedule_durations.get(
                scene_id, source.scenario.get("experiment_scene", {}).get(
                    "environment_schedule_basis_s", source.scenario["total_time_s"])))
            if index % 2 == 1:
                weather_candidates.append((duration_s, index))
            if index % 4 in (2, 3):
                daynight_candidates.append((duration_s, index))
        weather_candidates.sort()
        single_count = (len(weather_candidates) + 2) // 3
        remaining = len(weather_candidates) - single_count
        recover_count = (remaining + 1) // 2
        weather_pattern_by_index = {
            index: (
                "onset" if ordinal < single_count
                else "temporary"
                if ordinal < single_count + recover_count
                else "evolve")
            for ordinal, (_, index) in enumerate(weather_candidates)
        }
        daynight_candidates.sort()
        single_count = (len(daynight_candidates) + 2) // 3
        remaining = len(daynight_candidates) - single_count
        recover_count = (remaining + 1) // 2
        daynight_pattern_by_index = {
            index: (
                "single_step" if ordinal < single_count
                else "forward_two_step"
                if ordinal < single_count + recover_count
                else "light_boundary")
            for ordinal, (_, index) in enumerate(daynight_candidates)
        }

        for index, source in indexed_assets:
            scenario = copy.deepcopy(source.scenario)
            expected = copy.deepcopy(source.expected)
            stem = source.scene_id.split("_", 1)[-1]
            prefix = "basic" if study == "Basic" else "multi"
            scene_id = f"{prefix}_{index:03d}_{stem}"
            scenario["scenario_id"] = scene_id
            scenario["name"] = (
                f"{study} study task {index:03d}: "
                f"{scenario.get('name', source.scene_id)}")
            schedule_duration_s = float(schedule_durations.get(
                scene_id, scenario.get("experiment_scene", {}).get(
                    "environment_schedule_basis_s", scenario["total_time_s"])))
            if scene_id in schedule_durations:
                # A frozen calibration owns the task deadline. Environment
                # frames are fitted to that deadline, never vice versa.
                scenario["total_time_s"] = schedule_duration_s
            weather_keyframes, weather_profile = _formal_weather_profile(
                index, scene_id, schedule_duration_s,
                PINNED_REBUILT_ENVIRONMENT_PATTERNS.get(
                    scene_id, {}).get(
                        "weather",
                        pattern_overrides.get(scene_id, {}).get(
                            "weather", weather_pattern_by_index.get(index))),
                scenario["road_network_id"])
            if weather_keyframes:
                scenario["weather_keyframes"] = weather_keyframes
                expected.setdefault("setup_assertions", []).append({
                    "kind": "weather_transition",
                    "initial_condition": weather_profile[
                        "initial_condition"],
                    "target_condition": weather_profile[
                        "target_condition"],
                    "at_s": weather_profile["change_at_s"],
                    "second_condition": weather_profile[
                        "second_condition"],
                    "second_at_s": weather_profile[
                        "second_change_at_s"],
                })
            else:
                scenario.pop("weather_keyframes", None)
            daynight_keyframes, daynight_profile = (
                _formal_daynight_profile(
                    index, scene_id, schedule_duration_s,
                    PINNED_REBUILT_ENVIRONMENT_PATTERNS.get(
                        scene_id, {}).get(
                            "daynight",
                            pattern_overrides.get(scene_id, {}).get(
                                "daynight",
                                daynight_pattern_by_index.get(index))),
                    climate_month_value=weather_profile.get(
                        "climate_month"),
                    avoid_change_times_s=tuple(
                        time_s for time_s in (
                            weather_profile.get("change_at_s"),
                            weather_profile.get("second_change_at_s"))
                        if time_s is not None)))
            if daynight_keyframes:
                scenario["daynight_keyframes"] = daynight_keyframes
                expected.setdefault("setup_assertions", []).append({
                    "kind": "daynight_transition",
                    "initial_period": daynight_profile["initial_period"],
                    "target_period": daynight_profile["target_period"],
                    "at_s": daynight_profile["change_at_s"],
                    "second_period": daynight_profile["second_period"],
                    "second_at_s": daynight_profile[
                        "second_change_at_s"],
                })
            else:
                scenario.pop("daynight_keyframes", None)
            vehicles = scenario.get("vehicles", [])
            if not vehicles:
                raise ValueError(f"{source.scene_id} has no focal vehicle")
            focal_id = str(vehicles[0]["vehicle_id"])
            llm_ids = [focal_id]
            if study == "MultiLLM":
                eligible_peers = [
                    vehicle for vehicle in vehicles[1:]
                    if not bool(vehicle.get(
                        "initial_physical_state", {}).get("crashed"))]
                llm_ids.extend(
                    str(vehicle["vehicle_id"])
                    for vehicle in eligible_peers[:3])
                if len(llm_ids) < 2:
                    raise ValueError(
                        f"{source.scene_id} is not a multi-vehicle task")
            llm_id_set = set(llm_ids)
            for vehicle in vehicles:
                vehicle_id = str(vehicle["vehicle_id"])
                is_llm = vehicle_id in llm_id_set
                if (is_llm and vehicle["initial_node"]
                        == vehicle["destination_node"]):
                    raise ValueError(
                        f"{scene_id} evaluated vehicle {vehicle_id} has "
                        "identical start and destination nodes")
                vehicle["agent_config"] = {
                    "type": "llm" if is_llm else "sumo"}
                vehicle["is_evaluated"] = is_llm
            topology = Topology(
                scenario["road_network_id"], network_dir)
            normalize_native_npc_inputs(scenario)
            for pedestrian in scenario.get("pedestrians", []):
                pedestrian["agent_config"] = {"type": "sumo"}
                pedestrian["is_evaluated"] = False

            interaction_tags = list(source.scenario.get(
                "experiment_scene", {}).get("interaction_tags", []))
            if weather_profile["mode"] == "transition":
                interaction_tags.extend([
                    "weather_change",
                    f"weather_target_{weather_profile['primary_condition']}",
                    f"weather_pattern_{weather_profile['pattern']}",
                ])
                if weather_profile["second_condition"] is not None:
                    interaction_tags.append(
                        "weather_second_"
                        f"{weather_profile['second_condition']}")
            else:
                interaction_tags.append("stable_weather_control")
            if daynight_profile["mode"] == "transition":
                interaction_tags.extend([
                    "daynight_change",
                    f"daynight_target_{daynight_profile['target_period']}",
                    f"daynight_pattern_{daynight_profile['pattern']}",
                ])
                if daynight_profile["second_period"] is not None:
                    interaction_tags.append(
                        "daynight_second_"
                        f"{daynight_profile['second_period']}")
                if daynight_profile["joint_weather_change_times_s"]:
                    interaction_tags.append("joint_environment_response")
            else:
                interaction_tags.append("stable_daynight_control")
            environment_group = (
                "combined"
                if (weather_profile["mode"] == "transition"
                    and daynight_profile["mode"] == "transition")
                else "weather_only"
                if weather_profile["mode"] == "transition"
                else "daynight_only"
                if daynight_profile["mode"] == "transition"
                else "control"
            )
            interaction_tags.append(
                f"environment_group_{environment_group}")
            scenario["experiment_scene"] = {
                "schema": CATALOG_SCHEMA,
                "study": "basic" if study == "Basic" else "multi_llm",
                "source_template_id": source.scene_id,
                "focal_vehicle_id": focal_id,
                "fixed_peer_vehicle_ids": llm_ids[1:],
                "sumo_background_vehicle_ids": [
                    str(vehicle["vehicle_id"]) for vehicle in vehicles
                    if str(vehicle["vehicle_id"]) not in llm_id_set],
                "sumo_background_pedestrian_ids": [
                    str(pedestrian["ped_id"])
                    for pedestrian in scenario.get("pedestrians", [])],
                "interaction_tags": interaction_tags,
                "weather_profile": weather_profile,
                "daynight_profile": daynight_profile,
                "environment_group": environment_group,
                "network": scenario["road_network_id"],
            }
            expected["schema"] = CATALOG_SCHEMA
            expected["scene_id"] = scene_id
            adapt_scene(scenario, expected, topology.runtime)
            related_background = _augment_related_background_traffic(
                topology, scenario, expected, focal_id)
            normalize_native_npc_inputs(scenario)
            metadata = scenario["experiment_scene"]
            metadata["related_background_traffic"] = related_background
            metadata["sumo_background_vehicle_ids"] = [
                str(vehicle["vehicle_id"])
                for vehicle in scenario["vehicles"]
                if str(vehicle["vehicle_id"]) not in llm_id_set
            ]
            metadata["interaction_tags"].extend((
                "related_background_traffic",
                "route_aligned_background",
                "longitudinal_cohort_background",
            ))
            if related_background["crossing_vehicle_ids"]:
                metadata["interaction_tags"].append(
                    "crossing_background")
            projected.append(SceneAsset(
                experiment_id=study,
                scene_id=scene_id,
                scenario=scenario,
                expected=expected,
            ))
    return projected


def generate_scene_catalog(output_dir: Path, *, source_root: Path | None = None,
                           clean: bool = False,
                           only_scene_ids: Sequence[str] | None = None,
                           reset_calibration_scene_ids: Sequence[str] = (),
                           ) -> dict:
    """Generate checked-in scene directories and return catalog metadata."""
    if clean and only_scene_ids is not None:
        raise ValueError("Partial scene regeneration cannot clean the existing catalog")
    output_dir = Path(output_dir)
    source_root = source_root or Path(__file__).resolve().parents[2]
    network_dir = source_root / "simulation" / "road_networks"
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    # A frozen SUMO-background calibration survives deterministic regeneration, but
    # only while the full physical scenario fingerprint remains identical.
    from evaluation.experiments.time_window_calibration import (
        DEFAULT_REGISTRY_PATH, apply_calibration_entry,
        load_calibration_registry,
    )
    calibration_registry = load_calibration_registry(DEFAULT_REGISTRY_PATH)
    schedule_durations = {
        scene_id: float(entry["time_limit_s"])
        for scene_id, entry in calibration_registry.get("entries", {}).items()
        if entry.get("max_successful_completion_s") is not None
    }
    reset_calibration_ids = {
        str(scene_id) for scene_id in reset_calibration_scene_ids}
    for scene_id in reset_calibration_ids:
        schedule_durations.pop(scene_id, None)
    if only_scene_ids is not None:
        # Partial regeneration discards freshly built payloads for every
        # unselected task and reloads their frozen JSON below.  Do not let a
        # grandfathered short calibration from one of those untouched tasks
        # make the provisional full-catalog build fail before that reload.
        selected_scene_ids = {str(scene_id) for scene_id in only_scene_ids}
        schedule_durations = {
            scene_id: duration
            for scene_id, duration in schedule_durations.items()
            if scene_id in selected_scene_ids
        }
    pattern_overrides = {}
    existing_by_id = {}
    existing_by_slot = {}
    if (output_dir / "catalog.json").is_file():
        existing = json.loads(
            (output_dir / "catalog.json").read_text())["entries"]
        existing_by_id = {entry["scene_id"]: entry for entry in existing}
        existing_by_slot = {
            (entry["experiment_id"],
             int(entry["scene_id"].split("_")[1])): entry
            for entry in existing
        }
        for entry in existing:
            for kind in ("weather", "daynight"):
                pattern = entry.get(f"{kind}_profile", {}).get("pattern")
                if pattern is not None:
                    pattern_overrides.setdefault(entry["scene_id"], {})[
                        kind] = pattern
        if only_scene_ids is not None:
            # build_study_assets constructs the full 200-scene portfolio
            # before unselected scenes are replaced by their frozen payloads.
            # Give those provisional builds the frozen deadline as well: an
            # old short template duration may be unable to contain the frozen
            # one/two-step environment pattern even though the checked-in task
            # has a valid calibrated window.
            selected_scene_ids = {
                str(scene_id) for scene_id in only_scene_ids}
            for entry in existing:
                scene_id = str(entry["scene_id"])
                if scene_id in selected_scene_ids:
                    continue
                frozen_path = output_dir / str(entry["scenario"])
                frozen = json.loads(frozen_path.read_text())
                schedule_durations[scene_id] = float(
                    frozen["total_time_s"])
    if only_scene_ids is not None:
        # Balance treatments within the edited subset without changing other
        # frozen tasks. A partial catalog refresh must not silently reassign
        # weather or day/night patterns throughout the benchmark.
        existing = list(existing_by_id.values())
        for study, prefix in (("Basic", "basic"), ("MultiLLM", "multi")):
            selected = sorted(s for s in only_scene_ids if s.startswith(prefix + "_"))
            selected_indices = {
                int(scene_id.split("_")[1]) for scene_id in selected}
            untouched = [e for e in existing if e["experiment_id"] == study
                         and int(e["scene_id"].split("_")[1])
                         not in selected_indices
                         and e["source_template_id"].split("__", 1)[0]
                         not in (
                             RETIRED_PASSING_TEMPLATES
                             | RETIRED_BASIC_STUDY_TEMPLATES
                             | RETIRED_MULTI_STUDY_TEMPLATES)]
            for kind, patterns, remainders in (
                    ("weather", FORMAL_WEATHER_PATTERNS, {1, 3}),
                    ("daynight", FORMAL_DAYNIGHT_PATTERNS, {2, 3})):
                slots = [s for s in selected if int(s.split("_")[1]) % 4 in remainders]
                counts = {p: sum(e[kind + "_profile"].get("pattern") == p
                                 for e in untouched) for p in patterns}
                # Rebuilding geometry for an already frozen scene must not
                # silently change its treatment. Only genuinely new IDs are
                # assigned to the current deficits.
                new_slots = []
                for scene_id in slots:
                    previous = (
                        existing_by_id.get(scene_id)
                        or existing_by_slot.get((
                            study, int(scene_id.split("_")[1]))))
                    if previous is None:
                        new_slots.append(scene_id)
                        continue
                    pattern = previous[kind + "_profile"].get("pattern")
                    if pattern is not None:
                        pattern_overrides.setdefault(scene_id, {})[
                            kind] = pattern
                        counts[pattern] += 1
                total = sum(counts.values()) + len(new_slots)
                targets = {p: total // 3 + int(i < total % 3) for i, p in enumerate(patterns)}
                for scene_id in new_slots:
                    pattern = max(patterns, key=lambda p: targets[p] - counts[p])
                    counts[pattern] += 1
                    pattern_overrides.setdefault(scene_id, {})[kind] = pattern
    assets = build_study_assets(
        network_dir, environment_schedule_duration_s=schedule_durations,
        environment_pattern_overrides=pattern_overrides)
    applied_calibrations = 0
    stale_calibrations = []
    for asset in assets:
        if only_scene_ids is not None and asset.scene_id not in only_scene_ids:
            directory = output_dir / asset.experiment_id / asset.scene_id
            asset.scenario.clear()
            asset.scenario.update(json.loads((directory / "scenario.json").read_text()))
            asset.expected.clear()
            asset.expected.update(json.loads((directory / "expected.json").read_text()))
            previous = calibration_registry["entries"].get(asset.scene_id)
            if previous is not None:
                if apply_calibration_entry(copy.deepcopy(asset.scenario), previous):
                    applied_calibrations += 1
                else:
                    stale_calibrations.append(asset.scene_id)
            continue
        calibration = (
            None if asset.scene_id in reset_calibration_ids
            else calibration_registry["entries"].get(asset.scene_id))
        if calibration is None:
            continue
        if apply_calibration_entry(asset.scenario, calibration):
            applied_calibrations += 1
        else:
            stale_calibrations.append(asset.scene_id)
    parameter_payload = load_experiment_parameters()
    _json_write(output_dir / "parameters.json", parameter_payload)
    entries = []
    for asset in assets:
        relative = Path(asset.experiment_id) / asset.scene_id
        target = output_dir / relative
        _json_write(target / "scenario.json", asset.scenario)
        _json_write(target / "expected.json", asset.expected)
        entries.append({
            "experiment_id": asset.experiment_id,
            "scene_id": asset.scene_id,
            "source_template_id": asset.scenario[
                "experiment_scene"]["source_template_id"],
            "focal_vehicle_id": asset.scenario[
                "experiment_scene"]["focal_vehicle_id"],
            "fixed_peer_vehicle_ids": asset.scenario[
                "experiment_scene"]["fixed_peer_vehicle_ids"],
            "scenario": str(relative / "scenario.json"),
            "expected": str(relative / "expected.json"),
            "network": asset.scenario["road_network_id"],
            "tags": asset.scenario["experiment_scene"]["interaction_tags"],
            "weather_profile": asset.scenario[
                "experiment_scene"]["weather_profile"],
            "daynight_profile": asset.scenario[
                "experiment_scene"]["daynight_profile"],
            "environment_group": asset.scenario[
                "experiment_scene"]["environment_group"],
            "related_background_traffic": {
                "schema": "vehiclearena-related-background-v1",
                "vehicle_ids": asset.scenario["experiment_scene"][
                    "related_background_traffic"]["vehicle_ids"],
                "longitudinal_cohort_vehicle_count": len(
                    asset.scenario["experiment_scene"][
                        "related_background_traffic"][
                            "longitudinal_cohort_vehicle_ids"]),
                "route_aligned_vehicle_count": len(
                    asset.scenario["experiment_scene"][
                        "related_background_traffic"][
                            "route_aligned_vehicle_ids"]),
                "crossing_vehicle_count": len(
                    asset.scenario["experiment_scene"][
                        "related_background_traffic"][
                            "crossing_vehicle_ids"]),
            },
        })

    basic_by_capability: Dict[str, List[dict]] = defaultdict(list)
    for entry in entries:
        if entry["experiment_id"] == "Basic":
            capability = entry["source_template_id"].split("__", 1)[0]
            basic_by_capability[capability].append(entry)
    split_distribution = {}
    environment_groups = (
        "control", "weather_only", "daynight_only", "combined")
    capability_options = {}
    for capability in sorted(basic_by_capability):
        capability_entries = basic_by_capability[capability]
        expected_total = (
            18 if capability in BASIC_WEIGHTED_CAPABILITIES else 9)
        if len(capability_entries) != expected_total:
            raise ValueError(
                f"Basic capability {capability} has "
                f"{len(capability_entries)}/{expected_total} tasks")
        train_count = (
            10 if capability in BASIC_WEIGHTED_CAPABILITIES else 5)
        split_distribution[capability] = {
            "total": expected_total,
            "train": train_count,
            "test": expected_total - train_count,
        }
        # Compress all valid within-capability choices by their four-group
        # environment count.  For equivalent count vectors retain the stable
        # lowest-hash choice, then solve the global 25/25/25/25 constraint.
        best_by_environment = {}
        ordered = sorted(
            capability_entries, key=lambda entry: entry["scene_id"])
        for combination in itertools.combinations(ordered, train_count):
            group_counts = tuple(sum(
                entry["environment_group"] == group
                for entry in combination) for group in environment_groups)
            scene_ids = tuple(entry["scene_id"] for entry in combination)
            rank = sum(_stable_number(
                "basic-capability-environment-split-v1:" + scene_id)
                for scene_id in scene_ids)
            candidate = (rank, scene_ids)
            if (group_counts not in best_by_environment
                    or candidate < best_by_environment[group_counts]):
                best_by_environment[group_counts] = candidate
        capability_options[capability] = best_by_environment

    target_environment_counts = (25, 25, 25, 25)
    split_states = {(0, 0, 0, 0): (0, ())}
    for capability in sorted(capability_options):
        next_states = {}
        for prior_counts, (prior_rank, prior_ids) in split_states.items():
            for group_counts, (rank, scene_ids) in capability_options[
                    capability].items():
                combined_counts = tuple(
                    first + second for first, second
                    in zip(prior_counts, group_counts))
                if any(
                        actual > target
                        for actual, target in zip(
                            combined_counts, target_environment_counts)):
                    continue
                candidate = (prior_rank + rank, prior_ids + scene_ids)
                if (combined_counts not in next_states
                        or candidate < next_states[combined_counts]):
                    next_states[combined_counts] = candidate
        split_states = next_states
    if target_environment_counts not in split_states:
        raise AssertionError(
            "Basic categories cannot satisfy the 25-per-environment split")
    train_scene_ids = sorted(
        split_states[target_environment_counts][1])
    all_basic_scene_ids = {
        entry["scene_id"] for entries in basic_by_capability.values()
        for entry in entries}
    test_scene_ids = sorted(all_basic_scene_ids - set(train_scene_ids))
    train_scene_ids.sort()
    test_scene_ids.sort()
    if len(train_scene_ids) != 100 or len(test_scene_ids) != 80:
        raise AssertionError("Basic split must contain exactly 100/80 tasks")
    basic_split = {
        "schema": "vehiclearena-basic-stratified-split-v1",
        "description": (
            "Deterministic capability-and-environment-stratified 100/80 "
            "split. Both subsets use the same normalized 2:1 weighting for "
            "the three emphasized capabilities and 1 for every other "
            "capability; all four environment groups are also proportional."),
        "seed": "basic-capability-environment-split-v1",
        "train_count": len(train_scene_ids),
        "test_count": len(test_scene_ids),
        "weighted_capabilities": sorted(BASIC_WEIGHTED_CAPABILITIES),
        "capability_distribution": split_distribution,
        "environment_distribution": {
            group: {"total": 45, "train": 25, "test": 20}
            for group in environment_groups
        },
        "train_scene_ids": train_scene_ids,
        "test_scene_ids": test_scene_ids,
    }
    _json_write(output_dir / "Basic" / "split.json", basic_split)
    matrix_specs = {
        "Basic": {
            "design": "single_llm_fixed_sumo_background",
            "model_assignment": "replace_focal_only",
            "personal_agent": "live_before_driver_wake",
        },
        "MultiLLM": {
            "design": "focal_model_marginal_effect",
            "model_assignment": "fixed_peers_replace_focal_only",
            "personal_agent": "live_before_driver_wake",
        },
    }
    for experiment_id, matrix in matrix_specs.items():
        experiment_entries = [
            entry for entry in entries
            if entry["experiment_id"] == experiment_id]
        matrix_payload = {
            "schema": CATALOG_SCHEMA,
            "experiment_id": experiment_id,
            "scene_ids": [entry["scene_id"] for entry in experiment_entries],
            "scene_count": len(experiment_entries),
            "task_count": len(experiment_entries),
            "network_ids": sorted({entry["network"]
                                   for entry in experiment_entries}),
            "source_template_ids": sorted({
                entry["source_template_id"]
                for entry in experiment_entries}),
            **matrix,
        }
        if experiment_id == "Basic":
            matrix_payload["recommended_split"] = "split.json"
        _json_write(output_dir / experiment_id / "matrix.json", matrix_payload)
        scene_lines = "\n".join(
            f"- `{entry['scene_id']}`: {', '.join(entry['tags'])}"
            for entry in experiment_entries)
        _text_write(
            output_dir / experiment_id / "README.md",
            f"# {experiment_id} 场景目录\n\n"
            f"物理场景定义在各子目录的 `scenario.json`，验收条件在 "
            f"`expected.json`，实验因子在 `matrix.json`。\n\n{scene_lines}")

    counts = {
        experiment_id: sum(
            entry["experiment_id"] == experiment_id for entry in entries)
        for experiment_id in ("Basic", "MultiLLM")
    }
    catalog = {
        "schema": CATALOG_SCHEMA,
        "experiment_parameters": {
            "schema": PARAMETER_SCHEMA,
            "path": "parameters.json",
            "sha256": parameter_fingerprint(),
        },
        "scene_count": len(entries),
        "counts": counts,
        "basic_split": {
            "path": "Basic/split.json",
            "schema": basic_split["schema"],
            "train_count": basic_split["train_count"],
            "test_count": basic_split["test_count"],
            "weighted_capabilities": basic_split[
                "weighted_capabilities"],
        },
        "networks": sorted({entry["network"] for entry in entries}),
        "entries": entries,
        "related_background_traffic": {
            "schema": "vehiclearena-related-background-v1",
            "minimum_per_task": RELATED_BACKGROUND_MINIMUM,
            "base_minimum_per_task": RELATED_BACKGROUND_BASE_MINIMUM,
            "longitudinal_cohort_minimum_per_task": (
                min(entry["related_background_traffic"][
                    "longitudinal_cohort_vehicle_count"]
                    for entry in entries)),
            "longitudinal_cohort_maximum_per_task": (
                max(entry["related_background_traffic"][
                    "longitudinal_cohort_vehicle_count"]
                    for entry in entries)),
            "task_count": len(entries),
            "task_count_with_crossing_traffic": sum(
                bool(entry["related_background_traffic"][
                    "crossing_vehicle_count"])
                for entry in entries),
            "added_vehicle_count": sum(
                len(entry["related_background_traffic"]["vehicle_ids"])
                for entry in entries),
            "longitudinal_cohort_vehicle_count": sum(
                entry["related_background_traffic"][
                    "longitudinal_cohort_vehicle_count"]
                for entry in entries),
            "route_aligned_vehicle_count": sum(
                entry["related_background_traffic"][
                    "route_aligned_vehicle_count"]
                for entry in entries),
            "crossing_vehicle_count": sum(
                entry["related_background_traffic"][
                    "crossing_vehicle_count"]
                for entry in entries),
        },
        "weather_coverage": {
            "transition_scene_count": sum(
                entry["weather_profile"]["mode"] == "transition"
                for entry in entries),
            "stable_control_scene_count": sum(
                entry["weather_profile"]["mode"] == "stable_control"
                for entry in entries),
            "target_conditions": list(FORMAL_WEATHER_TARGET_CONDITIONS),
            "wakeable_change_conditions": list(
                FORMAL_WEATHER_CONDITIONS),
            "pattern_counts": {
                pattern: sum(
                    entry["weather_profile"].get("pattern") == pattern
                    for entry in entries)
                for pattern in FORMAL_WEATHER_PATTERNS
            },
            "primary_condition_counts": {
                condition: sum(
                    entry["weather_profile"].get("primary_condition")
                    == condition for entry in entries)
                for condition in FORMAL_WEATHER_TARGET_CONDITIONS
            },
            "allowed_transition_edges": [
                list(edge) for edge in sorted(WEATHER_TRANSITION_EDGES)
            ],
            "location_profiles": {
                "schema": load_location_weather_profiles()["schema"],
                "path": "../location_weather_profiles.yaml",
                "reference_period": load_location_weather_profiles()[
                    "reference_period"],
                "location_count": len(load_location_weather_profiles()[
                    "locations"]),
            },
            "first_change_fraction": list(
                FORMAL_WEATHER_FIRST_CHANGE_FRACTION),
            "second_change_fraction": list(
                FORMAL_WEATHER_SECOND_CHANGE_FRACTION),
        },
        "daynight_coverage": {
            "transition_scene_count": sum(
                entry["daynight_profile"]["mode"] == "transition"
                for entry in entries),
            "stable_control_scene_count": sum(
                entry["daynight_profile"]["mode"] == "stable_control"
                for entry in entries),
            "periods": list(FORMAL_DAYNIGHT_PERIODS),
            "dark_target_periods": list(
                FORMAL_DAYNIGHT_DARK_PERIODS),
            "bright_periods": list(FORMAL_DAYNIGHT_BRIGHT_PERIODS),
            "pattern_counts": {
                pattern: sum(
                    entry["daynight_profile"].get("pattern") == pattern
                    for entry in entries)
                for pattern in FORMAL_DAYNIGHT_PATTERNS
            },
            "allowed_transition_edges": [
                list(edge) for edge in sorted(DAYNIGHT_EDGES)
            ],
            "joint_environment_scene_count": sum(
                bool(entry["daynight_profile"].get(
                    "joint_weather_change_times_s"))
                for entry in entries),
            "first_change_fraction": list(
                FORMAL_DAYNIGHT_FIRST_CHANGE_FRACTION),
            "second_change_fraction": list(
                FORMAL_DAYNIGHT_SECOND_CHANGE_FRACTION),
        },
        "environment_factorial": {
            group: sum(
                entry["environment_group"] == group for entry in entries)
            for group in (
                "control", "weather_only", "daynight_only", "combined")
        },
        "time_window_calibration": {
            "schema": calibration_registry.get("schema"),
            "registry": str(DEFAULT_REGISTRY_PATH),
            "applied_scene_count": applied_calibrations,
            "stale_scene_ids": stale_calibrations,
        },
    }
    _json_write(output_dir / "catalog.json", catalog)
    _text_write(
        output_dir / "README.md",
        "# VehicleArena 正式实验场景\n\n"
        f"当前只包含 Basic（{counts['Basic']} 个任务）和 "
        f"Multi-LLM（{counts['MultiLLM']} 个任务）。"
        "Online RL 暂不实现。\n\n"
        "每个场景目录包含 `scenario.json`（可执行物理初态）和 "
        "`expected.json`（机器可读验收条件）。每个实验的 `matrix.json` "
        "定义模型替换规则；公平比较依赖同一份冻结物理场景，不使用"
        "运行时随机种子。\n\n"
        "生成：`python -m evaluation.experiments.cli prepare-scenes "
        "--output vehiclearena/evaluation/experiments/scenarios`\n\n"
        "校验：`python -m evaluation.experiments.cli validate-scenes "
        "--catalog vehiclearena/evaluation/experiments/scenarios --boot`")
    return catalog


def iter_catalog_scenes(catalog_dir: Path) -> Iterable[tuple[dict, Path, Path]]:
    catalog_dir = Path(catalog_dir)
    catalog = json.loads(
        (catalog_dir / "catalog.json").read_text(encoding="utf-8"))
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"unsupported scene catalog schema {catalog.get('schema')!r}")
    for entry in catalog["entries"]:
        yield (
            entry,
            catalog_dir / entry["scenario"],
            catalog_dir / entry["expected"],
        )
