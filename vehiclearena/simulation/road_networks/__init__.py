"""
Road-network registry and loader for VehicleArena.

The real OpenStreetMap JSON data is distributed separately as a versioned
offline bundle.  Install it before running map-backed simulations; see
``vehiclearena/simulation/MAPS.md``.

Load a road network by ID:

    from simulation.road_networks import load_road_network
    net = load_road_network("beijing_zhongguancun")

Available IDs (21 maps, 17 cities):
    - beijing_zhongguancun  365 nodes  Haidian/university district
    - beijing_wangjing      485 nodes  Wangjing residential + commercial
    - beijing_guomao        450 nodes  Guomao CBD (east 3rd ring)
    - beijing_tiananmen     500 nodes  Tiananmen/Dongcheng
    - shanghai_lujiazui     179 nodes  Lujiazui financial district
    - shanghai_xujiahui     385 nodes  Xujiahui commercial/interchange
    - guangzhou_tianhe      400 nodes  Tianhe CBD
    - shenzhen_nanshan      400 nodes  Nanshan tech park
    - chengdu_chunxi        400 nodes  Chunxi Road commercial
    - chongqing_jiefangbei  297 nodes  Jiefangbei (hilly topology)
    - hangzhou_xihu          244 nodes  West Lake district
    - nanjing_xinjiekou     400 nodes  Xinjiekou (roundabouts)
    - wuhan_guanggu         400 nodes  Optics Valley new district
    - xian_zhonglou         400 nodes  Bell Tower (grid layout)
    - suzhou_guanqian       389 nodes  Guanqian old town
    - tianjin_heping        400 nodes  Heping (radial roads)
    - changsha_wuyi         400 nodes  Wuyi Square
    - zhengzhou_erqi        352 nodes  Erqi Square
    - hefei_zhengwu         298 nodes  Zhengwu new district (wide roads)
    - dalian_zhongshan      400 nodes  Zhongshan Square (radial)
    - xiamen_zhongshan      230 nodes  Zhongshan Road (island city)

Each area defines `key_nodes` — pre-selected intersections for scenario
design (route start/end points):

    from simulation.road_networks import get_key_node, list_key_nodes
    start = get_key_node("beijing_zhongguancun", "zhongguancun_north")
    end   = get_key_node("beijing_zhongguancun", "zhongguancun_south")

To regenerate or add new areas:

    from simulation.osm_import import import_from_point
    net = import_from_point(lat, lng, dist_meters=1000)
    net.save_json("simulation/road_networks/my_area.json")
"""

from __future__ import annotations

import json
import math
import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_NETWORK_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_MANIFEST = os.path.join(
    os.path.dirname(_NETWORK_DIR), "map_bundle_manifest.json")


# ── Area registry with key nodes ────────────────────────────────

_AREA_CONFIGS = {
    "beijing_zhongguancun": {
        "city": "Beijing",
        "area": "Zhongguancun",
        "center": (39.9836, 116.3174),
        "key_nodes": {
            "zhongguancun_north": "n286387102",    # 中关村北大街北端
            "zhongguancun_south": "n240420386",    # 中关村东路南端 (信号灯)
            "north_4th_ring": "n245130319",        # 北四环西路 (信号灯)
            "haidian_street": "n1498572922",       # 海淀大街
            "chengfu_road_east": "n32618842",      # 成府路东
            "zhichun_road": "n1687193595",         # 知春路
            "zhongguancun_plaza": "n1877984745",   # 中关村北二街 (4叉路口)
            "south_road_west": "n494296089",       # 中关村南路西端
            "south_road_east": "n494296099",       # 中关村南路东端 (信号灯)
            "north_2nd_street": "n245130315",      # 中关村北二街 (信号灯, 4叉)
            "donglu_mid": "n9280806130",           # 中关村东路中段
            "chengfu_mid": "n1532866063",          # 成府路中段
        },
    },
    "shanghai_lujiazui": {
        "city": "Shanghai",
        "area": "Lujiazui",
        "center": (31.2400, 121.5000),
        "key_nodes": {
            "dongcheng_road": "n84466485",         # 东城路 (4叉路口)
            "dongyuan_road": "n84466496",          # 东园路
            "fucheng_road": "n84487663",           # 富城路 (信号灯)
            "pudong_south": "n93438508",           # 浦东南路
            "fenghe_road": "n93466273",            # 丰和路
            "dongchang_road": "n93792440",         # 东昌路 (信号灯)
            "shibu_street": "n84488167",           # 拾步街
        },
    },
    "beijing_wangjing": {
        "city": "Beijing",
        "area": "Wangjing",
        "center": (39.9920, 116.4740),
        "key_nodes": {
            "wangtong_road_n": "n4684883821",      # 望通路北 (4叉)
            "wangtong_road_s": "n4684883822",      # 望通路南 (4叉)
            "wangda_road_n": "n4684883825",        # 望达路北 (4叉)
            "wangda_road_s": "n4684883826",        # 望达路南 (4叉)
            "guangshun_north": "n5235827185",      # 广顺北
            "south_area": "n1239554945",           # 南区
            "center": "n1239554909",               # 中心 (4叉)
        },
    },
    "beijing_tiananmen": {
        "city": "Beijing",
        "area": "Tiananmen/Dongcheng",
        "center": (39.9100, 116.3970),
        "key_nodes": {
            "beiheyan_north": "n733788595",        # 北河沿大街北
            "dongsi_south": "n266111539",          # 东四南大街 (5叉)
            "beichizi": "n31194142",               # 北池子大街 (信号灯)
            "donganmen": "n31194143",              # 东安门大街 (信号灯)
            "dongxinglong": "n12652373913",        # 东兴隆街南
            "chongwenmen": "n237270566",           # 崇文门外大街
            "dongdan": "n266109099",               # 东单北大街 (4叉)
        },
    },
    "beijing_guomao": {
        "city": "Beijing",
        "area": "Guomao CBD",
        "center": (39.911, 116.457),
        "key_nodes": {
            "guanghua_road": "n33399858",          # 光华路
            "chaoyang_road": "n35722739",          # 朝阳路
            "east_3rd_ring": "n36704818",          # 东三环中路
            "langjia_west": "n2207734183",         # 郎家园西路
            "langjia_road": "n2207734187",         # 郎家园路
            "jintong_west": "n321803111",          # 金桐西路
        },
    },
    "shanghai_xujiahui": {
        "city": "Shanghai",
        "area": "Xujiahui",
        "center": (31.1904, 121.4388),
        "key_nodes": {
            "yongjia_road": "n85014614",           # 永嘉路
            "fanyu_road": "n107669631",            # 番禺路
            "hongqiao_road": "n107673297",         # 虹桥路
            "wanping_road": "n601712348",          # 宛平路
            "xietu_road": "n243843501",            # 斜土路
            "nandan_road": "n287798155",           # 南丹路
        },
    },
    "guangzhou_tianhe": {
        "city": "Guangzhou",
        "area": "Tianhe CBD",
        "center": (23.129, 113.3187),
        "key_nodes": {
            "zhongshan_yi": "n4414727440",         # 中山一路
            "huaqiang_road": "n3017910562",        # 华强路
            "tianhe_street": "n3293905661",        # 天河街
            "huaming_street": "n5070078282",       # 华明直街
            "center_signal": "n6552039674",        # 中心信号灯
        },
    },
    "shenzhen_nanshan": {
        "city": "Shenzhen",
        "area": "Nanshan Tech Park",
        "center": (22.5326, 113.9305),
        "key_nodes": {
            "xuefu_road": "n830984674",            # 学府路
            "nanguang_road": "n1169613964",        # 南光路
            "wenxin_2nd": "n1169612816",           # 文心二路
            "shenда_north": "n2400176747",         # 深大北门路
        },
    },
    "chengdu_chunxi": {
        "city": "Chengdu",
        "area": "Chunxi Road",
        "center": (30.6561, 104.0831),
        "key_nodes": {
            "binjiang_east": "n963549497",         # 滨江东路
            "dongjiaochang": "n1958204318",        # 东较场街
            "fude_street": "n3516900121",          # 福德街
            "tianxianqiao_south": "n4550018635",   # 天仙桥南路
            "cihuitang": "n1159131825",            # 慈惠堂街
            "huaxing_street": "n1159148556",       # 华兴上街
            "dianjiang_street": "n1159950981",     # 点将台街
        },
    },
    "chongqing_jiefangbei": {
        "city": "Chongqing",
        "area": "Jiefangbei",
        "center": (29.5599, 106.5734),
        "key_nodes": {
            "dongshuimen_bridge": "n11070301630",  # 东水门大桥
            "fenghuangtai": "n733200734",          # 凤凰台路
            "kaixuan_road": "n733201355",          # 凯旋路
            "mianhua_street": "n733200674",        # 棉花街
            "huoyaoju": "n733200694",              # 火药局街
            "chuqimen": "n733200736",              # 储奇门顺城街
        },
    },
    "hangzhou_xihu": {
        "city": "Hangzhou",
        "area": "Xihu District",
        "center": (30.2627, 120.1263),
        "key_nodes": {
            "hangda_road": "n25954993",            # 杭大路
            "qiushi_road": "n25982050",            # 求是路
            "zheda_road": "n25982051",             # 浙大路
            "yugu_road": "n26202700",              # 玉古路
            "beishan_street": "n26202715",         # 北山街
            "yuquan_road": "n26560314",            # 玉泉路
        },
    },
    "nanjing_xinjiekou": {
        "city": "Nanjing",
        "area": "Xinjiekou",
        "center": (32.0429, 118.7875),
        "key_nodes": {
            "zhujiang_road": "n428787925",         # 珠江路
            "doucaiqiao": "n1120932553",           # 豆菜桥
            "hongwu_road": "n4414643356",          # 洪武路
            "danfeng_street": "n305372994",        # 丹凤街
            "jinxianghe": "n428787926",            # 进香河路
            "taiping_south": "n770984123",         # 太平南路
        },
    },
    "wuhan_guanggu": {
        "city": "Wuhan",
        "area": "Guanggu/Optics Valley",
        "center": (30.5072, 114.4111),
        "key_nodes": {
            "zijing_road": "n1124484612",          # 紫荆路
            "zuiwan_road": "n1124730362",          # 醉晚路
            "gaoxin_avenue": "n1143965868",        # 高新大道
            "hongmian_road": "n1124476637",        # 红棉路
            "huazhong_road": "n1124476778",        # 华中路
        },
    },
    "xian_zhonglou": {
        "city": "Xi'an",
        "area": "Bell Tower",
        "center": (34.2593, 108.9471),
        "key_nodes": {
            "dache_alley": "n310348091",           # 大车家巷
            "xixin_street": "n491836260",          # 西新街
            "xiyi_road": "n699662678",             # 西一路
            "beiguangji": "n491836254",            # 北广济街
            "shuncheng_south": "n733110572",       # 顺城南路西段
            "baishulin": "n733110581",             # 柏树林
        },
    },
    "suzhou_guanqian": {
        "city": "Suzhou",
        "area": "Guanqian Street",
        "center": (31.3028, 120.6248),
        "key_nodes": {
            "daoqian_street": "n265596759",        # 道前街
            "renmin_road": "n266180816",           # 人民路
            "yangyu_alley": "n430312139",          # 养育巷
            "moye_road": "n566834114",             # 莫邪路
            "shiqi_alley": "n266783835",           # 侍其巷
            "minzhi_road": "n803318164",           # 民治路
        },
    },
    "tianjin_heping": {
        "city": "Tianjin",
        "area": "Heping District",
        "center": (39.1198, 117.202),
        "key_nodes": {
            "shanxi_road": "n269660177",           # 山西路
            "yingkou_dao": "n267786377",           # 营口道
            "center_1": "n8047231050",             # 中心路口
            "center_2": "n12514109953",            # 和平路口
            "center_3": "n12514109957",            # 滨江道口
        },
    },
    "changsha_wuyi": {
        "city": "Changsha",
        "area": "Wuyi Square",
        "center": (28.194, 112.9763),
        "key_nodes": {
            "yichang_street": "n1955253530",       # 怡长街
            "sanwang_street": "n1955253563",       # 三王街
            "wuwang_street": "n1955293084",        # 乌王街
            "baisha_road": "n1955841455",          # 白沙路
            "santai_street": "n4448731556",        # 三泰街
            "dongqing_street": "n1004215054",      # 东庆街
        },
    },
    "zhengzhou_erqi": {
        "city": "Zhengzhou",
        "area": "Erqi Square",
        "center": (34.749, 113.6584),
        "key_nodes": {
            "erdao_street": "n6796155338",         # 二道街
            "xichenzhuang": "n6852788756",         # 西陈庄前街
            "gongsan_street": "n6796154246",       # 工三街
            "shuncheng_street": "n6796154248",     # 顺城街
            "tuanjie_road": "n6796155260",         # 团结路
        },
    },
    "hefei_zhengwu": {
        "city": "Hefei",
        "area": "Zhengwu District",
        "center": (31.821, 117.2534),
        "key_nodes": {
            "wuxi_road": "n2959743543",            # 五溪路
            "yinxing_avenue": "n5006735941",       # 银杏大道
            "guohuai_road": "n5006735948",         # 国槐路
            "mutong_road": "n5006736065",          # 牧童路
            "daloushan_road": "n5006736087",       # 大楼山路
        },
    },
    "dalian_zhongshan": {
        "city": "Dalian",
        "area": "Zhongshan Square",
        "center": (38.9173, 121.6408),
        "key_nodes": {
            "ziwei_street": "n1750571060",         # 自卫街
            "shiji_street": "n430657687",          # 世纪街
            "anyang_street": "n613659278",         # 安阳街
            "jiqing_street": "n614249712",         # 吉庆街
            "qingshuang_street": "n614969766",     # 清爽街
            "mingze_street": "n614969805",         # 明泽街
        },
    },
    "xiamen_zhongshan": {
        "city": "Xiamen",
        "area": "Zhongshan Road",
        "center": (24.4503, 118.0809),
        "key_nodes": {
            "huyuan_road": "n1223210785",          # 虎园路
            "siming_south": "n1223211532",         # 思明南路
            "yanwu_road": "n5327961286",           # 演武路
            "shengping_road": "n1223210121",       # 升平路
            "dazhong_road": "n1223210213",         # 大中路
        },
    },
}


# Register the complete catalog from the small tracked manifest even when the
# large map bundle has not been installed.  Fall back to locally present base
# maps for developer-created bundles.  Lane-level companions are data for a
# base map, not separate network IDs.
_catalog_names = set()
try:
    with open(_BUNDLE_MANIFEST, "r", encoding="utf-8") as _manifest_stream:
        _catalog_names.update(
            json.load(_manifest_stream).get("base_networks", []))
except (OSError, ValueError, TypeError):
    logger.debug("Map bundle manifest unavailable: %s", _BUNDLE_MANIFEST)

for _f in os.listdir(_NETWORK_DIR):
    if _f.endswith(".json") and not _f.endswith("_lane_level.json"):
        _catalog_names.add(_f[:-5])

for _name in sorted(_catalog_names):
    if _name not in _AREA_CONFIGS:
        _AREA_CONFIGS[_name] = {
            "city": "auto",
            "area": _name,
            "center": (0.0, 0.0),
            "key_nodes": {},
        }


# ── Public API ──────────────────────────────────────────────────

def load_road_network(network_id: str) -> "RoadNetwork":
    """Load a pre-built road network by ID.

    Args:
        network_id: One of the registered area IDs (see module docstring).

    Returns:
        RoadNetwork instance ready for simulation.

    Raises:
        ValueError: If network_id is unknown or its offline data is missing.
    """
    from simulation.road_network import RoadNetwork

    if network_id not in _AREA_CONFIGS:
        raise ValueError(
            f"Unknown road_network_id: {network_id!r}. "
            f"Available: {sorted(_AREA_CONFIGS)}"
        )

    json_path = os.path.join(_NETWORK_DIR, f"{network_id}.json")
    if not os.path.exists(json_path):
        raise ValueError(
            f"Road network JSON not found: {json_path}. "
            "The map data is distributed separately from Git. Install the "
            "verified offline bundle with: python "
            "scripts/manage_map_bundle.py install --source <bundle.tar.gz> "
            "--replace. See vehiclearena/simulation/MAPS.md."
        )

    logger.info(f"Loading OSM road network: {json_path}")
    network = RoadNetwork.load_json(json_path)
    network.network_id = network_id
    lane_level_path = os.path.join(
        _NETWORK_DIR, f"{network_id}_lane_level.json")
    network.lane_level_path = (
        lane_level_path if os.path.exists(lane_level_path) else "")
    return network


def list_available() -> list:
    """Return list of available road network IDs."""
    return sorted(_AREA_CONFIGS.keys())


def get_key_node(network_id: str, key: str) -> str:
    """Get a pre-defined key node ID by semantic name.

    Args:
        network_id: Road network ID (e.g. "beijing_zhongguancun").
        key: Semantic node name (e.g. "zhongguancun_north").

    Returns:
        OSM node ID string (e.g. "n286387102").

    Raises:
        KeyError: If network_id or key not found.
    """
    cfg = _AREA_CONFIGS.get(network_id)
    if cfg is None:
        raise KeyError(f"Unknown network: {network_id!r}")
    nodes = cfg.get("key_nodes", {})
    if key not in nodes:
        raise KeyError(
            f"Unknown key node '{key}' for {network_id}. "
            f"Available: {sorted(nodes)}"
        )
    return nodes[key]


def list_key_nodes(network_id: str) -> dict:
    """List all pre-defined key nodes for a network.

    Returns:
        Dict of {semantic_name: node_id}.
    """
    cfg = _AREA_CONFIGS.get(network_id)
    if cfg is None:
        raise KeyError(f"Unknown network: {network_id!r}")
    return dict(cfg.get("key_nodes", {}))


def find_nearest_node(network: "RoadNetwork", lat: float, lng: float) -> str:
    """Find the nearest node to a coordinate.

    Args:
        network: A loaded RoadNetwork instance.
        lat: Latitude.
        lng: Longitude.

    Returns:
        Node ID of the nearest node.
    """
    best_nid = None
    best_dist = float('inf')
    for nid, node in network.nodes.items():
        # Simple Euclidean on lat/lng (sufficient for local areas)
        d = math.sqrt((node.lat - lat) ** 2 + (node.lng - lng) ** 2)
        if d < best_dist:
            best_dist = d
            best_nid = nid
    return best_nid
