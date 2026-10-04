#!/usr/bin/env python3
"""Create an isolated seeded challenge catalog; never overwrite the original."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'vehiclearena'), str(ROOT), str(ROOT / 'vehiclearena/evaluation')]
from simulation.native_npc_behavior import with_native_npc_behavior


def check_native_calibration(catalog, registry_path, only=()):
    """Reject LLM suite preparation with uncalibrated or mismatched treatment."""
    from evaluation.experiments.time_window_calibration import (
        load_calibration_registry, scenario_physical_fingerprint)
    catalog = Path(catalog)
    entries = json.loads((catalog / 'catalog.json').read_text())['entries']
    registry = None
    for entry in entries:
        if only and entry['scene_id'] not in only:
            continue
        raw = json.loads((catalog / entry['scenario']).read_text())
        if not raw.get('sumo_config', {}).get('npc_behavior'):
            continue
        if registry_path is None:
            raise ValueError('Seeded NPC traffic requires --calibration-registry')
        if registry is None:
            registry = load_calibration_registry(Path(registry_path))
        baseline = registry.get('entries', {}).get(entry['scene_id'], {})
        if (baseline.get('physical_fingerprint') != scenario_physical_fingerprint(raw)
                or not baseline.get('successful_case_count')
                or baseline.get('time_limit_s') != raw.get('total_time_s')):
            raise ValueError(f"Fresh matching NPC calibration required: {entry['scene_id']}")


def prepare(catalog, output, seed, only=()):
    catalog, output = Path(catalog).resolve(), Path(output).resolve()
    source = json.loads((catalog / 'catalog.json').read_text())
    selected = set(only)
    available = {e['scene_id'] for e in source['entries']}
    if selected - available:
        raise ValueError(f'Unknown scene IDs: {sorted(selected - available)}')
    entries = [e for e in source['entries'] if not selected or e['scene_id'] in selected]
    prepared = []
    for entry in entries:
        paths = [Path(entry[k]) for k in ('scenario', 'expected')]
        if any(p.is_absolute() or '..' in p.parts for p in paths):
            raise ValueError('Catalog paths must remain inside the catalog')
        raw = json.loads((catalog / paths[0]).read_text())
        prepared.append((entry, with_native_npc_behavior(raw, seed)))
    output.mkdir(parents=True, exist_ok=False)
    for entry, raw in prepared:
        target = output / entry['scenario']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n')
        expected = output / entry['expected']
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(catalog / entry['expected'], expected)
    source = {'schema': source['schema'], 'entries': entries,
              'scene_count': len(entries),
              'counts': dict(Counter(e['experiment_id'] for e in entries)),
              'networks': sorted({e['network'] for e in entries})}
    source['npc_behavior'] = {'seed': seed, 'calibration_required': True,
                              'source_catalog': str(catalog)}
    (output / 'catalog.json').write_text(json.dumps(source, ensure_ascii=False, indent=2) + '\n')
    return len(entries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--only', action='append', default=[])
    args = parser.parse_args()
    count = prepare(args.catalog, args.output, args.seed, args.only)
    print(f'Prepared {count} scenes in {args.output}; recalibrate before evaluation.')
