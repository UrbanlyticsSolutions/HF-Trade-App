"""
Optimization Config Manager — Store, Track, Apply, Rollback

Central place to manage optimization history and sync with strategy.json.

Commands:
  python -m config.config_manager list                 # Show all saved runs
  python -m config.config_manager show <run_id>        # Show details of a run
  python -m config.config_manager apply <run_id>       # Apply a run to strategy.json
  python -m config.config_manager rollback             # Revert to previous config
  python -m config.config_manager diff <id1> <id2>     # Compare two runs
  python -m config.config_manager current              # Show what's active now

Optimizers call:
  save_optimization_run(source, trade_config, risk_config, results, params)
"""
import sys
sys.path.insert(0, '.')

import json
import time
import copy
import argparse
from pathlib import Path

STRATEGY_PATH = Path('config/strategy.json')
HISTORY_PATH = Path('config/optimization_history.json')


# ============================================================
# CORE: Load / Save history
# ============================================================

def _load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {'runs': [], 'active_run_id': None}


def _save_history(history):
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def _load_strategy():
    with open(STRATEGY_PATH) as f:
        return json.load(f)


def _save_strategy(cfg):
    with open(STRATEGY_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)


# ============================================================
# API: Called by optimizers to record a run
# ============================================================

def save_optimization_run(source, trade_config, risk_config=None,
                          results=None, metadata=None):
    """
    Save an optimization run to history.

    Args:
        source: str — which optimizer produced this ('optimize_full', 'optimize_regime_v2', 'optimize')
        trade_config: dict — the trade_config params from the best trial
        risk_config: dict or None — risk_config params (if optimized)
        results: dict or None — performance metrics (IS, OOS, etc.)
        metadata: dict or None — extra info (trials, periods, etc.)

    Returns:
        run_id: int — the ID of the saved run
    """
    history = _load_history()

    run_id = len(history['runs']) + 1

    # Snapshot the current strategy.json at time of save
    current_cfg = _load_strategy()
    snapshot = {
        'trade_config': current_cfg.get('trade_config', {}),
        'risk_config': current_cfg.get('risk_config', {}),
    }

    run = {
        'run_id': run_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': source,
        'applied': False,
        'trade_config': trade_config,
        'risk_config': risk_config or {},
        'results': results or {},
        'metadata': metadata or {},
        'previous_snapshot': snapshot,
    }

    history['runs'].append(run)
    _save_history(history)

    print(f"  [config_manager] Saved run #{run_id} from {source}")
    return run_id


# ============================================================
# API: Apply a run to strategy.json
# ============================================================

def apply_run(run_id, dry_run=False):
    """
    Merge a saved run's trade_config and risk_config into strategy.json.

    Only updates keys that exist in the run — does not delete existing keys.
    Returns a dict of {key: (old_value, new_value)} changes.
    """
    history = _load_history()
    run = _find_run(history, run_id)
    if not run:
        print(f"  ERROR: Run #{run_id} not found")
        return None

    cfg = _load_strategy()
    changes = {}

    # Merge trade_config
    tc = cfg.setdefault('trade_config', {})
    for k, v in run['trade_config'].items():
        old = tc.get(k)
        if old != v:
            changes[f'trade_config.{k}'] = (old, v)
            if not dry_run:
                tc[k] = v

    # Merge risk_config
    if run.get('risk_config'):
        rc = cfg.setdefault('risk_config', {})
        for k, v in run['risk_config'].items():
            if v is None:
                continue
            old = rc.get(k)
            if old != v:
                changes[f'risk_config.{k}'] = (old, v)
                if not dry_run:
                    rc[k] = v

    # Update optimized_results metadata
    if not dry_run and run.get('results'):
        cfg['optimized_results'] = {
            'optimization_date': run['timestamp'].split(' ')[0],
            'source': run['source'],
            'run_id': run_id,
            **run['results'],
        }

    if dry_run:
        print(f"\n  DRY RUN — would change {len(changes)} keys:")
        _print_changes(changes)
        return changes

    if not changes:
        print(f"  No changes — run #{run_id} matches current config")
        return changes

    # Save snapshot before applying
    _save_strategy(cfg)

    # Mark as applied, unmark any previous
    for r in history['runs']:
        if r.get('applied'):
            r['applied'] = False
    run['applied'] = True
    run['applied_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    history['active_run_id'] = run_id
    _save_history(history)

    print(f"\n  Applied run #{run_id} ({run['source']}) -> strategy.json")
    print(f"  Changed {len(changes)} keys:")
    _print_changes(changes)
    return changes


def rollback_run(run_id=None):
    """
    Rollback to the state before a run was applied.
    If run_id is None, rolls back the currently active run.
    """
    history = _load_history()

    if run_id is None:
        run_id = history.get('active_run_id')
    if run_id is None:
        print("  ERROR: No active run to rollback")
        return False

    run = _find_run(history, run_id)
    if not run:
        print(f"  ERROR: Run #{run_id} not found")
        return False

    snapshot = run.get('previous_snapshot')
    if not snapshot:
        print(f"  ERROR: Run #{run_id} has no snapshot to rollback to")
        return False

    cfg = _load_strategy()
    cfg['trade_config'] = snapshot['trade_config']
    cfg['risk_config'] = snapshot['risk_config']
    _save_strategy(cfg)

    run['applied'] = False
    history['active_run_id'] = None
    _save_history(history)

    print(f"  Rolled back run #{run_id} — restored previous config")
    return True


# ============================================================
# CLI: List / Show / Diff
# ============================================================

def list_runs():
    history = _load_history()
    runs = history['runs']
    if not runs:
        print("  No optimization runs saved yet.")
        return

    active_id = history.get('active_run_id')
    print(f"\n  {'ID':>4}  {'Date':>19}  {'Source':<22}  {'Applied':>7}  {'Summary'}")
    print(f"  {'─'*90}")

    for r in runs:
        marker = ' *' if r['run_id'] == active_id else '  '
        applied = 'YES' if r.get('applied') else ''
        res = r.get('results', {})
        # Build summary from whatever results are available
        parts = []
        if 'is_results' in res:
            ir = res['is_results']
            parts.append(f"IS: WR={ir.get('wr', ir.get('win_rate', '?'))}%")
            parts.append(f"Ret={ir.get('ret', ir.get('return_pct', '?'))}%")
        elif 'train_2025' in res:
            tr = res['train_2025']
            parts.append(f"IS: WR={tr.get('win_rate', '?')}%")
            parts.append(f"Ret={tr.get('return_pct', '?')}%")
        if 'oos_results' in res:
            oo = res['oos_results']
            parts.append(f"OOS: Ret={oo.get('ret', oo.get('return_pct', '?'))}%")
        summary = ', '.join(parts) if parts else r['source']

        print(f"{marker}{r['run_id']:>4}  {r['timestamp']:>19}  {r['source']:<22}  {applied:>7}  {summary}")

    print(f"\n  * = currently active in strategy.json")


def show_run(run_id):
    history = _load_history()
    run = _find_run(history, run_id)
    if not run:
        print(f"  ERROR: Run #{run_id} not found")
        return

    print(f"\n  Run #{run['run_id']} — {run['source']}")
    print(f"  Date: {run['timestamp']}")
    print(f"  Applied: {run.get('applied', False)}")

    print(f"\n  Trade Config:")
    for k, v in sorted(run['trade_config'].items()):
        print(f"    {k}: {v}")

    if run.get('risk_config'):
        print(f"\n  Risk Config:")
        for k, v in sorted(run['risk_config'].items()):
            print(f"    {k}: {v}")

    if run.get('results'):
        print(f"\n  Results:")
        _print_nested(run['results'], indent=4)

    if run.get('metadata'):
        print(f"\n  Metadata:")
        for k, v in run['metadata'].items():
            print(f"    {k}: {v}")


def diff_runs(id1, id2):
    history = _load_history()
    r1 = _find_run(history, id1)
    r2 = _find_run(history, id2)
    if not r1 or not r2:
        print(f"  ERROR: Run not found")
        return

    print(f"\n  Comparing Run #{id1} vs Run #{id2}")
    print(f"  {'─'*70}")

    # Diff trade_config
    all_keys = sorted(set(list(r1['trade_config'].keys()) + list(r2['trade_config'].keys())))
    print(f"\n  {'Key':<35} {'Run #' + str(id1):>15} {'Run #' + str(id2):>15}")
    print(f"  {'─'*65}")
    for k in all_keys:
        v1 = r1['trade_config'].get(k)
        v2 = r2['trade_config'].get(k)
        if v1 != v2:
            print(f"  {k:<35} {_fmt(v1):>15} {_fmt(v2):>15}  <--")
        else:
            print(f"  {k:<35} {_fmt(v1):>15} {_fmt(v2):>15}")

    # Diff results
    res1 = r1.get('results', {})
    res2 = r2.get('results', {})
    if res1 or res2:
        print(f"\n  Results:")
        for section in ['is_results', 'oos_results']:
            s1 = res1.get(section, {})
            s2 = res2.get(section, {})
            if s1 or s2:
                print(f"\n    {section}:")
                all_k = sorted(set(list(s1.keys()) + list(s2.keys())))
                for k in all_k:
                    v1 = s1.get(k)
                    v2 = s2.get(k)
                    marker = '  <--' if v1 != v2 else ''
                    print(f"      {k:<25} {_fmt(v1):>12} {_fmt(v2):>12}{marker}")


def show_current():
    """Show what's currently active in strategy.json."""
    history = _load_history()
    active_id = history.get('active_run_id')

    cfg = _load_strategy()
    opt = cfg.get('optimized_results', {})

    print(f"\n  Active Run ID: {active_id or 'None (manual edits)'}")
    print(f"  Optimization Date: {opt.get('optimization_date', 'unknown')}")
    print(f"  Source: {opt.get('source', opt.get('optimization_phase', 'unknown'))}")

    tc = cfg.get('trade_config', {})
    key_params = [
        'strategy', 'profit_target_pct', 'stop_loss_pct', 'max_hold_bars',
        'call_profit_target_pct', 'put_profit_target_pct',
        'call_stop_loss_pct', 'put_stop_loss_pct',
        'rsi_call_threshold', 'rsi_put_threshold',
        'use_regime_detection', 'post_loss_strategy',
    ]
    print(f"\n  Key trade params:")
    for k in key_params:
        if k in tc:
            print(f"    {k}: {tc[k]}")

    rc = cfg.get('risk_config', {})
    print(f"\n  Risk params:")
    for k, v in sorted(rc.items()):
        print(f"    {k}: {v}")


# ============================================================
# Helpers
# ============================================================

def _find_run(history, run_id):
    for r in history['runs']:
        if r['run_id'] == run_id:
            return r
    return None


def _print_changes(changes):
    for key, (old, new) in sorted(changes.items()):
        print(f"    {key}: {_fmt(old)} -> {_fmt(new)}")


def _fmt(v):
    if v is None:
        return 'null'
    if isinstance(v, float):
        if abs(v) < 1:
            return f'{v:.4f}'
        return f'{v:.2f}'
    return str(v)


def _print_nested(d, indent=2):
    prefix = ' ' * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            _print_nested(v, indent + 2)
        else:
            print(f"{prefix}{k}: {v}")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Optimization Config Manager')
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('list', help='List all saved runs')
    sub.add_parser('current', help='Show current active config')

    p_show = sub.add_parser('show', help='Show details of a run')
    p_show.add_argument('run_id', type=int)

    p_apply = sub.add_parser('apply', help='Apply a run to strategy.json')
    p_apply.add_argument('run_id', type=int)
    p_apply.add_argument('--dry-run', action='store_true', help='Preview changes without applying')

    p_rollback = sub.add_parser('rollback', help='Rollback to previous config')
    p_rollback.add_argument('run_id', type=int, nargs='?', default=None)

    p_diff = sub.add_parser('diff', help='Compare two runs')
    p_diff.add_argument('id1', type=int)
    p_diff.add_argument('id2', type=int)

    args = parser.parse_args()

    if args.command == 'list':
        list_runs()
    elif args.command == 'current':
        show_current()
    elif args.command == 'show':
        show_run(args.run_id)
    elif args.command == 'apply':
        apply_run(args.run_id, dry_run=args.dry_run)
    elif args.command == 'rollback':
        rollback_run(args.run_id)
    elif args.command == 'diff':
        diff_runs(args.id1, args.id2)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
