"""
0DTE Strategy Optimizer — Fast Parallel Edition
=================================================
Pre-computes data once, then sweeps parameters in parallel across CPU cores.

Architecture:
  1. Load data once → compute features for all orb_minutes values
  2. Serialize to pickle for worker process access
  3. Run parameter combos in parallel using multiprocessing.Pool
  4. Each worker: creates fresh Backtest0DTE, runs run_no_ml(), returns metrics

Phases:
  1. Coarse grid: 5 strategies × exit params → 625 combos
  2. Expand top N: option price ranges × trading windows → ~360 combos
  3. Fine-tune: narrow grid around best → ~2000+ combos
  4. Advanced exits: trailing stop / time decay / quick exit

Usage:
  python scripts/optimize.py                    # Full (all 4 phases)
  python scripts/optimize.py --phase 1          # Phase 1 only
  python scripts/optimize.py --phase 2          # Phase 2 (needs Phase 1)
  python scripts/optimize.py --apply            # Apply best config
  python scripts/optimize.py --workers 12       # Limit workers
  python scripts/optimize.py --sequential       # Debug: run single-threaded
"""
import sys
sys.path.insert(0, '.')

import json
import time
import argparse
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from backtest.engine import Backtest0DTE, TradeConfig, Trade0DTE
from core.risk_manager import RiskManager, RiskConfig
from core.signals import compute_features
from config.config_manager import save_optimization_run

# ============================================================
# WORKER GLOBALS (initialized per-process)
# ============================================================
_W = {}  # worker state dict


def _init_worker(underlying_pkl, options_pkl, features_pkl, vol_pkl,
                 initial_capital, slippage, commission):
    """Initialize worker process with data from pickle files."""
    _W['underlying'] = pd.read_pickle(underlying_pkl)
    _W['options'] = pd.read_pickle(options_pkl)
    _W['features'] = pd.read_pickle(features_pkl)
    _W['vol'] = pd.read_pickle(vol_pkl)
    _W['capital'] = initial_capital
    _W['slippage'] = slippage
    _W['commission'] = commission
    
    # Pre-build option indexes ONCE per worker (biggest speedup)
    opts = _W['options']
    _W['opt_by_date_time'] = {k: v for k, v in opts.groupby(['date', 'time'])}
    _W['opt_by_ticker_date'] = {k: v.sort_values('time') for k, v in opts.groupby(['option_ticker', 'date'])}


def _run_one(params: dict) -> dict:
    """Run single backtest in worker process. Returns metrics dict."""
    # Build TradeConfig from params
    valid_fields = set(TradeConfig.__dataclass_fields__.keys())
    tc_kwargs = {k: v for k, v in params.items() if k in valid_fields}
    trade_cfg = TradeConfig(**tc_kwargs)

    # Build RiskConfig — accept overrides from params
    risk_fields = set(RiskConfig.__dataclass_fields__.keys())
    rc_kwargs = {k: v for k, v in params.items() if k in risk_fields}
    rc_defaults = dict(
        kelly_fraction=0.20,
        max_daily_losses=2,
        max_consecutive_losses=999,
        reduce_size_at_dd_pct=0.99,
        max_daily_loss_pct=0.99,
        max_trades_per_day=999,
        slippage_pct=_W['slippage'],
        commission_per_contract=_W['commission'],
    )
    rc_defaults.update(rc_kwargs)  # overrides win
    risk_cfg = RiskConfig(**rc_defaults)

    bt = Backtest0DTE(trade_cfg, risk_cfg, _W['capital'])
    
    # Inject pre-built indexes (avoid rebuilding per backtest)
    bt._opt_by_date_time = _W['opt_by_date_time']
    bt._opt_by_ticker_date = _W['opt_by_ticker_date']

    orb_min = params.get('orb_minutes', 30)
    features_df = _W['features'].get(orb_min)
    if features_df is None:
        features_df = compute_features(_W['underlying'], orb_minutes=orb_min)

    trades = bt.run_no_ml(
        _W['underlying'], _W['options'], features_df,
        _W['vol'], verbose=False
    )

    m = _metrics(trades, _W['capital'])
    m['label'] = params.get('_label', '')

    # Copy param keys into output
    m['strategy'] = params.get('strategy', trade_cfg.strategy)
    m['profit_target'] = params.get('profit_target_pct', trade_cfg.profit_target_pct)
    m['stop_loss'] = params.get('stop_loss_pct', trade_cfg.stop_loss_pct)
    m['max_hold_bars'] = params.get('max_hold_bars', trade_cfg.max_hold_bars)
    m['min_price'] = params.get('min_option_price', trade_cfg.min_option_price)
    m['max_price'] = params.get('max_option_price', trade_cfg.max_option_price)
    m['start_hour'] = params.get('trade_start_hour', trade_cfg.trade_start_hour)
    m['start_min'] = params.get('trade_start_minute', trade_cfg.trade_start_minute)
    m['end_hour'] = params.get('trade_end_hour', trade_cfg.trade_end_hour)
    m['end_min'] = params.get('trade_end_minute', trade_cfg.trade_end_minute)
    m['orb_buffer'] = params.get('orb_buffer_pct', trade_cfg.orb_buffer_pct)
    m['orb_minutes'] = params.get('orb_minutes', trade_cfg.orb_minutes)
    m['vwap_dev'] = params.get('vwap_dev_threshold', trade_cfg.vwap_dev_threshold)
    m['bb_buffer'] = params.get('bb_buffer_pct', trade_cfg.bb_buffer_pct)
    m['rsi_call'] = params.get('rsi_call_threshold', trade_cfg.rsi_call_threshold)
    m['rsi_put'] = params.get('rsi_put_threshold', trade_cfg.rsi_put_threshold)
    # Risk params
    m['daily_loss_limit'] = rc_defaults.get('max_daily_loss_pct', 0.99)
    m['consec_loss_max'] = rc_defaults.get('max_consecutive_losses', 999)
    m['dd_reduce_at'] = rc_defaults.get('reduce_size_at_dd_pct', 0.99)
    m['max_trades_day'] = rc_defaults.get('max_trades_per_day', 999)
    m['stop_first_loss'] = rc_defaults.get('max_daily_losses', 2)
    return m


def _metrics(trades, initial_capital):
    """Compute metrics from trade list."""
    if not trades or len(trades) < 3:
        return dict(trades=len(trades) if trades else 0, win_rate=0,
                    profit_factor=0, sharpe=-99, sortino=-99, max_dd=100.0,
                    final_capital=initial_capital, total_pnl=0, avg_pnl=0,
                    score=-999, monthly_wr=0, calmar=-99)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr = len(wins) / len(trades)
    gp = sum(t.pnl for t in wins) if wins else 0
    gl = abs(sum(t.pnl for t in losses)) if losses else 0.01
    pf = gp / gl if gl > 0 else 99
    total = sum(t.pnl for t in trades)
    final = trades[-1].capital

    peak = initial_capital
    mdd = 0
    for t in trades:
        if t.capital > peak: peak = t.capital
        dd = (peak - t.capital) / peak
        if dd > mdd: mdd = dd

    rets = [t.pnl / max(t.capital - t.pnl, 1) for t in trades]
    mu = np.mean(rets)
    sigma = np.std(rets) if len(rets) > 1 else 1
    down = [r for r in rets if r < 0]
    ds = np.std(down) if len(down) > 1 else 1
    sharpe = (mu * 252) / (sigma * np.sqrt(252)) if sigma > 0 else 0
    sortino = (mu * 252) / (ds * np.sqrt(252)) if ds > 0 else 0

    df = pd.DataFrame([{'date': t.date, 'pnl': t.pnl} for t in trades])
    df['m'] = pd.to_datetime(df['date']).dt.to_period('M')
    mp = df.groupby('m')['pnl'].sum()
    mwr = sum(1 for p in mp if p > 0) / len(mp) * 100 if len(mp) > 0 else 0
    calmar = (final / initial_capital - 1) / mdd if mdd > 0 else 0

    tc_bonus = min(len(trades) / 50, 1.0)
    score = sharpe * 0.30 + min(pf, 5) * 0.25 + wr * 5 * 0.20 + calmar * 0.15 + tc_bonus * 0.10
    if mdd > 0.30: score *= 0.5
    if mdd > 0.50: score *= 0.3

    return dict(trades=len(trades), win_rate=round(wr*100, 1),
                profit_factor=round(pf, 2), sharpe=round(sharpe, 2),
                sortino=round(sortino, 2), max_dd=round(mdd*100, 1),
                final_capital=round(final, 0), total_pnl=round(total, 0),
                avg_pnl=round(total/len(trades), 2), monthly_wr=round(mwr, 0),
                calmar=round(calmar, 2), score=round(score, 3))


# ============================================================
# OPTIMIZER
# ============================================================

class Optimizer:
    def __init__(self, capital=None, workers=None):
        if capital is None:
            from config import defaults as cfg
            capital = cfg.initial_capital()
        self.capital = capital
        self.workers = workers or max(1, cpu_count() - 2)
        self.tmp = Path('output/.optim_tmp')
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._underlying = None
        self._options = None
        self._features = {}
        self._vol = None

    def load_data(self, start, end):
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        cfg = TradeConfig(orb_minutes=30)
        bt = Backtest0DTE(cfg, RiskConfig(), self.capital)
        self._underlying, self._options, f30 = bt.load_data(start, end)
        self._features[30] = f30
        self._vol = bt.compute_historical_volatility(self._underlying)
        for om in [15, 20, 45]:
            self._features[om] = compute_features(self._underlying, orb_minutes=om)
        print(f"  Underlying: {len(self._underlying):,} bars, "
              f"Options: {len(self._options):,} bars, "
              f"Days: {self._underlying['date'].nunique()}, "
              f"Workers: {self.workers}")
        # Serialize
        self._underlying.to_pickle(str(self.tmp / 'u.pkl'))
        self._options.to_pickle(str(self.tmp / 'o.pkl'))
        pd.to_pickle(self._features, str(self.tmp / 'f.pkl'))
        pd.to_pickle(self._vol, str(self.tmp / 'v.pkl'))
        print("  Data serialized. Ready.\n")

    def _run(self, params_list, name, parallel=True):
        n = len(params_list)
        print(f"  {name}: {n} combos × {self.workers} workers")
        t0 = time.time()

        if parallel and n > 1:
            init = (str(self.tmp/'u.pkl'), str(self.tmp/'o.pkl'),
                    str(self.tmp/'f.pkl'), str(self.tmp/'v.pkl'),
                    self.capital, 0.005, 0.65)
            results = []
            with Pool(self.workers, _init_worker, init) as pool:
                for i, r in enumerate(pool.imap_unordered(_run_one, params_list, chunksize=8)):
                    results.append(r)
                    d = len(results)
                    if d % 25 == 0 or d == n:
                        el = time.time() - t0
                        rate = d / el if el > 0 else 1
                        eta = (n - d) / rate if rate > 0 else 0
                        print(f"    [{d}/{n}] {rate:.1f}/s  ETA {eta:.0f}s   ", end='\r')
        else:
            # Sequential fallback
            global _W
            _W['underlying'] = self._underlying
            _W['options'] = self._options
            _W['features'] = self._features
            _W['vol'] = self._vol
            _W['capital'] = self.capital
            _W['slippage'] = 0.005
            _W['commission'] = 0.65
            results = []
            for i, p in enumerate(params_list):
                results.append(_run_one(p))
                if (i+1) % 10 == 0:
                    el = time.time() - t0
                    print(f"    [{i+1}/{n}] {(i+1)/el:.1f}/s   ", end='\r')

        el = time.time() - t0
        print(f"\n  {name}: {n} in {el:.0f}s ({n/el:.1f}/s)")
        df = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
        return df

    # ---------- Phase 1 ----------
    def phase1(self):
        print("=" * 60)
        print("PHASE 1: COARSE GRID (625 combos)")
        print("=" * 60)
        strats = ['orb', 'momentum', 'mean_reversion', 'bb_breakout', 'vwap_reversion']
        pts = [0.08, 0.12, 0.18, 0.25, 0.40]
        sls = [0.08, 0.12, 0.18, 0.25, 0.40]
        mhbs = [1, 2, 4, 6, 12]
        params = []
        for s, pt, sl, mh in itertools.product(strats, pts, sls, mhbs):
            params.append(dict(
                strategy=s, profit_target_pct=pt, stop_loss_pct=sl,
                max_hold_bars=mh, trade_start_hour=10, trade_start_minute=0,
                trade_end_hour=11, trade_end_minute=0,
                min_option_price=0.50, max_option_price=2.00,
                orb_buffer_pct=0.10, orb_minutes=30,
                use_ml_filter=False, skip_day_filter=True,
                use_adaptive_exits=False, use_trailing_stop=False,
                use_time_decay_exit=False, use_quick_exit=False,
                _label=f"P1_{s}_{pt}_{sl}_{mh}",
            ))
        df = self._run(params, "Phase1")
        df.to_csv('output/optimize_phase1.csv', index=False)
        self._top(df, "Phase 1")
        return df

    # ---------- Phase 2 ----------
    def phase2(self, p1=None, top_n=15):
        print("\n" + "=" * 60)
        print(f"PHASE 2: EXPAND TOP {top_n}")
        print("=" * 60)
        if p1 is None: p1 = pd.read_csv('output/optimize_phase1.csv')
        top = p1.head(top_n)
        oranges = [(0.25,1.0),(0.50,2.0),(0.50,3.0),(1.0,5.0)]
        params = []
        for _, r in top.iterrows():
            for (mp,xp) in oranges:
                params.append(dict(
                    strategy=r['strategy'], profit_target_pct=r['profit_target'],
                    stop_loss_pct=r['stop_loss'], max_hold_bars=int(r['max_hold_bars']),
                    trade_start_hour=int(r.get('start_hour',10)),
                    trade_start_minute=int(r.get('start_min',0)),
                    trade_end_hour=int(r.get('end_hour',11)),
                    trade_end_minute=int(r.get('end_min',0)),
                    min_option_price=mp, max_option_price=xp,
                    orb_buffer_pct=r.get('orb_buffer',0.10),
                    orb_minutes=int(r.get('orb_minutes',30)),
                    use_ml_filter=False, skip_day_filter=True,
                    use_adaptive_exits=False, use_trailing_stop=False,
                    use_time_decay_exit=False, use_quick_exit=False,
                    _label=f"P2_{r['strategy']}_{mp}-{xp}",
                ))
        df = self._run(params, "Phase2")
        df.to_csv('output/optimize_phase2.csv', index=False)
        self._top(df, "Phase 2")
        return df

    # ---------- Phase 3 ----------
    def phase3(self, p2=None, top_n=5):
        print("\n" + "=" * 60)
        print(f"PHASE 3: FINE-TUNE TOP {top_n}")
        print("=" * 60)
        if p2 is None: p2 = pd.read_csv('output/optimize_phase2.csv')
        top = p2.head(top_n)
        params = []
        for _, r in top.iterrows():
            bpt = r['profit_target']; bsl = r['stop_loss']
            bmh = int(r['max_hold_bars']); st = r['strategy']
            pt_r = np.linspace(max(0.03, bpt*0.6), bpt*1.4, 9)
            sl_r = np.linspace(max(0.03, bsl*0.6), bsl*1.4, 9)
            mh_r = sorted(set([max(1,bmh-2),max(1,bmh-1),bmh,bmh+1,bmh+2,bmh+3]))
            if st == 'orb':
                extras = [dict(orb_buffer_pct=x) for x in [0.03,0.05,0.08,0.10,0.15,0.20]]
            elif st == 'vwap_reversion':
                extras = [dict(vwap_dev_threshold=x) for x in [0.10,0.15,0.20,0.25,0.30,0.40]]
            elif st in ('momentum','mean_reversion'):
                extras = [dict(rsi_call_threshold=c, rsi_put_threshold=p)
                          for c,p in [(60,40),(65,35),(70,30),(75,25)]]
            elif st == 'bb_breakout':
                extras = [dict(bb_buffer_pct=x) for x in [0.0,0.005,0.01,0.02]]
            else:
                extras = [{}]
            for pt,sl,mh,ex in itertools.product(pt_r, sl_r, mh_r, extras):
                p = dict(
                    strategy=st, profit_target_pct=round(float(pt),4),
                    stop_loss_pct=round(float(sl),4), max_hold_bars=int(mh),
                    trade_start_hour=int(r['start_hour']),
                    trade_start_minute=int(r['start_min']),
                    trade_end_hour=int(r['end_hour']),
                    trade_end_minute=int(r['end_min']),
                    min_option_price=float(r['min_price']),
                    max_option_price=float(r['max_price']),
                    orb_buffer_pct=float(r.get('orb_buffer',0.10)),
                    orb_minutes=int(r.get('orb_minutes',30)),
                    vwap_dev_threshold=float(r.get('vwap_dev',0.30)),
                    rsi_call_threshold=float(r.get('rsi_call',70)),
                    rsi_put_threshold=float(r.get('rsi_put',30)),
                    bb_buffer_pct=float(r.get('bb_buffer',0.0)),
                    use_ml_filter=False, skip_day_filter=True,
                    use_adaptive_exits=False, use_trailing_stop=False,
                    use_time_decay_exit=False, use_quick_exit=False,
                    _label=f"P3_{st}_pt{pt:.3f}_sl{sl:.3f}_h{mh}",
                )
                p.update(ex)
                params.append(p)
        df = self._run(params, "Phase3")
        df.to_csv('output/optimize_phase3.csv', index=False)
        self._top(df, "Phase 3")
        return df

    # ---------- Phase 4 ----------
    def phase4(self, p3=None, top_n=5):
        print("\n" + "=" * 60)
        print(f"PHASE 4: ADVANCED EXITS (top {top_n})")
        print("=" * 60)
        if p3 is None: p3 = pd.read_csv('output/optimize_phase3.csv')
        top = p3.head(top_n)
        exits = [
            dict(_el="baseline"),
            dict(_el="trail_10_50", use_trailing_stop=True, trail_activation_pct=0.10,
                 trail_distance_pct=0.50, breakeven_activation=0.08),
            dict(_el="trail_05_40", use_trailing_stop=True, trail_activation_pct=0.05,
                 trail_distance_pct=0.40, breakeven_activation=0.05),
            dict(_el="trail_08_60", use_trailing_stop=True, trail_activation_pct=0.08,
                 trail_distance_pct=0.60, breakeven_activation=0.06),
            dict(_el="tdecay_3", use_time_decay_exit=True,
                 time_decay_profit_per_bar=0.03, min_profit_target=0.05),
            dict(_el="tdecay_5", use_time_decay_exit=True,
                 time_decay_profit_per_bar=0.05, min_profit_target=0.03),
            dict(_el="qexit_15", use_quick_exit=True, underwater_stop_tighten=0.15,
                 quick_exit_profit_threshold=0.05, breakeven_buffer_pct=0.02),
            dict(_el="qexit_10", use_quick_exit=True, underwater_stop_tighten=0.10,
                 quick_exit_profit_threshold=0.03, breakeven_buffer_pct=0.01),
            dict(_el="trail+tdecay", use_trailing_stop=True, trail_activation_pct=0.08,
                 trail_distance_pct=0.50, breakeven_activation=0.06,
                 use_time_decay_exit=True, time_decay_profit_per_bar=0.03, min_profit_target=0.05),
            dict(_el="trail+qexit", use_trailing_stop=True, trail_activation_pct=0.08,
                 trail_distance_pct=0.50, use_quick_exit=True, underwater_stop_tighten=0.15),
            dict(_el="all3", use_trailing_stop=True, trail_activation_pct=0.08,
                 trail_distance_pct=0.50, use_time_decay_exit=True,
                 time_decay_profit_per_bar=0.03, min_profit_target=0.05,
                 use_quick_exit=True, underwater_stop_tighten=0.15),
        ]
        params = []
        for _, r in top.iterrows():
            for ex in exits:
                el = ex['_el']
                p = dict(
                    strategy=r['strategy'], profit_target_pct=float(r['profit_target']),
                    stop_loss_pct=float(r['stop_loss']), max_hold_bars=int(r['max_hold_bars']),
                    trade_start_hour=int(r['start_hour']), trade_start_minute=int(r['start_min']),
                    trade_end_hour=int(r['end_hour']), trade_end_minute=int(r['end_min']),
                    min_option_price=float(r['min_price']), max_option_price=float(r['max_price']),
                    orb_buffer_pct=float(r.get('orb_buffer',0.10)),
                    orb_minutes=int(r.get('orb_minutes',30)),
                    vwap_dev_threshold=float(r.get('vwap_dev',0.30)),
                    rsi_call_threshold=float(r.get('rsi_call',70)),
                    rsi_put_threshold=float(r.get('rsi_put',30)),
                    bb_buffer_pct=float(r.get('bb_buffer',0.0)),
                    use_ml_filter=False, skip_day_filter=True, use_adaptive_exits=False,
                    _label=f"P4_{r['strategy']}_{el}",
                )
                for k,v in ex.items():
                    if k != '_el': p[k] = v
                params.append(p)
        df = self._run(params, "Phase4")
        df.to_csv('output/optimize_phase4.csv', index=False)
        self._top(df, "Phase 4")
        return df

    def _top(self, df, name, n=20):
        print(f"\n  {name} — TOP {n}:")
        print("-" * 130)
        cols = ['strategy','profit_target','stop_loss','max_hold_bars',
                'trades','win_rate','profit_factor','sharpe','max_dd',
                'final_capital','total_pnl','score',
                'daily_loss_limit','consec_loss_max','dd_reduce_at','max_trades_day']
        avail = [c for c in cols if c in df.columns]
        print(df[avail].head(n).to_string(index=False))

    def apply_best(self, df):
        b = df.iloc[0]
        print("\n" + "=" * 60)
        print("APPLYING BEST CONFIG")
        print("=" * 60)
        with open('config/strategy.json','r') as f: cfg = json.load(f)
        tc = cfg['trade_config']
        tc['strategy'] = b['strategy']
        tc['profit_target_pct'] = float(b['profit_target'])
        tc['stop_loss_pct'] = float(b['stop_loss'])
        tc['max_hold_bars'] = int(b['max_hold_bars'])
        tc['min_option_price'] = float(b['min_price'])
        tc['max_option_price'] = float(b['max_price'])
        tc['trade_start_hour'] = int(b['start_hour'])
        tc['trade_start_minute'] = int(b['start_min'])
        tc['trade_end_hour'] = int(b['end_hour'])
        tc['trade_end_minute'] = int(b['end_min'])
        tc['orb_buffer_pct'] = float(b.get('orb_buffer',0.10))
        tc['vwap_dev_threshold'] = float(b.get('vwap_dev',0.30))
        tc['rsi_call_threshold'] = float(b.get('rsi_call',70))
        tc['rsi_put_threshold'] = float(b.get('rsi_put',30))
        tc['orb_minutes'] = int(b.get('orb_minutes',30))
        tc['max_hold_minutes'] = int(b['max_hold_bars']) * 5
        # Risk config
        rc = cfg.setdefault('risk_config', {})
        if 'daily_loss_limit' in b: rc['max_daily_loss_pct'] = float(b['daily_loss_limit'])
        if 'consec_loss_max' in b: rc['max_consecutive_losses'] = int(b['consec_loss_max'])
        if 'dd_reduce_at' in b: rc['reduce_size_at_dd_pct'] = float(b['dd_reduce_at'])
        if 'max_trades_day' in b: rc['max_trades_per_day'] = int(b['max_trades_day'])
        if 'stop_first_loss' in b: rc['max_daily_losses'] = int(b['stop_first_loss'])
        cfg['strategy'] = b['strategy']
        cfg['optimized_results'] = dict(
            optimization_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            trades=int(b['trades']), win_rate=float(b['win_rate']),
            profit_factor=float(b['profit_factor']), sharpe=float(b['sharpe']),
            max_dd=float(b['max_dd']), final_capital=float(b['final_capital']),
            score=float(b['score']),
        )
        with open('config/strategy.json','w') as f: json.dump(cfg, f, indent=2)
        print(f"  {b['strategy']} | PT={b['profit_target']:.1%} SL={b['stop_loss']:.1%} "
              f"Hold={int(b['max_hold_bars'])} | WR={b['win_rate']:.1f}% PF={b['profit_factor']:.2f} "
              f"Sharpe={b['sharpe']:.2f} DD={b['max_dd']:.1f}% | ${b['final_capital']:,.0f}")
        print("  Saved to config/strategy.json")

        # Save to centralized optimization history
        save_optimization_run(
            source='optimize',
            trade_config={k: tc[k] for k in tc},
            risk_config={k: v for k, v in cfg.get('risk_config', {}).items()},
            results=cfg.get('optimized_results', {}),
        )

    # ---------- Phase 5: Risk Management ----------
    def phase5(self, prev=None, top_n=8):
        """
        Phase 5: Risk Management Optimization
        Sweeps daily loss limits, consecutive-loss circuit breakers,
        drawdown-based position reduction, and max trades per day
        on the best strategies from prior phases.
        """
        print("\n" + "=" * 60)
        print(f"PHASE 5: RISK MANAGEMENT (top {top_n} strategies)")
        print("=" * 60)
        if prev is None:
            # Try loading from most recent phase
            for pf in ['optimize_phase4.csv', 'optimize_phase3.csv', 'optimize_phase2.csv']:
                p = Path('output') / pf
                if p.exists():
                    prev = pd.read_csv(str(p))
                    print(f"  Loaded {pf}")
                    break
        if prev is None:
            print("  ERROR: No prior phase results found")
            return None
        top = prev.head(top_n)

        # Risk parameter grid
        daily_loss_limits = [0.015, 0.02, 0.03, 0.05, 0.99]   # 1.5%, 2%, 3%, 5%, disabled
        consec_loss_maxes = [3, 5, 7, 999]                      # 3, 5, 7, disabled
        dd_reduce_ats = [0.05, 0.08, 0.12, 0.99]               # 5%, 8%, 12%, disabled
        max_trades_days = [3, 5, 8, 999]                        # 3, 5, 8, unlimited

        params = []
        for _, r in top.iterrows():
            for dll, clm, dra, mtd in itertools.product(
                daily_loss_limits, consec_loss_maxes, dd_reduce_ats, max_trades_days
            ):
                # Skip the fully-disabled combo (already tested)
                if dll == 0.99 and clm == 999 and dra == 0.99 and mtd == 999:
                    continue
                p = dict(
                    strategy=r['strategy'],
                    profit_target_pct=float(r['profit_target']),
                    stop_loss_pct=float(r['stop_loss']),
                    max_hold_bars=int(r['max_hold_bars']),
                    trade_start_hour=int(r['start_hour']),
                    trade_start_minute=int(r['start_min']),
                    trade_end_hour=int(r['end_hour']),
                    trade_end_minute=int(r['end_min']),
                    min_option_price=float(r['min_price']),
                    max_option_price=float(r['max_price']),
                    orb_buffer_pct=float(r.get('orb_buffer', 0.10)),
                    orb_minutes=int(r.get('orb_minutes', 30)),
                    vwap_dev_threshold=float(r.get('vwap_dev', 0.30)),
                    rsi_call_threshold=float(r.get('rsi_call', 70)),
                    rsi_put_threshold=float(r.get('rsi_put', 30)),
                    bb_buffer_pct=float(r.get('bb_buffer', 0.0)),
                    use_ml_filter=False, skip_day_filter=True,
                    use_adaptive_exits=False,
                    # Advanced exits from prior phase
                    use_trailing_stop=bool(r.get('label', '').find('trail') >= 0) if 'label' in r else False,
                    use_time_decay_exit=bool(r.get('label', '').find('tdecay') >= 0) if 'label' in r else False,
                    use_quick_exit=bool(r.get('label', '').find('qexit') >= 0) if 'label' in r else False,
                    # Risk management params (these go to RiskConfig)
                    max_daily_loss_pct=dll,
                    max_consecutive_losses=clm,
                    reduce_size_at_dd_pct=dra,
                    max_trades_per_day=mtd,
                    _label=f"P5_{r['strategy']}_dll{dll}_cl{clm}_dd{dra}_mt{mtd}",
                )
                params.append(p)

        print(f"  {top_n} strategies × {len(params)//top_n} risk combos = {len(params)} total")
        df = self._run(params, "Phase5")
        df.to_csv('output/optimize_phase5.csv', index=False)
        self._top(df, "Phase 5")

        # Print DD analysis
        print(f"\n  DD Distribution:")
        for dd_max in [5, 10, 15, 20, 30, 50]:
            subset = df[df['max_dd'] <= dd_max]
            if len(subset) > 0:
                best = subset.iloc[0]
                print(f"    DD≤{dd_max:2d}%: {len(subset):4d} combos | "
                      f"Best: {best['strategy']} ${best['total_pnl']:+,.0f} "
                      f"({best['total_pnl']/self.capital*100:+.0f}%) "
                      f"WR={best['win_rate']:.1f}% DD={best['max_dd']:.1f}%")
            else:
                print(f"    DD≤{dd_max:2d}%: None")
        return df

    def cleanup(self):
        import shutil
        if self.tmp.exists(): shutil.rmtree(self.tmp)


def main():
    ap = argparse.ArgumentParser(description='0DTE Optimizer (Parallel)')
    ap.add_argument('--phase', type=int, default=0, help='Phase 1-5, 0=all(1-4), 5=risk mgmt')
    ap.add_argument('--top', type=int, default=15, help='Top N to carry forward')
    ap.add_argument('--apply', action='store_true', help='Apply best to strategy.json')
    ap.add_argument('--start', default='2026-01-02')
    ap.add_argument('--end', default='2026-02-10')
    ap.add_argument('--capital', type=float, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--sequential', action='store_true', help='Single-threaded debug')
    ap.add_argument('--p5input', default=None, help='CSV file for Phase 5 input pool')
    a = ap.parse_args()

    o = Optimizer(a.capital, a.workers)
    o.load_data(a.start, a.end)
    t0 = time.time()

    p1 = o.phase1() if a.phase in (0,1) else None
    p2 = o.phase2(p1, a.top) if a.phase in (0,2) else None
    p3 = o.phase3(p2, min(a.top,5)) if a.phase in (0,3) else None
    p4 = o.phase4(p3, 5) if a.phase in (0,4) else None

    if a.phase == 5:
        # Phase 5: Risk Management
        if a.p5input:
            p5_input = pd.read_csv(a.p5input)
        else:
            # Build diverse pool from Phase 1
            p1_df = pd.read_csv('output/optimize_phase1.csv')
            bb = p1_df[p1_df['strategy']=='bb_breakout'].sort_values('score',ascending=False).head(4)
            mom = p1_df[p1_df['strategy']=='momentum'].sort_values('total_pnl',ascending=False).head(4)
            orb = p1_df[(p1_df['strategy']=='orb')&(p1_df['total_pnl']>0)].sort_values('total_pnl',ascending=False).head(2)
            p5_input = pd.concat([bb, mom, orb], ignore_index=True)
            print(f"\n  Diverse Phase 5 Pool ({len(p5_input)} strategies):")
            for _, r in p5_input.iterrows():
                print(f"    {r['strategy']:16s} PT={r['profit_target']:.0%} SL={r['stop_loss']:.0%} "
                      f"H={int(r['max_hold_bars']):2d} | {int(r['trades']):4d}t "
                      f"WR={r['win_rate']:.1f}% DD={r['max_dd']:.1f}% PnL=${r['total_pnl']:+,.0f}")
        p5 = o.phase5(p5_input, top_n=len(p5_input))
    else:
        p5 = None

    print(f"\n{'='*60}\nTOTAL: {time.time()-t0:.0f}s\n{'='*60}")

    final = next((x for x in [p5, p4, p3, p2, p1] if x is not None), None)
    if a.apply and final is not None:
        o.apply_best(final)
    elif final is not None:
        b = final.iloc[0]
        print(f"\n  BEST: {b['strategy']} PT={b['profit_target']:.0%} SL={b['stop_loss']:.0%} "
              f"H={int(b['max_hold_bars'])} | WR={b['win_rate']:.1f}% PF={b['profit_factor']:.2f} "
              f"Sharpe={b['sharpe']:.2f} | ${b['final_capital']:,.0f}")
        if 'daily_loss_limit' in final.columns and pd.notna(b.get('daily_loss_limit')):
            print(f"  RISK: DailyLoss={b['daily_loss_limit']:.1%} "
                  f"ConsecMax={int(b['consec_loss_max'])} "
                  f"DDReduce@={b['dd_reduce_at']:.0%} "
                  f"MaxTrades/Day={int(b['max_trades_day'])}")
        # Show momentum comparison for Phase 5
        if p5 is not None:
            mom = final[final['strategy'] == 'momentum']
            if len(mom) > 0:
                print(f"\n  TOP MOMENTUM (risk-controlled):")
                for i, (_, r) in enumerate(mom.head(5).iterrows()):
                    print(f"    #{i+1} PnL=${r['total_pnl']:+,.0f} DD={r['max_dd']:.1f}% "
                          f"WR={r['win_rate']:.1f}% Trades={int(r['trades'])} "
                          f"[DLL={r['daily_loss_limit']:.1%} CL={int(r['consec_loss_max'])} "
                          f"DDR={r['dd_reduce_at']:.0%} MT={int(r['max_trades_day'])}]")
                # Best under DD thresholds
                for dd_cap in [15, 20, 25]:
                    sub = mom[mom['max_dd'] <= dd_cap]
                    if len(sub) > 0:
                        s = sub.iloc[0]
                        print(f"    DD≤{dd_cap}%: PnL=${s['total_pnl']:+,.0f} DD={s['max_dd']:.1f}% "
                              f"[DLL={s['daily_loss_limit']:.1%} CL={int(s['consec_loss_max'])} MT={int(s['max_trades_day'])}]")
        print("\n  Run with --apply to save")
    o.cleanup()


if __name__ == '__main__':
    main()
