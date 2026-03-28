"""
Centralized defaults loaded from strategy.json.

Every default value in the system comes from here.
Other modules import these instead of hardcoding values.
"""
import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "strategy.json"
_config = {}


def _load():
    global _config
    if not _config:
        with open(_CONFIG_PATH) as f:
            _config = json.load(f)
    return _config


def get_config() -> dict:
    """Return the full strategy.json config dict."""
    return _load()


def get_trade_config() -> dict:
    return _load().get("trade_config", {})


def get_risk_config() -> dict:
    return _load().get("risk_config", {})


# ── Top-level defaults ──────────────────────────────────────
def initial_capital() -> float:
    """Fallback capital only used when broker query fails. Not the source of truth."""
    live_cfg = _load().get("live", {})
    return live_cfg.get("fallback_capital", _load().get("initial_capital", 100000))

def max_contracts() -> int:
    return _load().get("max_contracts", 50)

def ibkr_live_port() -> int:
    return _load().get("ibkr_live_port", 7496)

def ibkr_paper_port() -> int:
    return _load().get("ibkr_paper_port", 7497)

def dashboard_port() -> int:
    return _load().get("dashboard_port", 8050)


# ── Trade config helpers ────────────────────────────────────
def _tc(key, fallback):
    return get_trade_config().get(key, fallback)

def profit_target_pct() -> float:   return _tc("profit_target_pct", 0.50)
def stop_loss_pct() -> float:       return _tc("stop_loss_pct", 0.35)
def min_option_price() -> float:    return _tc("min_option_price", 0.50)
def max_option_price() -> float:    return _tc("max_option_price", 2.00)
def trade_start_hour() -> int:      return _tc("trade_start_hour", 9)
def trade_start_minute() -> int:    return _tc("trade_start_minute", 35)
def trade_end_hour() -> int:        return _tc("trade_end_hour", 15)
def trade_end_minute() -> int:      return _tc("trade_end_minute", 0)
def exit_hour() -> int:             return _tc("exit_hour", 15)
def max_hold_minutes() -> int:      return _tc("max_hold_minutes", 80)
def orb_minutes() -> int:           return _tc("orb_minutes", 30)
def orb_buffer_pct() -> float:      return _tc("orb_buffer_pct", 0.10)
def rsi_call_threshold() -> int:    return _tc("rsi_call_threshold", 70)
def rsi_put_threshold() -> int:     return _tc("rsi_put_threshold", 30)
def put_min_rsi() -> float:         return _tc("put_min_rsi", 25.0)
def put_skip_days() -> list:        return _tc("put_skip_days", None)
def put_min_entry_minutes() -> int: return _tc("put_min_entry_minutes", 0)
def put_filter_require_uptrend() -> bool: return _tc("put_filter_require_uptrend", True)
def put_adaptive_filter() -> bool:        return _tc("put_adaptive_filter", True)
def put_loss_streak_threshold() -> int:   return _tc("put_loss_streak_threshold", 2)
def put_adaptive_cooldown() -> int:       return _tc("put_adaptive_cooldown", 3)
def call_adaptive_filter() -> bool:       return _tc("call_adaptive_filter", False)
def call_loss_streak_threshold() -> int:  return _tc("call_loss_streak_threshold", 2)
def call_adaptive_cooldown() -> int:      return _tc("call_adaptive_cooldown", 3)

# ── Direction-aware loss escalation ─────────────────────────
def use_direction_loss_escalation() -> bool: return _tc("use_direction_loss_escalation", False)
def direction_loss_window() -> int:         return _tc("direction_loss_window", 3)
def direction_loss_threshold() -> int:      return _tc("direction_loss_threshold", 2)
def direction_loss_cooldown() -> int:       return _tc("direction_loss_cooldown", 3)
def consec_loss_rsi_buffer() -> int:        return _tc("consec_loss_rsi_buffer", 0)

# ── Limit order pricing ─────────────────────────────────────
def limit_offset_cents() -> float:  return _tc("limit_offset_cents", 1.0)

# ── Asymmetric CALL/PUT exits ───────────────────────────────
def call_profit_target_pct() -> float:  return _tc("call_profit_target_pct", None)
def put_profit_target_pct() -> float:   return _tc("put_profit_target_pct", None)
def call_stop_loss_pct() -> float:      return _tc("call_stop_loss_pct", None)
def put_stop_loss_pct() -> float:       return _tc("put_stop_loss_pct", None)
def call_max_hold_bars() -> int:        return _tc("call_max_hold_bars", None)
def put_max_hold_bars() -> int:         return _tc("put_max_hold_bars", None)

# ── Regime detection ────────────────────────────────────────
def use_regime_detection() -> bool:     return _tc("use_regime_detection", False)
def regime_lookback_days() -> int:      return _tc("regime_lookback_days", 5)
def regime_vol_percentile() -> float:   return _tc("regime_vol_percentile", 0.30)
def regime_trend_percentile() -> float: return _tc("regime_trend_percentile", 0.25)
def regime_size_reduction() -> float:   return _tc("regime_size_reduction", 0.50)
def regime_skip_first_bar() -> bool:    return _tc("regime_skip_first_bar", True)
def regime_rsi_buffer() -> int:         return _tc("regime_rsi_buffer", 5)
def regime_tighter_stop_pct():          return _tc("regime_tighter_stop_pct", None)


# ── Risk config helpers ─────────────────────────────────────
def _rc(key, fallback):
    return get_risk_config().get(key, fallback)

def kelly_fraction() -> float:          return _rc("kelly_fraction", 0.20)
def kelly_pct() -> float:              return _rc("kelly_pct", 0.03)
def max_risk_per_trade_pct() -> float:  return _rc("max_risk_per_trade_pct", 0.02)
def max_position_pct() -> float:        return _rc("max_position_pct", 0.07)
def max_position_value() -> float:      return _rc("max_position_value", 5000)
def max_daily_losses() -> int:       return _rc("max_daily_losses", 2)
def max_daily_loss_pct() -> float:      return _rc("max_daily_loss_pct", 0.008)
def max_consecutive_losses() -> int:    return _rc("max_consecutive_losses", 3)
def consec_loss_reduction() -> float:   return _rc("consec_loss_reduction", 0.50)
def wins_to_reset_streak() -> int:      return _rc("wins_to_reset_streak", 2)
