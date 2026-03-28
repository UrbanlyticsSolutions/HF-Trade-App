"""
Configuration Module

Single source of truth: config/strategy.json → config/defaults.py
All parameters flow from strategy.json. Never hardcode params elsewhere.
"""
from .defaults import (
    get_config, get_trade_config, get_risk_config,
    initial_capital, max_contracts, ibkr_live_port, ibkr_paper_port, dashboard_port,
    profit_target_pct, stop_loss_pct, min_option_price, max_option_price,
    trade_start_hour, trade_start_minute, trade_end_hour, trade_end_minute,
    exit_hour, max_hold_minutes, orb_minutes, orb_buffer_pct,
    rsi_call_threshold, rsi_put_threshold,
    kelly_fraction, max_risk_per_trade_pct, max_position_pct, max_position_value,
    max_daily_losses, max_daily_loss_pct, max_consecutive_losses,
    consec_loss_reduction, wins_to_reset_streak,
)

__all__ = [
    'get_config', 'get_trade_config', 'get_risk_config',
    'initial_capital', 'max_contracts', 'ibkr_live_port', 'ibkr_paper_port', 'dashboard_port',
    'profit_target_pct', 'stop_loss_pct', 'min_option_price', 'max_option_price',
    'trade_start_hour', 'trade_start_minute', 'trade_end_hour', 'trade_end_minute',
    'exit_hour', 'max_hold_minutes', 'orb_minutes', 'orb_buffer_pct',
    'rsi_call_threshold', 'rsi_put_threshold',
    'kelly_fraction', 'max_risk_per_trade_pct', 'max_position_pct', 'max_position_value',
    'max_daily_losses', 'max_daily_loss_pct', 'max_consecutive_losses',
    'consec_loss_reduction', 'wins_to_reset_streak',
]
