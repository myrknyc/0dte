PAPER_TRADING = {
    # ══════════════════════════════════════════════════════════
    #  GLOBAL
    # ══════════════════════════════════════════════════════════
    'tracks': ['decision_time', 'all_signals'],
    'timezone': 'America/New_York',

    # ── 3-tier fill model (shared by both tracks) ────────────
    'slippage_mode': 'pct_of_spread',   # fixed | pct_of_spread
    'slippage_fixed': 0.01,             # $ per contract (if mode=fixed)
    'slippage_pct': 0.10,               # fraction of spread (if mode=pct_of_spread)
    'commission_per_contract': 0.65,
    'exchange_fees_per_contract': 0.05,

    'max_quote_age_seconds_for_exits': 120,      # skip TP/SL if quote > 2 min old
    'max_quote_gap_minutes_for_forward_eval': 2,  # gap > 2 min → mark stale

    'eval_horizons_minutes': [1, 3, 5, 10, 15, 30, 60],

    'strategy_version': '1.0.0',

    'decision_times': [
        '09:30', '10:00', '10:30', '11:00',
        '13:00', '14:00', '15:00', '15:30',
    ],
    'max_trades_per_decision': 2,
    'max_open_positions': 5,

    'cooldown_minutes': 15,
    'cooldown_scope': 'same_direction',   # same_direction | any_direction
    'min_strike_spacing': 1.0,            # $ apart minimum

    'filters': {
        'min_edge': 0.02,
        'min_confidence': 0.50,
        'max_spread_pct': 0.10,
        'min_option_mid': 0.05,
        'max_option_mid': None,           # no cap
        'max_spot_age_seconds': 10,
        'skip_bernoulli_violated': True,
        'max_otm_dollars': 5.0,
        'min_tte_minutes': 15,
    },

    'selection_policy': 'risk_adjusted',
    'spread_floor': 0.01,                 # guard against ÷ tiny spread

    'sizing_mode': 'fixed_contracts',     # fixed_contracts | fixed_dollar_risk | confidence_scaled
    'fixed_contracts': 1,
    'fixed_dollar_risk': 100.0,

    'exit_mode': 'hybrid',                # time | tp_sl | eod | hybrid
    'exit_time_minutes': 30,
    'tp_pct': 0.20,                       # +20%
    'sl_pct': 0.15,                       # -15%
    'eod_exit_time': '15:55',

    
    'all_signals': {
        'enabled': True,
        'enter_on': 'every_scan',         # every_scan | decision_times
        'apply_filters': {
            'use_spot_age': True,         # data integrity (always on)
            'use_spread_pct': False,      # off → take everything
            'use_min_option_mid': False,
        },
        'dedup_policy': 'one_open_per_strike_action',
        'cooldown_minutes': 0,            # if dedup_policy='cooldown'
        'exit_mode': 'fixed_horizon',     # fixed_horizon | eod
        'exit_horizon_minutes': 5,
        'eod_exit_time': '15:55',
    },
}
