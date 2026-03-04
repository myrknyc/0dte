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
        'max_cvar_loss': -2.00,           # reject if avg worst-5% loss > $2 per contract
        'max_cvar_pct': -3.0,             # reject if CVaR / premium < -3× (normalized)
        'min_eu': 0.0,                    # H1: only enter when EU > 0
    },

    'selection_policy': 'eu_ranked',    # H8: EU / sqrt(CVaR) ranking
    'spread_floor': 0.01,                 # guard against ÷ tiny spread

    # Phase 2 flags (paper-scoped — overrides global OFF defaults)
    'use_eu_scoring': True,               # H1: EU-based entry gate
    'use_regime_thresholds': True,        # H4: regime-adaptive filter thresholds

    'sizing_mode': 'fixed_contracts',     # fixed_contracts | fixed_dollar_risk | confidence_scaled
    'fixed_contracts': 1,
    'fixed_dollar_risk': 100.0,

    'exit_mode': 'hybrid',                # time | tp_sl | eod | hybrid
    'exit_time_minutes': 30,
    'tp_pct': 0.20,                       # +20%
    'sl_pct': 0.15,                       # -15%
    'eod_exit_time': '15:55',

    'greeks_exit': {
        'enabled': True,
        'theta_decay_pct': 0.80,      # exit if projected decay > 80% of time value
        'lookforward_minutes': 15,     # projection window
        'min_profit_pct': 0.05,        # only apply when trade is >5% profitable
    },

    
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

    # ── Track C: buy-only (calls + puts, no option selling) ──
    'buy_only': {
        'enabled': True,
        'enter_on': 'decision_time',          # same cadence as Track A
        'action_filter': 'BUY',               # reject all SELL signals
        'option_types': ['call', 'put'],       # both directions
        'max_open_positions': 3,
        'max_trades_per_decision': 1,
        'use_eu_scoring': True,
        'use_regime_thresholds': True,
        'selection_policy': 'eu_ranked',
        'filters': {
            'min_confidence': 0.50,
            'min_edge': 0.02,
            'max_spread_pct': 0.40,
            'min_option_mid': 0.05,
            'max_spot_age_seconds': 10,
            'skip_bernoulli_violated': True,
            'max_otm_dollars': 5.0,
            'min_eu': 0.0,
        },
    },
}
