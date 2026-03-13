from datetime import datetime, date, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import json
import logging

from core.clock import now_et, now_utc, now_utc_str, to_utc_str, ET as _ET
from trading.paper_config import PAPER_TRADING
from trading.fill_model import simulate_fill, compute_fees, compute_pnl
from trading.fill_model import _fees_per_contract, _compute_slippage
from trading.paper_journal import PaperJournal
from calibration.probability_calibrator import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class PaperTrader:

    def __init__(self, journal: PaperJournal, config: dict = None):
        self.cfg = config or PAPER_TRADING
        self.journal = journal
        self.run_id = journal.start_run(
            config=self.cfg,
            strategy_version=self.cfg.get('strategy_version', ''),
        )

        # Decision-time dedup: track which decision times have fired today
        self._decision_fired_today: Dict[str, date] = {}

        # In-memory cache of open trades (synced with DB)
        self._open_trades_a: Dict[str, dict] = {}  # trade_id → trade row
        self._open_trades_b: Dict[str, dict] = {}
        self._open_trades_c: Dict[str, dict] = {}  # Track C: buy-only
        self._trade_close_times: Dict[str, datetime] = {}  # last close per (strike, action, track)
        self._strike_entry_counts: Dict[str, Dict[float, int]] = {}  # track → {strike → count}

        # H2 scaffold: probability calibrator (passthrough by default)
        self._calibrator = ProbabilityCalibrator()

        # Current regime (set per scan by on_scan or _make_decision_a)
        self._current_regime = 'unknown'
        self._intraday_move_pct = None

        # Crash recovery: reload any OPEN trades from DB
        self._load_open_trades()

    # ================================================================== #
    #  Main entry point                                                    #
    # ================================================================== #

    def on_scan(self, signals: List[dict], quote_map: Dict[float, dict],
                spot: float, timestamp_et: datetime,
                intraday_move_pct: float = None,
                trading_system=None):
        
        ts_utc = to_utc_str(timestamp_et)

        # Capture regime from trading system or explicit param
        if trading_system and hasattr(trading_system, '_current_regime'):
            self._current_regime = getattr(trading_system, '_current_regime', 'unknown')
            self._intraday_move_pct = getattr(trading_system, '_intraday_move_pct', None)
        elif intraday_move_pct is not None:
            from calibration.regime_detector import classify
            self._current_regime = classify(intraday_move_pct)
            self._intraday_move_pct = intraday_move_pct

        # 1. Record quotes to tape (always, for forward eval)
        self._record_quotes(quote_map, spot, ts_utc)

        # Pre-compute decision-time flag ONCE (fire-once gate shared by all tracks)
        is_dt = self._is_decision_time(timestamp_et)

        # 2. Track A: decision_time
        if is_dt:
            self._make_decision_a(signals, quote_map, spot, timestamp_et, ts_utc)

        # 3. Track B: all_signals
        b_cfg = self.cfg.get('all_signals', {})
        if b_cfg.get('enabled', False):
            enter_on = b_cfg.get('enter_on', 'every_scan')
            if enter_on == 'every_scan' or is_dt:
                self._enter_all_signals_b(signals, quote_map, spot, timestamp_et, ts_utc)

        # 4. Track C: buy_only
        c_cfg = self.cfg.get('buy_only', {})
        if c_cfg.get('enabled', False):
            enter_on = c_cfg.get('enter_on', 'decision_time')
            if enter_on == 'every_scan' or is_dt:
                self._make_decision_c(signals, quote_map, spot, timestamp_et, ts_utc)

        # 5. Check exits for all tracks
        self._check_exits(quote_map, timestamp_et, ts_utc)

    # ================================================================== #
    #  Crash recovery                                                       #
    # ================================================================== #

    def _load_open_trades(self):
        """Hydrate in-memory trade caches from DB (crash recovery)."""
        open_trades = self.journal.get_open_trades(self.run_id)
        # Note: on a fresh run_id there are no open trades.
        # This is primarily for future restart-same-run support.
        # For now, also check the *most recent* run for orphaned opens.
        if not open_trades:
            # Look for orphaned OPEN trades from any prior run
            rows = self.journal.conn.execute(
                "SELECT * FROM paper_trades WHERE status = 'OPEN' "
                "ORDER BY entry_timestamp_utc DESC"
            ).fetchall()
            if rows:
                logger.warning(
                    "Found %d orphaned OPEN trades from prior runs. "
                    "Consider running reconciliation.", len(rows)
                )
            return

        loaded = {'decision_time': 0, 'all_signals': 0, 'buy_only': 0}
        for t in open_trades:
            trade = {
                'trade_id': t['trade_id'],
                'strike': t['strike'],
                'action': t['action'],
                'option_type': t.get('option_type', 'call'),
                'entry_timestamp_utc': t['entry_timestamp_utc'],
                'entry_fill': {
                    'mid': t.get('entry_fill_mid', 0),
                    'touch': t.get('entry_fill_touch', 0),
                    'slippage': t.get('entry_fill_slippage', 0),
                    'spread': t.get('entry_spread', 0),
                },
                'quantity': t.get('quantity', 1),
                'entry_fees': t.get('entry_fees', 0),
                'track': t['track'],
            }
            track = t['track']
            tid = t['trade_id']
            if track == 'decision_time':
                self._open_trades_a[tid] = trade
            elif track == 'buy_only':
                self._open_trades_c[tid] = trade
            else:
                self._open_trades_b[tid] = trade
            loaded[track] = loaded.get(track, 0) + 1

        total = sum(loaded.values())
        if total > 0:
            logger.info(
                "Crash recovery: loaded %d open trades (A:%d B:%d C:%d)",
                total, loaded['decision_time'],
                loaded['all_signals'], loaded['buy_only']
            )

    
    def _make_decision_a(self, signals: List[dict], quote_map: Dict,
                         spot: float, ts_et: datetime, ts_utc: str):
        """Filter → rank → select → enter trades for Track A."""
        track = 'decision_time'
        rejections: Dict[str, int] = {}
        eligible = []

        # ── Regime overlay (H4): adjust thresholds for current regime ──
        filters = dict(self.cfg.get('filters', {}))
        filters['_use_eu_scoring'] = self.cfg.get('use_eu_scoring', False)
        if self.cfg.get('use_regime_thresholds', False):
            from calibration.regime_detector import get_adjusted_thresholds
            regime_overrides = get_adjusted_thresholds(self._current_regime)
            filters.update(regime_overrides)

        # ── Filter ──
        for sig in signals:
            reason = self._check_eligibility_a(sig, filters_override=filters)
            if reason:
                rejections[reason] = rejections.get(reason, 0) + 1
                self.journal.log_skip(track, sig, reason)
            else:
                eligible.append(sig)

        # ── Diversify: enforce min_strike_spacing ──
        eligible = self._diversify(eligible)

        # ── Rank ──
        ranked = self._rank_signals(eligible)

        # ── Select top-N ──
        max_trades = self.cfg.get('max_trades_per_decision', 2)
        max_open = self.cfg.get('max_open_positions', 5)
        current_open = len(self._open_trades_a)
        slots = min(max_trades, max_open - current_open)
        selected = ranked[:max(0, slots)]

        # ── Enter ──
        max_per_strike = self.cfg.get('max_entries_per_strike', 0)
        n_entered = 0
        for sig in selected:
            # Cooldown check
            if self._in_cooldown(sig, track, ts_et):
                rejections['cooldown'] = rejections.get('cooldown', 0) + 1
                self.journal.log_skip(track, sig, 'cooldown')
                continue
            # Per-strike concentration limit
            if max_per_strike > 0:
                strike = sig.get('strike')
                counts = self._strike_entry_counts.setdefault(track, {})
                if counts.get(strike, 0) >= max_per_strike:
                    rejections['strike_concentration'] = rejections.get('strike_concentration', 0) + 1
                    self.journal.log_skip(track, sig, 'strike_concentration')
                    continue
            self._enter_trade(sig, spot, ts_utc, track, decision_ts=ts_utc)
            n_entered += 1

        # Console summary
        time_str = ts_et.strftime('%H:%M')
        print(f"\n  [Paper] Decision @ {time_str} ET: "
              f"{len(signals)} signals -> {len(eligible)} eligible -> "
              f"{n_entered} entered | {len(self._open_trades_a)} open")
        if rejections:
            rej_str = ', '.join(f'{k}:{v}' for k, v in sorted(rejections.items(), key=lambda x: -x[1]))
            print(f"           Rejections: {rej_str}")

        # ── Snapshot ──
        self.journal.record_snapshot(
            run_id=self.run_id, track=track, timestamp_utc=ts_utc,
            spot_price=spot, n_signals=len(signals),
            n_eligible=len(eligible), n_selected=len(selected),
            n_entered=n_entered, rejection_breakdown=rejections,
            signals_json=json.dumps([_signal_summary(s) for s in signals], default=str),
            intraday_move_pct=self._intraday_move_pct,
            regime=self._current_regime,
        )

    def _check_eligibility_a(self, sig: dict,
                             filters_override: dict = None) -> Optional[str]:
        """Return rejection reason or None if eligible."""
        f = filters_override or self.cfg.get('filters', {})
        action = sig.get('action', 'HOLD')

        if action == 'HOLD':
            return 'hold_signal'

        # Action whitelist (e.g. disable SELL for Track A)
        allowed = self.cfg.get('allowed_actions')
        if allowed and action not in allowed:
            return 'action_not_allowed'
        if sig.get('confidence', 0) < f.get('min_confidence', 0):
            return 'low_confidence'
        if sig.get('edge', 0) < f.get('min_edge', 0):
            return 'low_edge'

        bid = sig.get('market_bid', 0)
        ask = sig.get('market_ask', 0)
        mid = (bid + ask) / 2.0 if (bid and ask) else 0
        spread = ask - bid if (bid and ask) else 0

        if mid > 0 and spread / mid > f.get('max_spread_pct', 1.0):
            return 'spread_too_wide'
        if mid < f.get('min_option_mid', 0):
            return 'option_too_cheap'
        max_mid = f.get('max_option_mid')
        if max_mid and mid > max_mid:
            return 'option_too_expensive'

        # Q4: reject sub-dime model prices — MC noise dominates edge
        min_model = f.get('min_model_price')
        if min_model is not None:
            model_price = sig.get('model_price', 0) or 0
            if model_price < min_model:
                return 'model_price_too_low'

        spot_age = sig.get('spot_age_seconds')
        if spot_age is not None and spot_age > f.get('max_spot_age_seconds', 999):
            return 'stale_spot'
        if f.get('skip_bernoulli_violated') and sig.get('bernoulli_violated'):
            return 'bernoulli_violated'

        otm = sig.get('otm_dollars', 0)
        max_otm = f.get('max_otm_dollars')
        if max_otm is not None and otm > max_otm:
            return 'too_far_otm'

        # min TTE
        min_tte = f.get('min_tte_minutes', 0)
        if min_tte > 0:
            eod_str = self.cfg.get('eod_exit_time', '15:55')
            eod_h, eod_m = int(eod_str.split(':')[0]), int(eod_str.split(':')[1])
            # We can't precisely check TTE without the timestamp, so skip
            # this filter gracefully (it's enforced by the caller context)

        # CVaR tail-risk filter (absolute $ OR % of premium)
        cvar = sig.get('cvar_95')
        if cvar is not None:
            # Absolute dollar threshold
            max_cvar_loss = f.get('max_cvar_loss')
            if max_cvar_loss is not None and cvar < max_cvar_loss:
                return 'tail_risk_too_high'
            
            # Premium-normalized threshold: cvar/premium < max_cvar_pct
            # e.g. max_cvar_pct=-3.0 means reject if tail loss > 3× premium
            max_cvar_pct = f.get('max_cvar_pct')
            premium = sig.get('model_price', 0) or sig.get('market_mid', 0)
            if max_cvar_pct is not None and premium > 0.01:
                cvar_ratio = cvar / premium
                if cvar_ratio < max_cvar_pct:
                    return 'tail_risk_too_high'

        # H1: EU gate (if enabled — per-track via filters_override)
        if f.get('_use_eu_scoring', False):
            eu = self._compute_eu(sig)
            sig['eu_score'] = eu
            min_eu = f.get('min_eu', 0.0)
            if eu < min_eu:
                return 'negative_eu'

        return None  # eligible

    # ================================================================== #
    #  H1 / H8: EU scoring and cost accounting                             #
    # ================================================================== #

    def _compute_eu(self, sig: dict) -> float:
        """Compute expected utility: EU = p·G − (1−p)·L − C.

        G and L are in PnL-space (not payoff-space).
        Direction-aware: BUY and SELL have mirrored G/L transforms.
        Cost C is round-trip friction counted exactly once.
        """
        p = self._calibrator.calibrate(sig.get('confidence', 0))
        action = sig.get('action', 'HOLD')

        payoff_pos = sig.get('payoff_mean_pos', 0)   # E[payoff | payoff > 0]
        payoff_zero = sig.get('payoff_mean_zero', 0)  # E[|payoff| | payoff ≤ 0]

        if action == 'BUY':
            exec_price = sig.get('market_ask', 0)
            # Win: payoff > entry price → surplus
            G = max(0.0, payoff_pos - exec_price)
            # Lose: payoff < entry price → loss
            L = max(0.0, exec_price - payoff_zero)
        elif action == 'SELL':
            exec_price = sig.get('market_bid', 0)
            # Win: option expires worthless/cheap → keep premium
            G = max(0.0, exec_price - payoff_zero)
            # Lose: option rallies → must buy back at higher price
            L = max(0.0, payoff_pos - exec_price)
        else:
            return 0.0

        C = self._round_trip_cost(sig)

        return p * G - (1 - p) * L - C

    def _round_trip_cost(self, sig: dict) -> float:
        """Entry + exit friction in option-price units (per contract ÷ 100).

        Uses fill_model for consistency with PnL computation.
        Counted exactly once — do not add spread penalties elsewhere.
        """
        spread = sig.get('spread', 0)

        # Fees: entry + exit (2 legs)
        fees = _fees_per_contract(self.cfg) * 2

        # Slippage: entry + exit
        slip_in = _compute_slippage(spread, self.cfg, direction=1)
        slip_out = _compute_slippage(spread, self.cfg, direction=-1)

        # Convert fees from per-contract-$ to per-option-$ (÷100 multiplier)
        return (fees / 100.0) + slip_in + slip_out

    def _enter_all_signals_b(self, signals: List[dict], quote_map: Dict,
                             spot: float, ts_et: datetime, ts_utc: str):
        """Enter all actionable signals for Track B (research)."""
        track = 'all_signals'
        b_cfg = self.cfg.get('all_signals', {})
        apply = b_cfg.get('apply_filters', {})
        rejections: Dict[str, int] = {}
        n_entered = 0

        for sig in signals:
            action = sig.get('action', 'HOLD')
            if action == 'HOLD':
                rejections['hold_signal'] = rejections.get('hold_signal', 0) + 1
                continue

            # Data-integrity filters
            reason = self._check_eligibility_b(sig, apply)
            if reason:
                rejections[reason] = rejections.get(reason, 0) + 1
                continue

            # Dedup: skip if open trade exists for same (strike, action)
            strike = sig.get('strike')
            if b_cfg.get('dedup_policy') == 'one_open_per_strike_action':
                if self.journal.has_open_trade(self.run_id, track, strike, action):
                    rejections['dedup_open'] = rejections.get('dedup_open', 0) + 1
                    continue

            self._enter_trade(sig, spot, ts_utc, track)
            n_entered += 1

        # Snapshot for Track B
        self.journal.record_snapshot(
            run_id=self.run_id, track=track, timestamp_utc=ts_utc,
            spot_price=spot, n_signals=len(signals),
            n_eligible=len(signals) - sum(rejections.values()),
            n_selected=n_entered, n_entered=n_entered,
            rejection_breakdown=rejections,
        )

    # ================================================================== #
    #  Track C: buy-only (calls + puts, no selling)                        #
    # ================================================================== #

    def _make_decision_c(self, signals: List[dict], quote_map: Dict,
                         spot: float, ts_et: datetime, ts_utc: str):
        """Filter → dedup → rank → select → enter for Track C (buy-only).

        Only BUY signals are accepted. Calls and puts both allowed.
        Uses Track C-specific config for thresholds, limits, and policy.
        Cross-track awareness: skips strikes already open in Track A.
        """
        track = 'buy_only'
        c_cfg = self.cfg.get('buy_only', {})
        rejections: Dict[str, int] = {}
        eligible = []

        # Build Track C filter config (merge base with track-specific)
        c_filters = dict(c_cfg.get('filters', self.cfg.get('filters', {})))
        c_filters['_use_eu_scoring'] = c_cfg.get('use_eu_scoring', False)

        # Regime overlay (H4) if enabled for Track C
        if c_cfg.get('use_regime_thresholds', False):
            from calibration.regime_detector import get_adjusted_thresholds
            regime_overrides = get_adjusted_thresholds(self._current_regime)
            c_filters.update(regime_overrides)

        # Pre-compute open strikes (Track C dedup + cross-track awareness)
        open_c_strikes = {
            (t['strike'], t.get('option_type', 'call'))
            for t in self._open_trades_c.values()
        }
        open_a_strikes = {
            (t['strike'], t.get('option_type', 'call'))
            for t in self._open_trades_a.values()
        }

        # Filter
        allowed_types = c_cfg.get('option_types', ['call', 'put'])
        for sig in signals:
            action = sig.get('action', 'HOLD')
            opt_type = sig.get('option_type', 'call')
            strike = sig.get('strike')

            # Track C: BUY only
            if action != 'BUY':
                rejections['not_buy'] = rejections.get('not_buy', 0) + 1
                continue

            # Option type filter
            if opt_type not in allowed_types:
                rejections['wrong_type'] = rejections.get('wrong_type', 0) + 1
                continue

            # Dedup: skip if Track C already has this (strike, type) open
            if (strike, opt_type) in open_c_strikes:
                rejections['dedup_open'] = rejections.get('dedup_open', 0) + 1
                continue

            # Cross-track: skip if Track A already holds same (strike, type)
            if (strike, opt_type) in open_a_strikes:
                rejections['cross_track_dup'] = rejections.get('cross_track_dup', 0) + 1
                continue

            # Apply standard eligibility with Track C filters
            reason = self._check_eligibility_a(sig, filters_override=c_filters)
            if reason:
                rejections[reason] = rejections.get(reason, 0) + 1
                self.journal.log_skip(track, sig, reason)
            else:
                eligible.append(sig)

        # Diversify
        eligible = self._diversify(eligible)

        # Rank (using Track C policy — passed explicitly, no cfg mutation)
        c_policy = c_cfg.get('selection_policy',
                             self.cfg.get('selection_policy', 'eu_ranked'))
        ranked = self._rank_signals(eligible, policy_override=c_policy)

        # Select top-N respecting position limits
        max_trades = c_cfg.get('max_trades_per_decision', 1)
        max_open = c_cfg.get('max_open_positions', 3)
        slots = max(0, min(max_trades, max_open - len(self._open_trades_c)))
        selected = ranked[:slots]

        # Enter
        n_entered = 0
        for sig in selected:
            if self._in_cooldown(sig, track, ts_et):
                rejections['cooldown'] = rejections.get('cooldown', 0) + 1
                self.journal.log_skip(track, sig, 'cooldown')
                continue
            self._enter_trade(sig, spot, ts_utc, track, decision_ts=ts_utc)
            n_entered += 1

        # Console summary
        time_str = ts_et.strftime('%H:%M')
        print(f"\n  [Paper] Track C Decision @ {time_str} ET: "
              f"{len(signals)} signals -> {len(eligible)} eligible -> "
              f"{n_entered} entered | {len(self._open_trades_c)} open")
        if rejections:
            rej_str = ', '.join(f'{k}:{v}' for k, v in sorted(
                rejections.items(), key=lambda x: -x[1]))
            print(f"           Rejections: {rej_str}")

        # Snapshot
        self.journal.record_snapshot(
            run_id=self.run_id, track=track, timestamp_utc=ts_utc,
            spot_price=spot, n_signals=len(signals),
            n_eligible=len(eligible), n_selected=len(selected),
            n_entered=n_entered, rejection_breakdown=rejections,
            signals_json=json.dumps(
                [_signal_summary(s) for s in signals], default=str),
            intraday_move_pct=self._intraday_move_pct,
            regime=self._current_regime,
        )

    def _check_eligibility_b(self, sig: dict, apply: dict) -> Optional[str]:
        """Minimal data-integrity filters for Track B."""
        if apply.get('use_spot_age', True):
            spot_age = sig.get('spot_age_seconds')
            max_age = self.cfg.get('filters', {}).get('max_spot_age_seconds', 10)
            if spot_age is not None and spot_age > max_age:
                return 'stale_spot'

        if apply.get('use_spread_pct', False):
            bid = sig.get('market_bid', 0)
            ask = sig.get('market_ask', 0)
            mid = (bid + ask) / 2.0 if (bid and ask) else 0
            spread = ask - bid if (bid and ask) else 0
            max_pct = self.cfg.get('filters', {}).get('max_spread_pct', 0.10)
            if mid > 0 and spread / mid > max_pct:
                return 'spread_too_wide'

        if apply.get('use_min_option_mid', False):
            bid = sig.get('market_bid', 0)
            ask = sig.get('market_ask', 0)
            mid = (bid + ask) / 2.0 if (bid and ask) else 0
            min_mid = self.cfg.get('filters', {}).get('min_option_mid', 0.05)
            if mid < min_mid:
                return 'option_too_cheap'

        return None


    def _enter_trade(self, sig: dict, spot: float, ts_utc: str,
                     track: str, decision_ts: str = None):
        """Enter a paper trade using the fill model."""
        bid = sig.get('market_bid', 0)
        ask = sig.get('market_ask', 0)
        action = sig['action']

        fill = simulate_fill(action, bid, ask, self.cfg)
        qty = self._compute_quantity(sig)
        fees = compute_fees(qty, self.cfg)

        trade_id = self.journal.open_trade(
            run_id=self.run_id, track=track, signal=sig,
            entry_fill=fill, spot=spot, timestamp_utc=ts_utc,
            quantity=qty, entry_fees=fees,
            decision_timestamp_utc=decision_ts,
        )

        # Cache in memory
        trade = {
            'trade_id': trade_id,
            'strike': sig.get('strike'),
            'action': action,
            'option_type': sig.get('option_type', 'call'),
            'entry_timestamp_utc': ts_utc,
            'entry_fill': fill,
            'quantity': qty,
            'entry_fees': fees,
            'track': track,
        }
        # Track per-strike entry count
        counts = self._strike_entry_counts.setdefault(track, {})
        strike_val = sig.get('strike')
        counts[strike_val] = counts.get(strike_val, 0) + 1

        if track == 'decision_time':
            self._open_trades_a[trade_id] = trade
            print(f"\n  >>> PAPER TRADE ENTERED [Track A] <<<")
            print(f"      {action} {sig.get('option_type', 'call').upper()} "
                  f"{sig.get('strike')} @ mid=${fill['mid']:.4f}  "
                  f"touch=${fill['touch']:.4f}  spread=${fill['spread']:.4f}")
            print(f"      edge={sig.get('edge', 0)*100:.1f}%  "
                  f"conf={sig.get('confidence', 0)*100:.0f}%  "
                  f"OTM=${sig.get('otm_dollars', 0):.1f}")
        elif track == 'buy_only':
            self._open_trades_c[trade_id] = trade
            print(f"\n  >>> PAPER TRADE ENTERED [Track C] <<<")
            print(f"      BUY {sig.get('option_type', 'call').upper()} "
                  f"{sig.get('strike')} @ mid=${fill['mid']:.4f}  "
                  f"touch=${fill['touch']:.4f}  spread=${fill['spread']:.4f}")
            print(f"      edge={sig.get('edge', 0)*100:.1f}%  "
                  f"conf={sig.get('confidence', 0)*100:.0f}%  "
                  f"OTM=${sig.get('otm_dollars', 0):.1f}")
        else:
            self._open_trades_b[trade_id] = trade


    def _get_quote(self, quote_map, strike, option_type='call'):
        """Look up quote by (strike, type) tuple, falling back to strike-only."""
        return quote_map.get((strike, option_type)) or quote_map.get(strike)

    def _check_exits(self, quote_map: Dict,
                     ts_et: datetime, ts_utc: str):
        """Check exit conditions for all open trades, all tracks."""
        # Merge all track caches
        all_open = {
            **{tid: {**t, '_cache': '_a'} for tid, t in self._open_trades_a.items()},
            **{tid: {**t, '_cache': '_b'} for tid, t in self._open_trades_b.items()},
            **{tid: {**t, '_cache': '_c'} for tid, t in self._open_trades_c.items()},
        }

        for trade_id, trade in list(all_open.items()):
            strike = trade['strike']
            action = trade['action']
            track = trade['track']
            opt_type = trade.get('option_type', 'call')
            quote = self._get_quote(quote_map, strike, opt_type)

            # Determine entry time
            entry_ts = datetime.fromisoformat(trade['entry_timestamp_utc'])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=ZoneInfo('UTC'))
            hold_td = ts_et.astimezone(ZoneInfo('UTC')) - entry_ts.astimezone(ZoneInfo('UTC'))
            hold_minutes = hold_td.total_seconds() / 60.0

            exit_reason = None
            exit_bid = None
            exit_ask = None
            spot_exit = None

            # ── EOD exit (clock-based, always checked) ──
            if track == 'all_signals':
                eod_str = self.cfg['all_signals']['eod_exit_time']
            elif track == 'buy_only':
                eod_str = self.cfg.get('buy_only', {}).get('eod_exit_time',
                          self.cfg.get('eod_exit_time', '15:55'))
            else:
                eod_str = self.cfg.get('eod_exit_time', '15:55')
            eod_h, eod_m = int(eod_str.split(':')[0]), int(eod_str.split(':')[1])
            if ts_et.hour > eod_h or (ts_et.hour == eod_h and ts_et.minute >= eod_m):
                exit_reason = 'eod_exit'

            # ── Option-quote-based exits ──
            if quote and not exit_reason:
                exit_bid = quote.get('bid', 0)
                exit_ask = quote.get('ask', 0)
                spot_exit = quote.get('spot')
                quote_age = quote.get('spot_age', 0)
                max_age = self.cfg.get('max_quote_age_seconds_for_exits', 120)

                quote_fresh = (quote_age is not None and quote_age <= max_age)

                if track == 'decision_time' or track == 'buy_only':
                    # Resolve per-track exit config
                    t_cfg = (self.cfg.get('buy_only', {})
                             if track == 'buy_only' else self.cfg)
                    exit_reason = self._check_exit_a(
                        trade, exit_bid, exit_ask, hold_minutes, quote_fresh,
                        ts_et=ts_et, spot=spot_exit, track_cfg=t_cfg
                    )
                else:
                    exit_reason = self._check_exit_b(
                        trade, hold_minutes, exit_bid, exit_ask)

            elif not quote and not exit_reason:
                # No quote available — skip price-based exits
                continue

            if exit_reason:
                self._close_trade(trade_id, trade, exit_bid, exit_ask,
                                  spot_exit, ts_utc, exit_reason, hold_minutes)

    def _check_exit_a(self, trade: dict, bid: float, ask: float,
                      hold_minutes: float, quote_fresh: bool,
                      ts_et: datetime = None, spot: float = None,
                      track_cfg: dict = None) -> Optional[str]:
        """Track A/C exit: hybrid (TP/SL/theta_decay/time/EOD)."""
        cfg = track_cfg or self.cfg
        mode = cfg.get('exit_mode', self.cfg.get('exit_mode', 'hybrid'))
        mid = (bid + ask) / 2.0

        entry_fill = trade['entry_fill']
        entry_mid = entry_fill.get('mid', 0)
        action = trade['action']

        if entry_mid <= 0:
            return None

        # Compute unrealized return on mid
        if action == 'BUY':
            unrealized_pct = (mid - entry_mid) / entry_mid
        else:
            unrealized_pct = (entry_mid - mid) / entry_mid

        # TP/SL (only if quote is fresh)
        if quote_fresh and mode in ('tp_sl', 'hybrid'):
            # H4 extension: regime-conditioned TP/SL
            base_tp = cfg.get('tp_pct', self.cfg.get('tp_pct', 0.20))
            base_sl = cfg.get('sl_pct', self.cfg.get('sl_pct', 0.15))

            if self.cfg.get('use_regime_thresholds', False):
                from calibration.regime_detector import get_exit_params
                regime_exits = get_exit_params(
                    self._current_regime,
                    default_tp=base_tp, default_sl=base_sl
                )
                tp = regime_exits['tp_pct']
                sl = regime_exits['sl_pct']
            else:
                tp = base_tp
                sl = base_sl

            if unrealized_pct >= tp:
                return 'take_profit'
            if unrealized_pct <= -sl:
                return 'stop_loss'

        # ── Theta-decay-aware exit (improved) ──
        greeks_cfg = self.cfg.get('greeks_exit', {})
        if greeks_cfg.get('enabled', False) and ts_et is not None and spot is not None:
            min_profit = greeks_cfg.get('min_profit_pct', 0.05)

            if unrealized_pct > min_profit:
                strike = trade['strike']
                option_type = trade.get('option_type', 'call')

                # Correct intrinsic value using spot vs strike
                if option_type == 'call':
                    intrinsic = max(0.0, spot - strike)
                else:
                    intrinsic = max(0.0, strike - spot)

                # Time value = option mid − intrinsic
                time_value = max(0.0, mid - intrinsic)

                # Minutes remaining until EOD
                eod_str = self.cfg.get('eod_exit_time', '15:55')
                eod_h, eod_m = int(eod_str.split(':')[0]), int(eod_str.split(':')[1])
                eod_minutes = eod_h * 60 + eod_m
                now_minutes = ts_et.hour * 60 + ts_et.minute
                minutes_left = max(1, eod_minutes - now_minutes)

                # Don't fire theta exit right before close — EOD exit handles that
                min_minutes = greeks_cfg.get('min_minutes_left', 5)
                if minutes_left <= min_minutes:
                    pass  # skip theta check, let EOD handle it
                elif time_value > 0:
                    lookforward = greeks_cfg.get('lookforward_minutes', 15)
                    decay_threshold = greeks_cfg.get('theta_decay_pct', 0.80)

                    # √t decay model: theta ∝ 1/√(minutes_left)
                    # Projected fraction of time_value lost in lookforward window:
                    #   1 - √(minutes_left - lookforward) / √(minutes_left)
                    remaining_after = max(1, minutes_left - lookforward)
                    projected_decay_frac = 1.0 - (remaining_after / minutes_left) ** 0.5

                    # Spread cost guard: don't exit if spread > remaining time value
                    exit_spread = ask - bid
                    if exit_spread > 0.8 * time_value:
                        pass  # spread would eat most of what we're trying to save
                    elif projected_decay_frac > decay_threshold:
                        return 'theta_decay'

        # Time exit
        if mode in ('time', 'hybrid'):
            max_mins = cfg.get('exit_time_minutes',
                               self.cfg.get('exit_time_minutes', 30))
            if hold_minutes >= max_mins:
                return 'time_exit'

        return None

    def _check_exit_b(self, trade: dict, hold_minutes: float,
                      bid: float = None, ask: float = None) -> Optional[str]:
        """Track B exit: fixed_horizon, absolute_stop, or eod."""
        b_cfg = self.cfg.get('all_signals', {})
        mode = b_cfg.get('exit_mode', 'fixed_horizon')

        # Absolute dollar stop-loss (per contract, checked before horizon)
        max_loss = b_cfg.get('max_loss_dollars')
        if max_loss is not None and bid is not None and ask is not None:
            entry_mid = trade['entry_fill'].get('mid', 0)
            mid = (bid + ask) / 2.0
            action = trade['action']
            if entry_mid > 0:
                if action == 'BUY':
                    unrealized_dollar = (entry_mid - mid) * 100  # loss is positive
                else:
                    unrealized_dollar = (mid - entry_mid) * 100
                if unrealized_dollar >= max_loss:
                    return 'absolute_stop'

        if mode == 'fixed_horizon':
            horizon = b_cfg.get('exit_horizon_minutes', 10)
            if hold_minutes >= horizon:
                return 'fixed_horizon'

        return None

    def _close_trade(self, trade_id: str, trade: dict,
                     exit_bid: float, exit_ask: float,
                     spot_exit: float, ts_utc: str,
                     exit_reason: str, hold_minutes: float):
        """Close a trade: compute 3-tier PnL and persist."""
        action = trade['action']
        qty = trade.get('quantity', 1)

        # If no exit quotes available (e.g. EOD no quote), use entry as fallback
        if exit_bid is None or exit_ask is None:
            exit_bid = trade['entry_fill'].get('touch', 0)
            exit_ask = trade['entry_fill'].get('touch', 0)

        exit_fill = simulate_fill(
            'SELL' if action == 'BUY' else 'BUY',
            exit_bid, exit_ask, self.cfg
        )

        entry_fees = trade.get('entry_fees', 0)
        exit_fees = compute_fees(qty, self.cfg)

        entry_fill = trade['entry_fill']

        # 3-tier PnL
        pnl_mid = compute_pnl(action, entry_fill['mid'], exit_fill['mid'],
                              qty, entry_fees, exit_fees)
        pnl_touch = compute_pnl(action, entry_fill['touch'], exit_fill['touch'],
                                qty, entry_fees, exit_fees)
        pnl_slippage = compute_pnl(action, entry_fill['slippage'],
                                   exit_fill['slippage'], qty,
                                   entry_fees, exit_fees)

        # MAE/MFE from quote tape
        mae, mfe = self._compute_mae_mfe(trade)

        self.journal.close_trade(
            trade_id=trade_id,
            exit_bid=exit_bid, exit_ask=exit_ask, exit_fill=exit_fill,
            spot_exit=spot_exit or 0, timestamp_utc=ts_utc,
            exit_reason=exit_reason,
            pnl_mid=pnl_mid, pnl_touch=pnl_touch, pnl_slippage=pnl_slippage,
            hold_minutes=hold_minutes, exit_fees=exit_fees,
            mae=mae, mfe=mfe,
        )

        # Remove from in-memory cache
        track = trade['track']
        if track == 'decision_time':
            self._open_trades_a.pop(trade_id, None)
        elif track == 'buy_only':
            self._open_trades_c.pop(trade_id, None)
        else:
            self._open_trades_b.pop(trade_id, None)

        # Record close time for cooldown
        strike = trade['strike']
        action = trade['action']
        key = f"{track}:{strike}:{action}"
        self._trade_close_times[key] = datetime.fromisoformat(ts_utc)

        # Console notification
        pnl_val = pnl_touch.get('net_pnl', 0)
        tag = {'decision_time': 'Track A', 'all_signals': 'Track B',
               'buy_only': 'Track C'}.get(track, track)
        icon = 'WIN' if pnl_val > 0 else 'LOSS'
        if track in ('decision_time', 'buy_only'):
            print(f"\n  <<< PAPER TRADE CLOSED [{tag}] {icon} >>>")
            print(f"      {action} {strike} | {exit_reason} | "
                  f"PnL(touch) ${pnl_val:+.2f} | held {hold_minutes:.0f}min")


    def _is_decision_time(self, ts_et: datetime) -> bool:
        """Check if current ET time matches a decision time (±30s, fire once)."""
        current_hhmm = ts_et.strftime('%H:%M')
        today = ts_et.date()

        for dt_str in self.cfg.get('decision_times', []):
            if self._decision_fired_today.get(dt_str) == today:
                continue  # already fired today

            dt_h, dt_m = int(dt_str.split(':')[0]), int(dt_str.split(':')[1])
            decision_time = ts_et.replace(hour=dt_h, minute=dt_m, second=0, microsecond=0)
            diff = abs((ts_et - decision_time).total_seconds())

            if diff <= 30:
                self._decision_fired_today[dt_str] = today
                return True

        return False

    def _rank_signals(self, signals: List[dict],
                      policy_override: str = None) -> List[dict]:
        """Rank signals by the configured selection policy."""
        policy = policy_override or self.cfg.get('selection_policy', 'risk_adjusted')
        spread_floor = self.cfg.get('spread_floor', 0.01)

        def _score(s):
            edge = abs(s.get('edge', 0))
            conf = s.get('confidence', 0)
            bid = s.get('market_bid', 0)
            ask = s.get('market_ask', 0)
            spread = max(ask - bid, spread_floor)

            if policy == 'confidence':
                return conf
            elif policy == 'edge':
                return edge
            elif policy == 'risk_adjusted':
                return edge * conf / spread
            elif policy == 'top1_confidence':
                return conf  # caller takes top-1
            elif policy == 'atm_only':
                # Lower OTM distance = higher score
                otm = s.get('otm_dollars', 0)
                return -otm  # closest to ATM first
            elif policy == 'closest_to_spot':
                otm = s.get('otm_dollars', 0)
                return -otm
            elif policy == 'eu_ranked':
                # H8: score = EU / sqrt(|CVaR|)
                # Compute EU if not already set by eligibility gate
                eu = s.get('eu_score')
                if eu is None:
                    eu = self._compute_eu(s)
                    s['eu_score'] = eu
                cvar = abs(s.get('cvar_95', 1.0)) or 1.0
                return eu / max(0.01, cvar ** 0.5)
            else:
                return edge * conf / spread

        return sorted(signals, key=_score, reverse=True)

    def _diversify(self, signals: List[dict]) -> List[dict]:
        """Enforce min_strike_spacing to prevent adjacent-strike clustering."""
        spacing = self.cfg.get('min_strike_spacing', 1.0)
        if spacing <= 0:
            return signals

        # Sort by score first (we want to keep higher-ranked ones)
        ranked = self._rank_signals(signals)
        result = []
        used_strikes = []

        for sig in ranked:
            strike = sig.get('strike', 0)
            too_close = any(abs(strike - s) < spacing for s in used_strikes)
            if not too_close:
                result.append(sig)
                used_strikes.append(strike)

        return result

    def _in_cooldown(self, sig: dict, track: str, ts_et: datetime) -> bool:
        """Check if a trade is in cooldown period."""
        cooldown = self.cfg.get('cooldown_minutes', 0)
        if cooldown <= 0:
            return False

        scope = self.cfg.get('cooldown_scope', 'same_direction')
        strike = sig.get('strike')
        action = sig.get('action')

        if scope == 'same_direction':
            key = f"{track}:{strike}:{action}"
        else:  # any_direction
            # Check both directions
            for a in ['BUY', 'SELL']:
                key = f"{track}:{strike}:{a}"
                last_close = self._trade_close_times.get(key)
                if last_close:
                    ts_utc = ts_et.astimezone(ZoneInfo('UTC'))
                    if last_close.tzinfo is None:
                        last_close = last_close.replace(tzinfo=ZoneInfo('UTC'))
                    elapsed = (ts_utc - last_close).total_seconds() / 60.0
                    if elapsed < cooldown:
                        return True
            return False

        last_close = self._trade_close_times.get(key)
        if not last_close:
            return False

        ts_utc = ts_et.astimezone(ZoneInfo('UTC'))
        if last_close.tzinfo is None:
            last_close = last_close.replace(tzinfo=ZoneInfo('UTC'))
        elapsed = (ts_utc - last_close).total_seconds() / 60.0
        return elapsed < cooldown

    def _compute_quantity(self, sig: dict) -> int:
        """Compute position size based on config."""
        mode = self.cfg.get('sizing_mode', 'fixed_contracts')

        if mode == 'fixed_contracts':
            return self.cfg.get('fixed_contracts', 1)
        elif mode == 'fixed_dollar_risk':
            # risk = entry_price × 100 × sl_pct
            bid = sig.get('market_bid', 0)
            ask = sig.get('market_ask', 0)
            mid = (bid + ask) / 2.0
            sl_pct = self.cfg.get('sl_pct', 0.15)
            dollar_risk = self.cfg.get('fixed_dollar_risk', 100)
            risk_per = mid * 100 * sl_pct
            if risk_per > 0:
                return max(1, int(dollar_risk / risk_per))
            return 1
        elif mode == 'confidence_scaled':
            conf = sig.get('confidence', 0.5)
            base = self.cfg.get('fixed_contracts', 1)
            return max(1, int(base * conf * 2))
        return 1

    def _compute_mae_mfe(self, trade: dict) -> Tuple[Optional[float], Optional[float]]:
        """Compute max adverse/favorable excursion from quote tape."""
        strike = trade['strike']
        entry_ts = trade['entry_timestamp_utc']
        entry_mid = trade['entry_fill'].get('mid', 0)
        action = trade['action']

        if entry_mid <= 0:
            return None, None

        quotes = self.journal.get_quotes_for_strike(
            self.run_id, strike, since_utc=entry_ts
        )
        if not quotes:
            return None, None

        max_gap = self.cfg.get('max_quote_gap_minutes_for_forward_eval', 2)
        mae = 0.0
        mfe = 0.0

        prev_ts = None
        for q in quotes:
            mid = q.get('mid', 0)
            if mid <= 0:
                continue

            # Check gap
            if prev_ts:
                try:
                    qt = datetime.fromisoformat(q['timestamp_utc'])
                    pt = datetime.fromisoformat(prev_ts)
                    gap_min = (qt - pt).total_seconds() / 60.0
                    if gap_min > max_gap:
                        continue  # stale gap, skip
                except (ValueError, TypeError):
                    pass
            prev_ts = q.get('timestamp_utc')

            if action == 'BUY':
                pnl_pct = (mid - entry_mid) / entry_mid
            else:
                pnl_pct = (entry_mid - mid) / entry_mid

            mfe = max(mfe, pnl_pct)
            mae = min(mae, pnl_pct)

        return round(mae, 6) if mae < 0 else None, round(mfe, 6) if mfe > 0 else None

    def _record_quotes(self, quote_map: Dict, spot: float,
                       ts_utc: str):
        """Record all current quotes to the tape.
        
        Supports both tuple-key (strike, option_type) and
        legacy strike-only key formats.
        """
        quotes = []
        for key, q in quote_map.items():
            # Handle tuple key (strike, opt_type) or plain strike
            if isinstance(key, tuple):
                strike, opt_type = key
            else:
                strike = key
                opt_type = q.get('option_type', 'call')
            bid = q.get('bid', 0)
            ask = q.get('ask', 0)
            mid = (bid + ask) / 2.0 if bid and ask else 0
            spread = ask - bid if bid and ask else 0
            quotes.append({
                'strike': strike,
                'option_type': opt_type,
                'bid': bid,
                'ask': ask,
                'mid': round(mid, 4),
                'spread': round(spread, 4),
                'spot': spot,
                'spot_age': q.get('spot_age', 0),
            })
        self.journal.record_quotes(self.run_id, ts_utc, quotes)

    def close_eod(self, quote_map: Dict[float, dict] = None,
                  spot: float = 0, ts_et: datetime = None):
        """Force-close all remaining open trades at EOD."""
        if ts_et is None:
            ts_et = now_et()
        ts_utc = to_utc_str(ts_et)

        # Single loop over all tracks
        all_caches = [
            self._open_trades_a,
            self._open_trades_b,
            self._open_trades_c,
        ]
        for cache in all_caches:
            for trade_id, trade in list(cache.items()):
                strike = trade['strike']
                opt_type = trade.get('option_type', 'call')
                # Look up quote by (strike, type) tuple first, then plain strike
                q = (
                    (quote_map or {}).get((strike, opt_type))
                    or (quote_map or {}).get(strike, {})
                )
                exit_bid = q.get('bid', trade['entry_fill'].get('touch', 0))
                exit_ask = q.get('ask', trade['entry_fill'].get('touch', 0))
                entry_ts = datetime.fromisoformat(trade['entry_timestamp_utc'])
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.replace(tzinfo=ZoneInfo('UTC'))
                hold = (ts_et.astimezone(ZoneInfo('UTC')) - entry_ts).total_seconds() / 60.0
                self._close_trade(trade_id, trade, exit_bid, exit_ask,
                                  spot, ts_utc, 'eod_exit', hold)

        self.journal.end_run(self.run_id)

    def get_stats(self, track: str = None) -> dict:
        """Get aggregate stats for the current run."""
        return self.journal.trade_summary(self.run_id, track=track)


def _signal_summary(sig: dict) -> dict:
    """Compact version of a signal for JSON storage."""
    return {
        'strike': sig.get('strike'),
        'action': sig.get('action'),
        'edge': sig.get('edge'),
        'confidence': sig.get('confidence'),
        'model_price': sig.get('model_price'),
        'market_mid': sig.get('market_mid'),
    }
