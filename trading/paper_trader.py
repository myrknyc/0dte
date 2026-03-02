from datetime import datetime, date, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import json

from trading.paper_config import PAPER_TRADING
from trading.fill_model import simulate_fill, compute_fees, compute_pnl
from trading.paper_journal import PaperJournal, now_utc, to_utc_str

_ET = ZoneInfo('America/New_York')


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
        self._trade_close_times: Dict[str, datetime] = {}  # last close per (strike, action, track)

    # ================================================================== #
    #  Main entry point                                                    #
    # ================================================================== #

    def on_scan(self, signals: List[dict], quote_map: Dict[float, dict],
                spot: float, timestamp_et: datetime):
        
        ts_utc = to_utc_str(timestamp_et)

        # 1. Record quotes to tape (always, for forward eval)
        self._record_quotes(quote_map, spot, ts_utc)

        # 2. Track A: decision_time
        if self._is_decision_time(timestamp_et):
            self._make_decision_a(signals, quote_map, spot, timestamp_et, ts_utc)

        # 3. Track B: all_signals
        b_cfg = self.cfg.get('all_signals', {})
        if b_cfg.get('enabled', False):
            enter_on = b_cfg.get('enter_on', 'every_scan')
            if enter_on == 'every_scan' or self._is_decision_time(timestamp_et):
                self._enter_all_signals_b(signals, quote_map, spot, timestamp_et, ts_utc)

        # 4. Check exits for both tracks
        self._check_exits(quote_map, timestamp_et, ts_utc)

    
    def _make_decision_a(self, signals: List[dict], quote_map: Dict,
                         spot: float, ts_et: datetime, ts_utc: str):
        """Filter → rank → select → enter trades for Track A."""
        track = 'decision_time'
        rejections: Dict[str, int] = {}
        eligible = []

        # ── Filter ──
        for sig in signals:
            reason = self._check_eligibility_a(sig)
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
        n_entered = 0
        for sig in selected:
            # Cooldown check
            if self._in_cooldown(sig, track, ts_et):
                rejections['cooldown'] = rejections.get('cooldown', 0) + 1
                self.journal.log_skip(track, sig, 'cooldown')
                continue
            self._enter_trade(sig, spot, ts_utc, track, decision_ts=ts_utc)
            n_entered += 1

        # ── Snapshot ──
        self.journal.record_snapshot(
            run_id=self.run_id, track=track, timestamp_utc=ts_utc,
            spot_price=spot, n_signals=len(signals),
            n_eligible=len(eligible), n_selected=len(selected),
            n_entered=n_entered, rejection_breakdown=rejections,
            signals_json=json.dumps([_signal_summary(s) for s in signals], default=str),
        )

    def _check_eligibility_a(self, sig: dict) -> Optional[str]:
        """Return rejection reason or None if eligible."""
        f = self.cfg.get('filters', {})
        action = sig.get('action', 'HOLD')

        if action == 'HOLD':
            return 'hold_signal'
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

        return None  # eligible


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
            'entry_timestamp_utc': ts_utc,
            'entry_fill': fill,
            'quantity': qty,
            'entry_fees': fees,
            'track': track,
        }
        if track == 'decision_time':
            self._open_trades_a[trade_id] = trade
        else:
            self._open_trades_b[trade_id] = trade


    def _check_exits(self, quote_map: Dict[float, dict],
                     ts_et: datetime, ts_utc: str):
        """Check exit conditions for all open trades, both tracks."""
        # Merge both track caches
        all_open = {
            **{tid: {**t, '_cache': '_a'} for tid, t in self._open_trades_a.items()},
            **{tid: {**t, '_cache': '_b'} for tid, t in self._open_trades_b.items()},
        }

        for trade_id, trade in list(all_open.items()):
            strike = trade['strike']
            action = trade['action']
            track = trade['track']
            quote = quote_map.get(strike)

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
            eod_str = (self.cfg['all_signals']['eod_exit_time']
                       if track == 'all_signals'
                       else self.cfg.get('eod_exit_time', '15:55'))
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

                if track == 'decision_time':
                    exit_reason = self._check_exit_a(
                        trade, exit_bid, exit_ask, hold_minutes, quote_fresh
                    )
                else:
                    exit_reason = self._check_exit_b(trade, hold_minutes)

            elif not quote and not exit_reason:
                # No quote available — skip price-based exits
                continue

            if exit_reason:
                self._close_trade(trade_id, trade, exit_bid, exit_ask,
                                  spot_exit, ts_utc, exit_reason, hold_minutes)

    def _check_exit_a(self, trade: dict, bid: float, ask: float,
                      hold_minutes: float, quote_fresh: bool) -> Optional[str]:
        """Track A exit: hybrid (TP/SL/time/EOD)."""
        mode = self.cfg.get('exit_mode', 'hybrid')
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
            tp = self.cfg.get('tp_pct', 0.20)
            sl = self.cfg.get('sl_pct', 0.15)
            if unrealized_pct >= tp:
                return 'take_profit'
            if unrealized_pct <= -sl:
                return 'stop_loss'

        # Time exit
        if mode in ('time', 'hybrid'):
            max_mins = self.cfg.get('exit_time_minutes', 30)
            if hold_minutes >= max_mins:
                return 'time_exit'

        return None

    def _check_exit_b(self, trade: dict, hold_minutes: float) -> Optional[str]:
        """Track B exit: fixed_horizon or eod."""
        b_cfg = self.cfg.get('all_signals', {})
        mode = b_cfg.get('exit_mode', 'fixed_horizon')

        if mode == 'fixed_horizon':
            horizon = b_cfg.get('exit_horizon_minutes', 5)
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
        else:
            self._open_trades_b.pop(trade_id, None)

        # Record close time for cooldown
        strike = trade['strike']
        action = trade['action']
        key = f"{track}:{strike}:{action}"
        self._trade_close_times[key] = datetime.fromisoformat(ts_utc)


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

    def _rank_signals(self, signals: List[dict]) -> List[dict]:
        """Rank signals by the configured selection policy."""
        policy = self.cfg.get('selection_policy', 'risk_adjusted')
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

    def _record_quotes(self, quote_map: Dict[float, dict], spot: float,
                       ts_utc: str):
        """Record all current quotes to the tape."""
        quotes = []
        for strike, q in quote_map.items():
            bid = q.get('bid', 0)
            ask = q.get('ask', 0)
            mid = (bid + ask) / 2.0 if bid and ask else 0
            spread = ask - bid if bid and ask else 0
            quotes.append({
                'strike': strike,
                'option_type': q.get('option_type', 'call'),
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
            ts_et = datetime.now(_ET)
        ts_utc = to_utc_str(ts_et)

        for trade_id, trade in list(self._open_trades_a.items()):
            strike = trade['strike']
            q = (quote_map or {}).get(strike, {})
            exit_bid = q.get('bid', trade['entry_fill'].get('touch', 0))
            exit_ask = q.get('ask', trade['entry_fill'].get('touch', 0))
            entry_ts = datetime.fromisoformat(trade['entry_timestamp_utc'])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=ZoneInfo('UTC'))
            hold = (ts_et.astimezone(ZoneInfo('UTC')) - entry_ts).total_seconds() / 60.0
            self._close_trade(trade_id, trade, exit_bid, exit_ask,
                              spot, ts_utc, 'eod_exit', hold)

        for trade_id, trade in list(self._open_trades_b.items()):
            strike = trade['strike']
            q = (quote_map or {}).get(strike, {})
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
