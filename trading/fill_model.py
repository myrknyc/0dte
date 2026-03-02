from trading.paper_config import PAPER_TRADING


def simulate_fill(action: str, bid: float, ask: float,
                  config: dict = None) -> dict:
    
    cfg = config or PAPER_TRADING
    mid = (bid + ask) / 2.0
    spread = ask - bid

    if action == 'BUY':
        touch = ask
        slip = _compute_slippage(spread, cfg, direction=1)
        slippage_fill = touch + slip
    elif action == 'SELL':
        touch = bid
        slip = _compute_slippage(spread, cfg, direction=-1)
        slippage_fill = touch - slip
    else:
        # HOLD — no fill
        return {'mid': mid, 'touch': mid, 'slippage': mid,
                'fees_per_contract': 0.0, 'spread': spread}

    fees = _fees_per_contract(cfg)

    return {
        'mid': round(mid, 4),
        'touch': round(touch, 4),
        'slippage': round(slippage_fill, 4),
        'fees_per_contract': round(fees, 4),
        'spread': round(spread, 4),
    }


def compute_fees(quantity: int, config: dict = None) -> float:
    
    cfg = config or PAPER_TRADING
    per = _fees_per_contract(cfg)
    return round(per * quantity, 4)


def compute_pnl(action: str, entry_fill: float, exit_fill: float,
                quantity: int, entry_fees: float, exit_fees: float) -> dict:
    
    multiplier = 100  # options contract multiplier

    if action == 'BUY':
        gross_pnl = (exit_fill - entry_fill) * multiplier * quantity
    elif action == 'SELL':
        gross_pnl = (entry_fill - exit_fill) * multiplier * quantity
    else:
        gross_pnl = 0.0

    total_fees = entry_fees + exit_fees
    net_pnl = gross_pnl - total_fees

    cost_basis = entry_fill * multiplier * quantity
    return_pct = (net_pnl / cost_basis) if cost_basis > 0 else 0.0

    return {
        'gross_pnl': round(gross_pnl, 2),
        'net_pnl': round(net_pnl, 2),
        'return_pct': round(return_pct, 6),
    }


def _compute_slippage(spread: float, cfg: dict, direction: int) -> float:
    
    mode = cfg.get('slippage_mode', 'pct_of_spread')

    if mode == 'fixed':
        return max(0.0, cfg.get('slippage_fixed', 0.01))
    elif mode == 'pct_of_spread':
        pct = cfg.get('slippage_pct', 0.10)
        return max(0.0, spread * pct)
    else:
        # Default to pct_of_spread
        return max(0.0, spread * 0.10)


def _fees_per_contract(cfg: dict) -> float:
    """Commission + exchange/reg fees per contract."""
    comm = cfg.get('commission_per_contract', 0.65)
    exch = cfg.get('exchange_fees_per_contract', 0.05)
    return comm + exch
