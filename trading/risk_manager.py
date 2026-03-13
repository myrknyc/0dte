"""DEPRECATED: This module is unused — paper_trader.py implements its own
exit logic (TP/SL/theta_decay/time/EOD).  Kept for reference only.
Remove once paper trader exit logic is fully validated."""

from datetime import datetime, time
from typing import Dict, List, Optional
from config import RISK_DEFAULTS

class Position:    
    def __init__(self, symbol: str, entry_price: float, quantity: int):
        self.symbol = symbol
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = datetime.now()
        
        # Manual overrides (None = use automated)
        self.manual_sl_price = None  # Set specific SL price
        self.manual_tp_price = None  # Set specific TP price
        self.manual_sl_pct = None    # Or set SL percentage
        self.manual_tp_pct = None    # Or set TP percentage
        
    def set_manual_sl(self, price: float = None, pct: float = None):
        if price:
            self.manual_sl_price = price
            print(f"✓ Manual SL set: ${price:.2f}")
        elif pct:
            self.manual_sl_pct = pct
            self.manual_sl_price = self.entry_price * (1 - pct)
            print(f"✓ Manual SL set: {pct*100:.0f}% (${self.manual_sl_price:.2f})")
    
    def set_manual_tp(self, price: float = None, pct: float = None):
        if price:
            self.manual_tp_price = price
            print(f"✓ Manual TP set: ${price:.2f}")
        elif pct:
            self.manual_tp_pct = pct
            self.manual_tp_price = self.entry_price * (1 + pct)
            print(f"✓ Manual TP set: {pct*100:.0f}% (${self.manual_tp_price:.2f})")
    
    def clear_manual_overrides(self):
        """Reset to automated SL/TP"""
        self.manual_sl_price = None
        self.manual_tp_price = None
        self.manual_sl_pct = None
        self.manual_tp_pct = None
        print("✓ Cleared manual overrides, using automated SL/TP")


class FlexibleRiskManager:
    
    def __init__(self):
        # Automated rules from config (override by passing kwargs)
        self.default_sl_pct = RISK_DEFAULTS['sl_pct']
        self.default_tp_pct = RISK_DEFAULTS['tp_pct']
        _ct = RISK_DEFAULTS['close_time']
        self.close_time = time(_ct[0], _ct[1])
        self.max_daily_loss = RISK_DEFAULTS['max_daily_loss']
        
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0
    
    def add_position(self, symbol: str, entry_price: float, quantity: int) -> Position:
        """Add new position with automated SL/TP"""
        position = Position(symbol, entry_price, quantity)
        self.positions[symbol] = position
        
        print(f"\n🆕 Position opened: {symbol}")
        print(f"   Entry: ${entry_price:.2f} × {quantity}")
        print(f"   Automated SL: {self.default_sl_pct*100:.0f}%")
        print(f"   Automated TP: {self.default_tp_pct*100:.0f}%")
        
        return position
    
    def check_position(self, symbol: str, current_price: float) -> Dict:
        if symbol not in self.positions:
            return {'action': 'HOLD', 'reason': 'Unknown position'}
        
        position = self.positions[symbol]
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        pnl_dollars = (current_price - position.entry_price) * 100 * position.quantity
        
        # Check 1: Manual Stop Loss (takes priority)
        if position.manual_sl_price and current_price <= position.manual_sl_price:
            return {
                'action': 'CLOSE',
                'reason': f'Manual SL Hit (${position.manual_sl_price:.2f})',
                'mode': 'MANUAL',
                'pnl': pnl_dollars
            }
        
        # Check 2: Manual Take Profit (takes priority)
        if position.manual_tp_price and current_price >= position.manual_tp_price:
            return {
                'action': 'CLOSE',
                'reason': f'Manual TP Hit (${position.manual_tp_price:.2f})',
                'mode': 'MANUAL',
                'pnl': pnl_dollars
            }
        
        # Check 3: Automated Stop Loss
        sl_pct = position.manual_sl_pct if position.manual_sl_pct else self.default_sl_pct
        if pnl_pct <= -sl_pct:
            mode = 'MANUAL' if position.manual_sl_pct else 'AUTO'
            return {
                'action': 'CLOSE',
                'reason': f'{mode} SL Hit ({pnl_pct*100:.1f}%)',
                'mode': mode,
                'pnl': pnl_dollars
            }
        
        # Check 4: Automated Take Profit
        tp_pct = position.manual_tp_pct if position.manual_tp_pct else self.default_tp_pct
        if pnl_pct >= tp_pct:
            mode = 'MANUAL' if position.manual_tp_pct else 'AUTO'
            return {
                'action': 'CLOSE',
                'reason': f'{mode} TP Hit ({pnl_pct*100:.1f}%)',
                'mode': mode,
                'pnl': pnl_dollars
            }
        
        # Check 5: Time-based exit (always applies)
        if datetime.now().time() >= self.close_time:
            return {
                'action': 'CLOSE',
                'reason': 'Time Exit (3:45 PM)',
                'mode': 'AUTO',
                'pnl': pnl_dollars
            }
        
        # All good
        return {
            'action': 'HOLD',
            'reason': 'Within limits',
            'current_sl': position.manual_sl_price or position.entry_price * (1 - sl_pct),
            'current_tp': position.manual_tp_price or position.entry_price * (1 + tp_pct),
            'pnl': pnl_dollars
        }
    
    def get_position_status(self, symbol: str) -> str:
        """Get human-readable status for dashboard"""
        if symbol not in self.positions:
            return "Unknown"
        
        position = self.positions[symbol]
        
        # Determine SL/TP mode
        if position.manual_sl_price or position.manual_tp_price:
            sl_text = f"${position.manual_sl_price:.2f}" if position.manual_sl_price else f"{self.default_sl_pct*100:.0f}% (Auto)"
            tp_text = f"${position.manual_tp_price:.2f}" if position.manual_tp_price else f"{self.default_tp_pct*100:.0f}% (Auto)"
            mode = "🔧 MANUAL"
        else:
            sl_text = f"{self.default_sl_pct*100:.0f}% (Auto)"
            tp_text = f"{self.default_tp_pct*100:.0f}% (Auto)"
            mode = "🤖 AUTO"
        
        return f"{mode} | SL: {sl_text} | TP: {tp_text}"

