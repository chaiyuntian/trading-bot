"""Risk Manager — position sizing, stop management, and portfolio risk.

Key improvements over v1:
- ATR-adaptive stop-loss and take-profit (not fixed %)
- Trailing stop enabled by default (let winners run)
- Minimum hold period (prevent fee-churning)
- Fee-aware position sizing
- Cooldown after stop-loss (prevent revenge trading)
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger("risk")


@dataclass
class Trade:
    id: str
    symbol: str
    side: str           # "buy" or "sell"
    entry_price: float
    amount: float
    stop_loss: float
    take_profit: float
    entry_time: float = field(default_factory=time.time)
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"  # open, closed, stopped
    entry_candle: int = 0  # candle index at entry
    peak_price: float = 0.0  # highest price since entry (for trailing)


class RiskManager:
    """Manages position sizing, stop losses, and portfolio risk."""

    def __init__(self, config: dict):
        risk_cfg = config.get("risk", {})
        self.initial_capital = float(config["trading"]["initial_capital"])
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital

        self.max_position_pct = risk_cfg.get("max_position_pct", 0.30)
        self.stop_loss_pct = risk_cfg.get("stop_loss_pct", 0.03)
        self.take_profit_pct = risk_cfg.get("take_profit_pct", 0.06)
        self.max_daily_loss_pct = risk_cfg.get("max_daily_loss_pct", 0.05)
        self.max_drawdown_pct = risk_cfg.get("max_drawdown_pct", 0.25)
        self.risk_per_trade_pct = risk_cfg.get("risk_per_trade_pct", 0.02)
        self.max_open_trades = risk_cfg.get("max_open_trades", 3)
        self.trailing_stop = risk_cfg.get("trailing_stop", True)
        self.trailing_stop_pct = risk_cfg.get("trailing_stop_pct", 0.02)

        # New: minimum hold period (candles) — prevent fee-churning
        self.min_hold_candles = risk_cfg.get("min_hold_candles", 8)
        # New: cooldown after stop-loss (candles)
        self.sl_cooldown_candles = risk_cfg.get("sl_cooldown_candles", 4)
        # New: fee rate for break-even calculation
        self.fee_rate = risk_cfg.get("fee_rate", 0.001)

        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.daily_pnl = 0.0
        self.daily_start_capital = self.capital
        self.halted = False
        self.halt_reason = ""

        self._current_candle = 0
        self._last_sl_candle = -100  # last stop-loss candle (for cooldown)

    def set_candle_index(self, idx: int):
        self._current_candle = idx

    def update_capital(self, new_capital: float):
        self.capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

    def get_drawdown(self) -> float:
        if self.peak_capital == 0:
            return 0
        return (self.peak_capital - self.capital) / self.peak_capital

    def can_open_trade(self) -> tuple[bool, str]:
        if self.halted:
            return False, f"Trading halted: {self.halt_reason}"

        if len(self.open_trades) >= self.max_open_trades:
            return False, f"Max open trades ({self.max_open_trades}) reached"

        # Cooldown after stop-loss
        if self._current_candle - self._last_sl_candle < self.sl_cooldown_candles:
            return False, "Stop-loss cooldown"

        drawdown = self.get_drawdown()
        if drawdown >= self.max_drawdown_pct:
            self.halted = True
            self.halt_reason = f"Max drawdown {drawdown:.1%} >= {self.max_drawdown_pct:.1%}"
            logger.critical(self.halt_reason)
            return False, self.halt_reason

        daily_loss = (self.daily_start_capital - self.capital) / self.daily_start_capital if self.daily_start_capital > 0 else 0
        if daily_loss >= self.max_daily_loss_pct:
            self.halted = True
            self.halt_reason = f"Daily loss {daily_loss:.1%} >= {self.max_daily_loss_pct:.1%}"
            logger.critical(self.halt_reason)
            return False, self.halt_reason

        return True, "OK"

    def calculate_position_size(self, entry_price: float, stop_price: float,
                                confidence: float = 1.0) -> float:
        """Position size based on risk per trade, stop distance, and confidence.

        Scales position UP with confidence (not just down), and accounts
        for round-trip fees in the sizing.
        """
        risk_amount = self.capital * self.risk_per_trade_pct
        stop_distance = abs(entry_price - stop_price) / entry_price

        if stop_distance == 0:
            return 0

        # Account for round-trip fees in the risk budget
        fee_cost = entry_price * self.fee_rate * 2  # buy + sell fee
        effective_risk = risk_amount - fee_cost

        if effective_risk <= 0:
            return 0

        position_value = effective_risk / stop_distance
        max_position = self.capital * self.max_position_pct

        # Scale by confidence: 0.5 -> 60%, 0.7 -> 80%, 1.0 -> 100%
        confidence_scale = 0.4 + 0.6 * min(1.0, confidence)
        position_value = min(position_value * confidence_scale, max_position)

        amount = position_value / entry_price
        return amount

    def get_stop_loss(self, entry_price: float, side: str = "buy",
                      atr: float = None) -> float:
        """ATR-adaptive stop-loss when ATR is available."""
        if atr and atr > 0:
            # 2x ATR stop — adapts to current volatility
            multiplier = 2.0
            if side == "buy":
                return entry_price - atr * multiplier
            return entry_price + atr * multiplier

        if side == "buy":
            return entry_price * (1 - self.stop_loss_pct)
        return entry_price * (1 + self.stop_loss_pct)

    def get_take_profit(self, entry_price: float, side: str = "buy",
                        atr: float = None) -> float:
        """ATR-adaptive take-profit when ATR is available."""
        if atr and atr > 0:
            # 4x ATR target — 2:1 reward/risk ratio
            multiplier = 4.0
            if side == "buy":
                return entry_price + atr * multiplier
            return entry_price - atr * multiplier

        if side == "buy":
            return entry_price * (1 + self.take_profit_pct)
        return entry_price * (1 - self.take_profit_pct)

    def open_trade(self, trade_id: str, symbol: str, side: str,
                   entry_price: float, amount: float,
                   stop_loss: float = None, take_profit: float = None,
                   atr: float = None) -> Trade:
        sl = stop_loss or self.get_stop_loss(entry_price, side, atr)
        tp = take_profit or self.get_take_profit(entry_price, side, atr)

        trade = Trade(
            id=trade_id, symbol=symbol, side=side,
            entry_price=entry_price, amount=amount,
            stop_loss=sl, take_profit=tp,
            entry_candle=self._current_candle,
            peak_price=entry_price,
        )
        self.open_trades.append(trade)
        logger.info(
            f"OPEN {side.upper()} {amount:.6f} {symbol} @ {entry_price:.2f} "
            f"| SL={sl:.2f} TP={tp:.2f}"
        )
        return trade

    def close_trade(self, trade: Trade, exit_price: float, reason: str = "signal"):
        trade.exit_price = exit_price
        trade.exit_time = time.time()
        trade.status = "closed"

        if trade.side == "buy":
            trade.pnl = (exit_price - trade.entry_price) * trade.amount
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.amount

        self.capital += trade.pnl
        self.daily_pnl += trade.pnl
        self.update_capital(self.capital)

        if reason == "stop_loss":
            self._last_sl_candle = self._current_candle

        self.open_trades = [t for t in self.open_trades if t.id != trade.id]
        self.closed_trades.append(trade)

        logger.info(
            f"CLOSE {trade.side.upper()} {trade.symbol} @ {exit_price:.2f} "
            f"| PnL={trade.pnl:+.2f} | Reason={reason} "
            f"| Capital={self.capital:.2f}"
        )

    def can_sell_trade(self, trade: Trade) -> bool:
        """Check if trade has met minimum hold period."""
        return (self._current_candle - trade.entry_candle) >= self.min_hold_candles

    def check_stops(self, symbol: str, current_price: float) -> list[Trade]:
        """Check and close any trades that hit stop loss or take profit.

        Trailing stop: moves stop up as price rises, locks in profit.
        """
        trades_to_close = []

        for trade in self.open_trades:
            if trade.symbol != symbol:
                continue

            if trade.side == "buy":
                # Update peak price for trailing stop
                if current_price > trade.peak_price:
                    trade.peak_price = current_price

                # Trailing stop: once in profit, trail from peak
                if self.trailing_stop:
                    trail_stop = trade.peak_price * (1 - self.trailing_stop_pct)
                    if trail_stop > trade.stop_loss:
                        trade.stop_loss = trail_stop

                if current_price <= trade.stop_loss:
                    trades_to_close.append((trade, trade.stop_loss, "stop_loss"))
                elif current_price >= trade.take_profit:
                    trades_to_close.append((trade, trade.take_profit, "take_profit"))

        for trade, price, reason in trades_to_close:
            self.close_trade(trade, price, reason)

        return [t for t, _, _ in trades_to_close]

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_start_capital = self.capital
        if self.halted and "Daily" in self.halt_reason:
            self.halted = False
            self.halt_reason = ""
            logger.info("Daily halt reset")

    def get_stats(self) -> dict:
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            return {
                "total_trades": 0, "win_rate": 0, "total_pnl": 0,
                "avg_pnl": 0, "max_drawdown": 0, "capital": self.capital,
                "roi_pct": 0,
            }

        wins = [t for t in self.closed_trades if t.pnl and t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl and t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.closed_trades if t.pnl)

        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win * len(wins) / (avg_loss * len(losses))) if losses and avg_loss != 0 else float("inf")

        return {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / total_trades,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total_trades,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": self.get_drawdown(),
            "capital": self.capital,
            "roi_pct": (self.capital - self.initial_capital) / self.initial_capital * 100,
        }
