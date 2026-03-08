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
        self.trailing_stop = risk_cfg.get("trailing_stop", False)
        self.trailing_stop_pct = risk_cfg.get("trailing_stop_pct", 0.02)

        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.daily_pnl = 0.0
        self.daily_start_capital = self.capital
        self.halted = False
        self.halt_reason = ""

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

        drawdown = self.get_drawdown()
        if drawdown >= self.max_drawdown_pct:
            self.halted = True
            self.halt_reason = f"Max drawdown {drawdown:.1%} >= {self.max_drawdown_pct:.1%}"
            logger.critical(self.halt_reason)
            return False, self.halt_reason

        daily_loss = (self.daily_start_capital - self.capital) / self.daily_start_capital
        if daily_loss >= self.max_daily_loss_pct:
            self.halted = True
            self.halt_reason = f"Daily loss {daily_loss:.1%} >= {self.max_daily_loss_pct:.1%}"
            logger.critical(self.halt_reason)
            return False, self.halt_reason

        return True, "OK"

    def calculate_position_size(self, entry_price: float, stop_price: float) -> float:
        """Position size based on risk per trade and stop distance."""
        risk_amount = self.capital * self.risk_per_trade_pct
        stop_distance = abs(entry_price - stop_price) / entry_price

        if stop_distance == 0:
            return 0

        position_value = risk_amount / stop_distance
        max_position = self.capital * self.max_position_pct

        position_value = min(position_value, max_position)
        amount = position_value / entry_price
        return amount

    def get_stop_loss(self, entry_price: float, side: str = "buy") -> float:
        if side == "buy":
            return entry_price * (1 - self.stop_loss_pct)
        return entry_price * (1 + self.stop_loss_pct)

    def get_take_profit(self, entry_price: float, side: str = "buy") -> float:
        if side == "buy":
            return entry_price * (1 + self.take_profit_pct)
        return entry_price * (1 - self.take_profit_pct)

    def open_trade(self, trade_id: str, symbol: str, side: str,
                   entry_price: float, amount: float) -> Trade:
        stop_loss = self.get_stop_loss(entry_price, side)
        take_profit = self.get_take_profit(entry_price, side)

        trade = Trade(
            id=trade_id, symbol=symbol, side=side,
            entry_price=entry_price, amount=amount,
            stop_loss=stop_loss, take_profit=take_profit
        )
        self.open_trades.append(trade)
        logger.info(
            f"OPEN {side.upper()} {amount:.6f} {symbol} @ {entry_price:.2f} "
            f"| SL={stop_loss:.2f} TP={take_profit:.2f}"
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

        self.open_trades = [t for t in self.open_trades if t.id != trade.id]
        self.closed_trades.append(trade)

        logger.info(
            f"CLOSE {trade.side.upper()} {trade.symbol} @ {exit_price:.2f} "
            f"| PnL={trade.pnl:+.2f} | Reason={reason} "
            f"| Capital={self.capital:.2f}"
        )

    def check_stops(self, symbol: str, current_price: float) -> list[Trade]:
        """Check and close any trades that hit stop loss or take profit."""
        trades_to_close = []

        for trade in self.open_trades:
            if trade.symbol != symbol:
                continue

            if trade.side == "buy":
                if self.trailing_stop:
                    new_stop = current_price * (1 - self.trailing_stop_pct)
                    if new_stop > trade.stop_loss:
                        trade.stop_loss = new_stop

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
