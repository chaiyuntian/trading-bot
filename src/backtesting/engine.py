import pandas as pd
import numpy as np
from tabulate import tabulate
from src.risk.manager import RiskManager
from src.strategies.base import BaseStrategy
from src.utils.logger import setup_logger

logger = setup_logger("backtest")


class BacktestEngine:
    """
    Backtesting engine that simulates trading on historical data.
    Accounts for fees, slippage, and realistic execution.
    """

    def __init__(self, config: dict, strategy: BaseStrategy):
        self.config = config
        self.strategy = strategy
        self.fee_rate = 0.001  # 0.1% per trade (Binance base)
        self.slippage = 0.0005  # 0.05% slippage estimate
        self.risk_manager = RiskManager(config)
        self.equity_curve: list[float] = []
        self.trades_log: list[dict] = []

    def run(self, df: pd.DataFrame) -> dict:
        """Run backtest on historical OHLCV data."""
        logger.info(
            f"Starting backtest: {self.strategy.name()} | "
            f"Capital=${self.risk_manager.initial_capital} | "
            f"{len(df)} candles"
        )

        lookback = 60  # Minimum candles for indicator calculation
        trade_counter = 0

        for i in range(lookback, len(df)):
            window = df.iloc[:i + 1].copy()
            current_price = float(window.iloc[-1]["close"])

            # Track equity
            open_pnl = sum(
                (current_price - t.entry_price) * t.amount
                if t.side == "buy" else
                (t.entry_price - current_price) * t.amount
                for t in self.risk_manager.open_trades
            )
            self.equity_curve.append(self.risk_manager.capital + open_pnl)

            # Check stops on open trades
            self.risk_manager.check_stops(
                self.config["trading"]["symbol"], current_price
            )

            # Can we trade?
            can_trade, reason = self.risk_manager.can_open_trade()
            if not can_trade and not self.risk_manager.open_trades:
                continue

            # Generate signal
            signal = self.strategy.generate_signal(window)

            if signal == "buy" and can_trade:
                stop_price = self.risk_manager.get_stop_loss(current_price, "buy")
                amount = self.risk_manager.calculate_position_size(
                    current_price, stop_price
                )

                if amount * current_price < 5:  # Min order ~$5
                    continue

                # Apply fees + slippage
                exec_price = current_price * (1 + self.slippage)
                fee = amount * exec_price * self.fee_rate
                self.risk_manager.capital -= fee

                trade_counter += 1
                self.risk_manager.open_trade(
                    f"bt_{trade_counter}",
                    self.config["trading"]["symbol"],
                    "buy", exec_price, amount
                )
                self.trades_log.append({
                    "id": trade_counter,
                    "time": window.index[-1],
                    "side": "buy",
                    "price": exec_price,
                    "amount": amount,
                    "fee": fee
                })

            elif signal == "sell" and self.risk_manager.open_trades:
                for trade in list(self.risk_manager.open_trades):
                    exec_price = current_price * (1 - self.slippage)
                    fee = trade.amount * exec_price * self.fee_rate
                    self.risk_manager.capital -= fee
                    self.risk_manager.close_trade(trade, exec_price, "signal")
                    self.trades_log.append({
                        "id": trade_counter,
                        "time": window.index[-1],
                        "side": "sell",
                        "price": exec_price,
                        "amount": trade.amount,
                        "fee": fee
                    })

        # Close any remaining open trades at last price
        last_price = float(df.iloc[-1]["close"])
        for trade in list(self.risk_manager.open_trades):
            self.risk_manager.close_trade(trade, last_price, "backtest_end")

        return self._generate_report(df)

    def _generate_report(self, df: pd.DataFrame) -> dict:
        stats = self.risk_manager.get_stats()

        # Calculate additional metrics
        equity = np.array(self.equity_curve) if self.equity_curve else np.array([self.risk_manager.initial_capital])
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / peak
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        # Sharpe ratio (annualized, assuming 15m candles)
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(365 * 24 * 4) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        total_fees = sum(t.get("fee", 0) for t in self.trades_log)

        report = {
            **stats,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "total_fees": total_fees,
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "candles": len(df),
        }

        self._print_report(report)
        return report

    def _print_report(self, report: dict):
        print("\n" + "=" * 60)
        print(f"  BACKTEST REPORT: {self.strategy.name()}")
        print("=" * 60)

        table = [
            ["Period", f"{report['start_date']} to {report['end_date']}"],
            ["Candles", f"{report['candles']}"],
            ["Initial Capital", f"${self.risk_manager.initial_capital:.2f}"],
            ["Final Capital", f"${report['capital']:.2f}"],
            ["Total PnL", f"${report['total_pnl']:+.2f}"],
            ["ROI", f"{report['roi_pct']:+.2f}%"],
            ["Total Trades", f"{report['total_trades']}"],
            ["Wins / Losses", f"{report.get('wins', 0)} / {report.get('losses', 0)}"],
            ["Win Rate", f"{report['win_rate']:.1%}"],
            ["Avg Win", f"${report.get('avg_win', 0):+.2f}"],
            ["Avg Loss", f"${report.get('avg_loss', 0):+.2f}"],
            ["Profit Factor", f"{report.get('profit_factor', 0):.2f}"],
            ["Max Drawdown", f"{report['max_drawdown_pct']:.2f}%"],
            ["Sharpe Ratio", f"{report['sharpe_ratio']:.2f}"],
            ["Total Fees", f"${report['total_fees']:.2f}"],
        ]
        print(tabulate(table, tablefmt="simple"))
        print("=" * 60 + "\n")
