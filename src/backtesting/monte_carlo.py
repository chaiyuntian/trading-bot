"""Monte Carlo Simulation for Robustness Testing.

Answers: "Is this strategy's profit due to skill or luck?"

Method:
1. Take the list of actual trade PnLs from a backtest
2. Randomly shuffle the trade order N times (default 1000)
3. Rebuild equity curves from shuffled trades
4. Calculate confidence intervals for key metrics

If 95% of shuffled simulations are profitable, the strategy is robust.
If only 50% are profitable, it's probably luck.
"""

import numpy as np
from tabulate import tabulate
from src.utils.logger import setup_logger

logger = setup_logger("monte_carlo")


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy robustness."""

    def __init__(self, initial_capital: float = 100, n_simulations: int = 1000):
        self.initial_capital = initial_capital
        self.n_simulations = n_simulations

    def run(self, trade_pnls: list[float]) -> dict:
        """Run Monte Carlo simulation on a list of trade PnLs.

        Args:
            trade_pnls: List of profit/loss values from each trade.

        Returns:
            Report with confidence intervals.
        """
        if len(trade_pnls) < 5:
            return {
                "valid": False,
                "reason": "Need at least 5 trades for Monte Carlo",
            }

        pnls = np.array(trade_pnls)
        n_trades = len(pnls)

        logger.info(
            f"Monte Carlo: {self.n_simulations} simulations x {n_trades} trades"
        )

        # Run simulations
        final_capitals = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        sharpe_ratios = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            # Shuffle trade order
            shuffled = np.random.permutation(pnls)

            # Build equity curve
            equity = np.cumsum(shuffled) + self.initial_capital
            final_capitals[i] = equity[-1]

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_drawdowns[i] = np.max(dd)

            # Simplified Sharpe
            returns = shuffled / self.initial_capital
            std = np.std(returns)
            if std > 0:
                sharpe_ratios[i] = np.mean(returns) / std * np.sqrt(n_trades)
            else:
                sharpe_ratios[i] = 0

        # Calculate percentiles
        profitable_pct = np.mean(final_capitals > self.initial_capital)
        ruin_pct = np.mean(final_capitals < self.initial_capital * 0.5)  # Lost 50%+

        report = {
            "valid": True,
            "n_simulations": self.n_simulations,
            "n_trades": n_trades,
            "original_total_pnl": float(np.sum(pnls)),
            "original_final_capital": float(np.sum(pnls) + self.initial_capital),
            # Confidence intervals
            "final_capital_5th": float(np.percentile(final_capitals, 5)),
            "final_capital_25th": float(np.percentile(final_capitals, 25)),
            "final_capital_median": float(np.percentile(final_capitals, 50)),
            "final_capital_75th": float(np.percentile(final_capitals, 75)),
            "final_capital_95th": float(np.percentile(final_capitals, 95)),
            # Risk metrics
            "profitable_pct": profitable_pct,
            "ruin_risk_pct": ruin_pct,
            "median_max_drawdown": float(np.percentile(max_drawdowns, 50)),
            "worst_case_drawdown_95": float(np.percentile(max_drawdowns, 95)),
            # Sharpe distribution
            "sharpe_5th": float(np.percentile(sharpe_ratios, 5)),
            "sharpe_median": float(np.percentile(sharpe_ratios, 50)),
            "sharpe_95th": float(np.percentile(sharpe_ratios, 95)),
            # Verdict
            "robust": profitable_pct >= 0.70 and ruin_pct < 0.05,
        }

        self._print_report(report)
        return report

    def _print_report(self, report: dict):
        print("\n" + "=" * 60)
        print("  MONTE CARLO ROBUSTNESS REPORT")
        print("=" * 60)

        table = [
            ["Simulations", f"{report['n_simulations']:,}"],
            ["Trades per Sim", f"{report['n_trades']}"],
            ["Original PnL", f"${report['original_total_pnl']:+.2f}"],
            ["", ""],
            ["--- Profit Confidence ---", ""],
            ["Profitable Simulations", f"{report['profitable_pct']:.0%}"],
            ["Ruin Risk (50%+ loss)", f"{report['ruin_risk_pct']:.1%}"],
            ["", ""],
            ["--- Final Capital Distribution ---", ""],
            ["5th Percentile (worst case)", f"${report['final_capital_5th']:.2f}"],
            ["25th Percentile", f"${report['final_capital_25th']:.2f}"],
            ["Median", f"${report['final_capital_median']:.2f}"],
            ["75th Percentile", f"${report['final_capital_75th']:.2f}"],
            ["95th Percentile (best case)", f"${report['final_capital_95th']:.2f}"],
            ["", ""],
            ["--- Risk ---", ""],
            ["Median Max Drawdown", f"{report['median_max_drawdown']:.1%}"],
            ["95th Pctile Drawdown", f"{report['worst_case_drawdown_95']:.1%}"],
            ["", ""],
            ["--- Sharpe Distribution ---", ""],
            ["5th Percentile", f"{report['sharpe_5th']:.2f}"],
            ["Median", f"{report['sharpe_median']:.2f}"],
            ["95th Percentile", f"{report['sharpe_95th']:.2f}"],
        ]
        print(tabulate(table, tablefmt="simple"))

        print()
        if report["robust"]:
            print("  VERDICT: ROBUST -- Strategy profits are NOT due to luck.")
            print(f"  {report['profitable_pct']:.0%} of random trade orderings are profitable.")
        else:
            print("  VERDICT: FRAGILE -- Strategy may depend on trade order (luck).")
            if report["profitable_pct"] < 0.7:
                print(f"  Only {report['profitable_pct']:.0%} of simulations profitable (need 70%+).")
            if report["ruin_risk_pct"] >= 0.05:
                print(f"  Ruin risk is {report['ruin_risk_pct']:.1%} (need <5%).")
        print("=" * 60 + "\n")
