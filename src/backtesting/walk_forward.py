"""Walk-Forward Optimization — detects overfitting by testing out-of-sample.

Splits historical data into rolling windows:
  [train_1][test_1] [train_2][test_2] ... [train_n][test_n]

For each window:
  1. Backtest on train period (in-sample)
  2. Backtest on test period (out-of-sample)
  3. Compare metrics — if OOS degrades badly, the strategy is overfit

This is the single most important validation step before deploying a strategy.
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
from src.backtesting.engine import BacktestEngine
from src.strategies.base import BaseStrategy
from src.utils.logger import setup_logger

logger = setup_logger("walk_forward")


class WalkForwardValidator:
    def __init__(self, config: dict, strategy_cls: type,
                 train_days: int = 30, test_days: int = 10,
                 step_days: int = 10):
        self.config = config
        self.strategy_cls = strategy_cls
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days

    def _candles_per_day(self, timeframe: str) -> int:
        tf_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15,
            "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
        }
        minutes = tf_minutes.get(timeframe, 15)
        return 1440 // minutes

    def run(self, df: pd.DataFrame) -> dict:
        """Run walk-forward validation on historical data."""
        timeframe = self.config["trading"].get("timeframe", "15m")
        cpd = self._candles_per_day(timeframe)

        train_candles = self.train_days * cpd
        test_candles = self.test_days * cpd
        step_candles = self.step_days * cpd
        window_size = train_candles + test_candles

        total_candles = len(df)
        if total_candles < window_size:
            logger.warning(
                f"Not enough data for walk-forward: need {window_size} candles, "
                f"have {total_candles}"
            )
            return {"error": "insufficient_data"}

        results = []
        fold = 0

        i = 0
        while i + window_size <= total_candles:
            fold += 1
            train_df = df.iloc[i:i + train_candles]
            test_df = df.iloc[i + train_candles:i + window_size]

            logger.info(
                f"Fold {fold}: train {train_df.index[0]} to {train_df.index[-1]}, "
                f"test {test_df.index[0]} to {test_df.index[-1]}"
            )

            # In-sample backtest
            train_strategy = self.strategy_cls(self.config)
            train_engine = BacktestEngine(self.config, train_strategy)
            train_result = train_engine.run(train_df)

            # Out-of-sample backtest
            test_strategy = self.strategy_cls(self.config)
            test_engine = BacktestEngine(self.config, test_strategy)
            test_result = test_engine.run(test_df)

            results.append({
                "fold": fold,
                "train_start": str(train_df.index[0]),
                "train_end": str(train_df.index[-1]),
                "test_start": str(test_df.index[0]),
                "test_end": str(test_df.index[-1]),
                "train_roi": train_result.get("roi_pct", 0),
                "test_roi": test_result.get("roi_pct", 0),
                "train_sharpe": train_result.get("sharpe_ratio", 0),
                "test_sharpe": test_result.get("sharpe_ratio", 0),
                "train_sortino": train_result.get("sortino_ratio", 0),
                "test_sortino": test_result.get("sortino_ratio", 0),
                "train_win_rate": train_result.get("win_rate", 0),
                "test_win_rate": test_result.get("win_rate", 0),
                "train_trades": train_result.get("total_trades", 0),
                "test_trades": test_result.get("total_trades", 0),
                "train_max_dd": train_result.get("max_drawdown_pct", 0),
                "test_max_dd": test_result.get("max_drawdown_pct", 0),
            })

            i += step_candles

        return self._generate_report(results)

    def _generate_report(self, results: list[dict]) -> dict:
        if not results:
            return {"error": "no_results"}

        n = len(results)
        avg_train_roi = np.mean([r["train_roi"] for r in results])
        avg_test_roi = np.mean([r["test_roi"] for r in results])
        avg_train_sharpe = np.mean([r["train_sharpe"] for r in results])
        avg_test_sharpe = np.mean([r["test_sharpe"] for r in results])
        avg_train_sortino = np.mean([r["train_sortino"] for r in results])
        avg_test_sortino = np.mean([r["test_sortino"] for r in results])

        # Overfitting ratio: if OOS performance is much worse than IS
        roi_ratio = avg_test_roi / avg_train_roi if avg_train_roi != 0 else 0
        sharpe_ratio_oos = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe != 0 else 0

        # A ratio near 1.0 means consistent; <0.5 means likely overfit
        overfit_risk = "LOW" if roi_ratio > 0.6 else ("MEDIUM" if roi_ratio > 0.3 else "HIGH")
        if avg_train_roi <= 0 and avg_test_roi <= 0:
            overfit_risk = "N/A (both negative)"

        profitable_oos_folds = sum(1 for r in results if r["test_roi"] > 0)

        print("\n" + "=" * 70)
        print("  WALK-FORWARD VALIDATION REPORT")
        print("=" * 70)

        # Per-fold table
        headers = ["Fold", "Train ROI%", "Test ROI%", "Train Sharpe", "Test Sharpe", "Test Trades"]
        rows = []
        for r in results:
            rows.append([
                r["fold"],
                f"{r['train_roi']:+.2f}%",
                f"{r['test_roi']:+.2f}%",
                f"{r['train_sharpe']:.2f}",
                f"{r['test_sharpe']:.2f}",
                r["test_trades"],
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Summary
        summary = [
            ["Total Folds", n],
            ["Avg Train ROI", f"{avg_train_roi:+.2f}%"],
            ["Avg Test ROI (OOS)", f"{avg_test_roi:+.2f}%"],
            ["OOS/IS ROI Ratio", f"{roi_ratio:.2f}"],
            ["Avg Train Sharpe", f"{avg_train_sharpe:.2f}"],
            ["Avg Test Sharpe (OOS)", f"{avg_test_sharpe:.2f}"],
            ["Avg Train Sortino", f"{avg_train_sortino:.2f}"],
            ["Avg Test Sortino (OOS)", f"{avg_test_sortino:.2f}"],
            ["Profitable OOS Folds", f"{profitable_oos_folds}/{n}"],
            ["Overfitting Risk", overfit_risk],
        ]
        print("\n" + tabulate(summary, tablefmt="simple"))
        print("=" * 70 + "\n")

        report = {
            "folds": n,
            "results": results,
            "avg_train_roi": avg_train_roi,
            "avg_test_roi": avg_test_roi,
            "avg_train_sharpe": avg_train_sharpe,
            "avg_test_sharpe": avg_test_sharpe,
            "avg_train_sortino": avg_train_sortino,
            "avg_test_sortino": avg_test_sortino,
            "oos_is_roi_ratio": roi_ratio,
            "oos_is_sharpe_ratio": sharpe_ratio_oos,
            "profitable_oos_folds": profitable_oos_folds,
            "overfit_risk": overfit_risk,
        }

        logger.info(
            f"Walk-forward complete: {n} folds | "
            f"OOS ROI={avg_test_roi:+.2f}% | "
            f"OOS/IS ratio={roi_ratio:.2f} | "
            f"Overfit risk={overfit_risk}"
        )

        return report
