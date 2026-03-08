"""Walk-Forward Validation Engine.

The GOLD STANDARD for strategy validation. Prevents overfitting by:
1. Split data into N folds (e.g., 6 months train, 1 month test)
2. Train/optimize on each fold, test on the NEXT unseen fold
3. Only strategies that profit on unseen data are considered valid

If a strategy profits in backtest but fails walk-forward, it's OVERFITTED.

Usage:
    python -m src --walk-forward
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
from src.backtesting.engine import BacktestEngine
from src.strategies.base import BaseStrategy
from src.utils.logger import setup_logger

logger = setup_logger("walk_forward")


class WalkForwardValidator:
    """Walk-forward validation to detect overfitting."""

    def __init__(self, config: dict, strategy_class: type,
                 train_periods: int = 6, test_periods: int = 1,
                 n_folds: int = 6):
        """
        Args:
            train_periods: Number of months for training window
            test_periods: Number of months for out-of-sample testing
            n_folds: Number of walk-forward folds
        """
        self.config = config
        self.strategy_class = strategy_class
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.n_folds = n_folds
        self.fold_results: list[dict] = []

    def run(self, df: pd.DataFrame) -> dict:
        """Run walk-forward validation on historical data."""
        total_candles = len(df)
        fold_size = total_candles // (self.n_folds + self.train_periods)
        train_size = fold_size * self.train_periods
        test_size = fold_size * self.test_periods

        if train_size + test_size > total_candles:
            logger.warning("Not enough data for walk-forward. Using 70/30 split.")
            train_size = int(total_candles * 0.7)
            test_size = total_candles - train_size
            self.n_folds = 1

        logger.info(
            f"Walk-Forward: {self.n_folds} folds | "
            f"train={train_size} candles | test={test_size} candles"
        )

        in_sample_results = []
        out_of_sample_results = []

        for fold in range(self.n_folds):
            offset = fold * test_size
            train_start = offset
            train_end = offset + train_size
            test_start = train_end
            test_end = min(test_start + test_size, total_candles)

            if test_end > total_candles or test_start >= total_candles:
                break

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            if len(train_df) < 100 or len(test_df) < 30:
                continue

            # ── In-sample (training) backtest ──
            strategy_is = self.strategy_class(self.config)
            engine_is = BacktestEngine(self.config, strategy_is)
            is_result = engine_is.run(train_df)

            # ── Out-of-sample (test) backtest ──
            strategy_oos = self.strategy_class(self.config)
            engine_oos = BacktestEngine(self.config, strategy_oos)
            oos_result = engine_oos.run(test_df)

            fold_data = {
                "fold": fold + 1,
                "train_period": f"{train_df.index[0].date()} to {train_df.index[-1].date()}",
                "test_period": f"{test_df.index[0].date()} to {test_df.index[-1].date()}",
                "is_roi": is_result["roi_pct"],
                "is_sharpe": is_result["sharpe_ratio"],
                "is_trades": is_result["total_trades"],
                "is_win_rate": is_result["win_rate"],
                "oos_roi": oos_result["roi_pct"],
                "oos_sharpe": oos_result["sharpe_ratio"],
                "oos_trades": oos_result["total_trades"],
                "oos_win_rate": oos_result["win_rate"],
                "oos_max_dd": oos_result["max_drawdown_pct"],
            }
            self.fold_results.append(fold_data)
            in_sample_results.append(is_result)
            out_of_sample_results.append(oos_result)

            logger.info(
                f"Fold {fold+1}: IS_ROI={is_result['roi_pct']:+.2f}% "
                f"OOS_ROI={oos_result['roi_pct']:+.2f}% "
                f"OOS_Sharpe={oos_result['sharpe_ratio']:.2f}"
            )

        return self._generate_report(in_sample_results, out_of_sample_results)

    def _generate_report(self, is_results: list, oos_results: list) -> dict:
        if not oos_results:
            return {"valid": False, "reason": "No folds completed"}

        # Aggregate out-of-sample metrics
        oos_rois = [r["roi_pct"] for r in oos_results]
        oos_sharpes = [r["sharpe_ratio"] for r in oos_results]
        oos_win_rates = [r["win_rate"] for r in oos_results]
        oos_max_dds = [r["max_drawdown_pct"] for r in oos_results]
        is_rois = [r["roi_pct"] for r in is_results]

        # Key validation metrics
        profitable_folds = sum(1 for r in oos_rois if r > 0)
        consistency_ratio = profitable_folds / len(oos_rois)

        # Efficiency ratio: how much of in-sample performance survives OOS
        avg_is_roi = np.mean(is_rois) if is_rois else 0
        avg_oos_roi = np.mean(oos_rois)
        efficiency = avg_oos_roi / avg_is_roi if avg_is_roi != 0 else 0

        # OVERFIT DETECTION
        # If IS performance is great but OOS is negative = overfitted
        is_overfitted = avg_is_roi > 5 and avg_oos_roi < 0
        is_consistent = consistency_ratio >= 0.5
        is_profitable = avg_oos_roi > 0
        avg_oos_sharpe = np.mean(oos_sharpes)

        verdict = "PASS" if (is_profitable and is_consistent and not is_overfitted) else "FAIL"

        report = {
            "valid": verdict == "PASS",
            "verdict": verdict,
            "n_folds": len(oos_results),
            "profitable_folds": profitable_folds,
            "consistency_ratio": consistency_ratio,
            "avg_is_roi": avg_is_roi,
            "avg_oos_roi": avg_oos_roi,
            "efficiency": efficiency,
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_oos_win_rate": np.mean(oos_win_rates),
            "avg_oos_max_dd": np.mean(oos_max_dds),
            "worst_oos_roi": min(oos_rois),
            "best_oos_roi": max(oos_rois),
            "is_overfitted": is_overfitted,
            "fold_details": self.fold_results,
        }

        self._print_report(report)
        return report

    def _print_report(self, report: dict):
        print("\n" + "=" * 70)
        print(f"  WALK-FORWARD VALIDATION REPORT")
        print("=" * 70)

        # Fold details table
        if report["fold_details"]:
            headers = ["Fold", "Test Period", "IS ROI%", "OOS ROI%", "OOS Sharpe", "OOS WR", "OOS MaxDD"]
            rows = []
            for f in report["fold_details"]:
                rows.append([
                    f["fold"], f["test_period"],
                    f"{f['is_roi']:+.2f}%", f"{f['oos_roi']:+.2f}%",
                    f"{f['oos_sharpe']:.2f}", f"{f['oos_win_rate']:.0%}",
                    f"{f['oos_max_dd']:.1f}%",
                ])
            print(tabulate(rows, headers=headers, tablefmt="grid"))

        print()
        summary = [
            ["Verdict", f"{'PASS' if report['valid'] else 'FAIL -- DO NOT DEPLOY'}"],
            ["Folds", f"{report['n_folds']}"],
            ["Profitable Folds", f"{report['profitable_folds']}/{report['n_folds']} ({report['consistency_ratio']:.0%})"],
            ["Avg In-Sample ROI", f"{report['avg_is_roi']:+.2f}%"],
            ["Avg Out-of-Sample ROI", f"{report['avg_oos_roi']:+.2f}%"],
            ["Efficiency (OOS/IS)", f"{report['efficiency']:.1%}"],
            ["Avg OOS Sharpe", f"{report['avg_oos_sharpe']:.2f}"],
            ["Avg OOS Win Rate", f"{report['avg_oos_win_rate']:.0%}"],
            ["Avg OOS Max Drawdown", f"{report['avg_oos_max_dd']:.1f}%"],
            ["Best OOS Fold", f"{report['best_oos_roi']:+.2f}%"],
            ["Worst OOS Fold", f"{report['worst_oos_roi']:+.2f}%"],
            ["Overfitted?", f"{'YES -- DANGER' if report['is_overfitted'] else 'No'}"],
        ]
        print(tabulate(summary, tablefmt="simple"))

        if not report["valid"]:
            print("\n  STRATEGY FAILED WALK-FORWARD VALIDATION!")
            print("  This strategy is likely overfitted to historical data.")
            print("  DO NOT deploy with real money until it passes.\n")
        else:
            print("\n  Strategy passed walk-forward validation.")
            print("  Proceed to paper trading for 2+ weeks before live deployment.\n")

        print("=" * 70 + "\n")
