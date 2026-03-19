"""Simulation Loop — runs paper trading on historical data and exports metrics
for the evolver (or any agent) to analyze and improve strategy parameters.

Usage:
  python -m src.sim_loop                  # Run simulation, export metrics
  python -m src.sim_loop --cycles 500     # Custom cycle count
  python -m src.sim_loop --evolve         # Run simulation + trigger evolver

The loop:
  1. Fetches historical data from Binance (no API key needed for public data)
  2. Simulates paper trading cycle-by-cycle through the data
  3. Exports performance metrics to data/sim_results.json
  4. Optionally triggers the evolver to analyze and suggest improvements

The exported metrics file is designed to be consumed by:
  - The evolver (.evolver/): reads sim_results.json to find weak spots
  - Claude/AI agents: reads metrics to suggest parameter changes
  - Human review: clear JSON format for manual analysis
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.backtesting.walk_forward import WalkForwardValidator
from src.bot import STRATEGY_MAP
from src.utils.logger import setup_logger

logger = setup_logger("sim_loop")

DEFAULT_CONFIG = {
    "trading": {
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "initial_capital": 1000,
        "mode": "paper",
    },
    "strategy": {
        "name": "ensemble",
        "params": {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ema_period": 50,
            "grid_levels": 10,
            "grid_spacing_atr_mult": 0.5,
            "dca_interval": 4,
            "ensemble_min_consensus": 0.4,
            "kama_period": 10,
            "er_threshold": 0.3,
        },
    },
    "risk": {
        "max_position_pct": 0.30,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "max_daily_loss_pct": 0.05,
        "max_drawdown_pct": 0.25,
        "risk_per_trade_pct": 0.02,
        "max_open_trades": 3,
        "trailing_stop": False,
        "trailing_stop_pct": 0.02,
    },
}


def fetch_data(config: dict) -> pd.DataFrame:
    """Fetch historical data for simulation."""
    import ccxt

    symbol = config["trading"]["symbol"]
    timeframe = config["trading"].get("timeframe", "15m")

    logger.info(f"Fetching data for {symbol} ({timeframe})...")

    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()

    all_data = []
    since = exchange.parse8601("2024-01-01T00:00:00Z")

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    logger.info(f"Loaded {len(df)} candles: {df.index[0]} to {df.index[-1]}")
    return df


def run_simulation(config: dict, df: pd.DataFrame) -> dict:
    """Run backtest for all strategies and collect comparative metrics."""
    results = {}

    for name, cls in STRATEGY_MAP.items():
        logger.info(f"Simulating strategy: {name}")
        strategy = cls(config)
        engine = BacktestEngine(config, strategy)
        result = engine.run(df)
        results[name] = result

    return results


def run_walk_forward_check(config: dict, df: pd.DataFrame, strategy_name: str) -> dict:
    """Run walk-forward validation for a specific strategy."""
    if strategy_name not in STRATEGY_MAP:
        return {"error": f"Unknown strategy: {strategy_name}"}

    validator = WalkForwardValidator(
        config, STRATEGY_MAP[strategy_name],
        train_days=30, test_days=10, step_days=10
    )
    return validator.run(df)


def generate_improvement_report(results: dict, wf_result: dict = None) -> dict:
    """Analyze simulation results and generate an improvement report.

    This report is designed to be consumed by an AI agent (evolver, Claude, etc.)
    to identify weaknesses and suggest parameter changes.
    """
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "strategy_comparison": {},
        "best_strategy": None,
        "weaknesses": [],
        "suggestions": [],
    }

    # Compare strategies
    best_name = None
    best_sortino = float("-inf")

    for name, r in results.items():
        entry = {
            "roi_pct": r.get("roi_pct", 0),
            "sharpe_ratio": r.get("sharpe_ratio", 0),
            "sortino_ratio": r.get("sortino_ratio", 0),
            "win_rate": r.get("win_rate", 0),
            "total_trades": r.get("total_trades", 0),
            "max_drawdown_pct": r.get("max_drawdown_pct", 0),
            "profit_factor": r.get("profit_factor", 0),
            "total_fees": r.get("total_fees", 0),
        }
        report["strategy_comparison"][name] = entry

        sortino = r.get("sortino_ratio", 0)
        if sortino > best_sortino:
            best_sortino = sortino
            best_name = name

    report["best_strategy"] = best_name

    # Identify weaknesses
    for name, r in results.items():
        if r.get("win_rate", 0) < 0.45:
            report["weaknesses"].append(
                f"{name}: low win rate ({r['win_rate']:.0%}). "
                f"Consider adjusting signal thresholds."
            )
        if r.get("max_drawdown_pct", 0) > 15:
            report["weaknesses"].append(
                f"{name}: high drawdown ({r['max_drawdown_pct']:.1f}%). "
                f"Consider tighter stop-loss or smaller position size."
            )
        if r.get("total_trades", 0) < 5:
            report["weaknesses"].append(
                f"{name}: too few trades ({r['total_trades']}). "
                f"Consider relaxing entry conditions."
            )
        fee_pct = r.get("total_fees", 0) / max(r.get("capital", 1), 1) * 100
        if fee_pct > 5:
            report["weaknesses"].append(
                f"{name}: fees ate {fee_pct:.1f}% of capital. "
                f"Consider reducing trade frequency or using limit orders."
            )

    # Walk-forward overfit check
    if wf_result and "overfit_risk" in wf_result:
        if wf_result["overfit_risk"] == "HIGH":
            report["weaknesses"].append(
                "Walk-forward shows HIGH overfitting risk. "
                "Strategy parameters may be curve-fit to historical data."
            )
            report["suggestions"].append(
                "Run walk-forward optimization with shorter training windows "
                "or use more conservative parameters."
            )

    # Generate suggestions
    if best_name:
        report["suggestions"].append(
            f"Best performing strategy: {best_name} "
            f"(Sortino={best_sortino:.2f}). Consider using this as primary."
        )

    for name, r in results.items():
        if r.get("roi_pct", 0) < 0:
            report["suggestions"].append(
                f"{name} is losing money (ROI={r['roi_pct']:+.2f}%). "
                f"Parameters need adjustment or this strategy should be "
                f"disabled in the ensemble for current market conditions."
            )

    return report


def export_results(results: dict, report: dict, output_dir: str = "data"):
    """Export results to JSON for consumption by evolver or agents."""
    os.makedirs(output_dir, exist_ok=True)

    # Full results
    results_path = os.path.join(output_dir, "sim_results.json")
    serializable = {}
    for name, r in results.items():
        serializable[name] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                              for k, v in r.items()}

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results exported to {results_path}")

    # Improvement report
    report_path = os.path.join(output_dir, "improvement_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Improvement report exported to {report_path}")

    return results_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Trading Bot Simulation Loop")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--strategy", default="ensemble", help="Strategy to focus on")
    parser.add_argument("--walk-forward", action="store_true", help="Include walk-forward validation")
    parser.add_argument("--evolve", action="store_true", help="Trigger evolver after simulation")
    parser.add_argument("--output", default="data", help="Output directory for results")
    args = parser.parse_args()

    # Load config
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    config["strategy"]["name"] = args.strategy

    print("\n" + "=" * 60)
    print("  SIMULATION LOOP")
    print(f"  Strategy: {args.strategy}")
    print(f"  Symbol: {config['trading']['symbol']}")
    print("=" * 60 + "\n")

    # 1. Fetch data
    df = fetch_data(config)

    # 2. Run all strategies
    results = run_simulation(config, df)

    # 3. Walk-forward validation (optional)
    wf_result = None
    if args.walk_forward:
        print("\nRunning walk-forward validation...")
        wf_result = run_walk_forward_check(config, df, args.strategy)

    # 4. Generate improvement report
    report = generate_improvement_report(results, wf_result)

    # 5. Export results
    results_path, report_path = export_results(results, report, args.output)

    # 6. Print summary
    print("\n" + "=" * 60)
    print("  SIMULATION SUMMARY")
    print("=" * 60)
    print(f"\n  Best strategy: {report['best_strategy']}")
    if report["weaknesses"]:
        print("\n  Weaknesses found:")
        for w in report["weaknesses"]:
            print(f"    - {w}")
    if report["suggestions"]:
        print("\n  Suggestions:")
        for s in report["suggestions"]:
            print(f"    - {s}")
    print(f"\n  Results: {results_path}")
    print(f"  Report:  {report_path}")
    print("=" * 60 + "\n")

    # 7. Trigger evolver (optional)
    if args.evolve:
        evolver_path = os.path.join(os.path.dirname(__file__), "..", ".evolver")
        if os.path.exists(os.path.join(evolver_path, "index.js")):
            print("Triggering evolver for self-improvement...")
            os.system(f"cd {evolver_path} && node index.js")
        else:
            print("Evolver not found at .evolver/. Skipping.")
            print("To use the evolver, install it per the ROADMAP instructions.")

    return 0


if __name__ == "__main__":
    exit(main())
