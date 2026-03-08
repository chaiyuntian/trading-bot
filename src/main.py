"""
Crypto Trading Bot - Main Entry Point
======================================
Usage:
  python -m src                      # Run bot (paper trading)
  python -m src --backtest           # Run backtest
  python -m src --backtest-all       # Backtest all strategies
  python -m src --walk-forward       # Walk-forward validation (anti-overfit)
  python -m src --monte-carlo        # Monte Carlo robustness test
  python -m src --validate           # Full validation suite (WF + MC)
  python -m src --config path        # Custom config file
  python -m src --strategy rsi_macd  # Override strategy
"""

import argparse
import os
import sys
import yaml
from src.bot import TradingBot, STRATEGY_MAP
from src.backtesting.engine import BacktestEngine
from src.backtesting.walk_forward import WalkForwardValidator
from src.backtesting.monte_carlo import MonteCarloSimulator
from src.utils.logger import setup_logger


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Config file not found: {path}")
        print("Copy config/config.example.yaml to config/config.yaml and edit it.")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fetch_backtest_data(config: dict):
    """Fetch historical data for backtesting."""
    import ccxt
    import pandas as pd

    symbol = config["trading"]["symbol"]
    timeframe = config["trading"].get("timeframe", "15m")

    print(f"Fetching historical data for {symbol} ({timeframe})...")

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

    print(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def run_backtest(config: dict, strategy_name: str = None):
    name = strategy_name or config["strategy"]["name"]
    if name not in STRATEGY_MAP:
        print(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
        return None

    strategy = STRATEGY_MAP[name](config)
    df = fetch_backtest_data(config)
    engine = BacktestEngine(config, strategy)
    return engine.run(df)


def run_all_backtests(config: dict):
    from tabulate import tabulate

    results = {}
    df = fetch_backtest_data(config)

    for name, cls in STRATEGY_MAP.items():
        print(f"\n{'='*60}")
        print(f"  Backtesting: {name}")
        print(f"{'='*60}")

        strategy = cls(config)
        engine = BacktestEngine(config, strategy)
        results[name] = engine.run(df)

    # Comparison table
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)

    headers = ["Strategy", "ROI%", "Trades", "Win Rate", "PF", "MaxDD%", "Sharpe"]
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            f"{r['roi_pct']:+.2f}%",
            r["total_trades"],
            f"{r['win_rate']:.0%}",
            f"{r.get('profit_factor', 0):.2f}",
            f"{r['max_drawdown_pct']:.1f}%",
            f"{r['sharpe_ratio']:.2f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()

    best = max(results.items(), key=lambda x: x[1]["roi_pct"])
    print(f"Best strategy: {best[0]} (ROI: {best[1]['roi_pct']:+.2f}%)")

    return results


def run_walk_forward(config: dict, strategy_name: str = None):
    """Walk-forward validation — the anti-overfitting test."""
    name = strategy_name or config["strategy"]["name"]
    if name not in STRATEGY_MAP:
        print(f"Unknown strategy: {name}")
        return None

    df = fetch_backtest_data(config)
    validator = WalkForwardValidator(
        config, STRATEGY_MAP[name],
        train_periods=4, test_periods=1, n_folds=6
    )
    return validator.run(df)


def run_monte_carlo(config: dict, strategy_name: str = None):
    """Monte Carlo robustness test — is profit skill or luck?"""
    name = strategy_name or config["strategy"]["name"]
    if name not in STRATEGY_MAP:
        print(f"Unknown strategy: {name}")
        return None

    # First run a regular backtest to get trade PnLs
    strategy = STRATEGY_MAP[name](config)
    df = fetch_backtest_data(config)
    engine = BacktestEngine(config, strategy)
    engine.run(df)

    # Extract PnLs from closed trades
    trade_pnls = [
        t.pnl for t in engine.risk_manager.closed_trades
        if t.pnl is not None
    ]

    if not trade_pnls:
        print("No trades to simulate!")
        return None

    mc = MonteCarloSimulator(
        initial_capital=float(config["trading"]["initial_capital"]),
        n_simulations=1000,
    )
    return mc.run(trade_pnls)


def run_full_validation(config: dict, strategy_name: str = None):
    """Complete validation suite: Walk-Forward + Monte Carlo."""
    name = strategy_name or config["strategy"]["name"]
    print("\n" + "#" * 70)
    print(f"  FULL VALIDATION SUITE: {name}")
    print("#" * 70)

    # 1. Walk-Forward
    print("\n  [1/2] Walk-Forward Validation...")
    wf_result = run_walk_forward(config, name)

    # 2. Monte Carlo
    print("\n  [2/2] Monte Carlo Robustness Test...")
    mc_result = run_monte_carlo(config, name)

    # Final verdict
    print("\n" + "#" * 70)
    print("  FINAL VALIDATION VERDICT")
    print("#" * 70)

    wf_pass = wf_result and wf_result.get("valid", False)
    mc_pass = mc_result and mc_result.get("robust", False)

    print(f"  Walk-Forward:  {'PASS' if wf_pass else 'FAIL'}")
    print(f"  Monte Carlo:   {'PASS' if mc_pass else 'FAIL'}")

    if wf_pass and mc_pass:
        print("\n  STRATEGY VALIDATED -- Ready for paper trading.")
        print("  Paper trade for 2+ weeks before deploying with real capital.")
    elif wf_pass:
        print("\n  PARTIALLY VALIDATED -- Walk-Forward passed, but Monte Carlo failed.")
        print("  Strategy may be sensitive to trade sequencing. Use with caution.")
    elif mc_pass:
        print("\n  PARTIALLY VALIDATED -- Monte Carlo passed, but Walk-Forward failed.")
        print("  Strategy may be overfitted to specific time periods.")
    else:
        print("\n  STRATEGY FAILED VALIDATION")
        print("  DO NOT deploy with real money. Needs parameter tuning or redesign.")

    print("#" * 70 + "\n")
    return {"walk_forward": wf_result, "monte_carlo": mc_result}


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest mode")
    parser.add_argument("--backtest-all", action="store_true",
                        help="Backtest all strategies and compare")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Walk-forward validation (anti-overfit test)")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Monte Carlo robustness test")
    parser.add_argument("--validate", action="store_true",
                        help="Full validation suite (WF + MC)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Override strategy name")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.strategy:
        config["strategy"]["name"] = args.strategy

    log_cfg = config.get("logging", {})
    setup_logger(
        "trading_bot",
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/trading.log"),
        console=log_cfg.get("console", True),
    )

    if args.validate:
        run_full_validation(config)
    elif args.walk_forward:
        run_walk_forward(config)
    elif args.monte_carlo:
        run_monte_carlo(config)
    elif args.backtest_all:
        run_all_backtests(config)
    elif args.backtest:
        run_backtest(config)
    else:
        print("""
+============================================================+
|           CRYPTO TRADING BOT v2.0                          |
|                                                            |
|  WARNING: Trading involves risk of loss!                   |
|  Start with PAPER mode before using real money!            |
|  Never invest more than you can afford to lose!            |
|                                                            |
|  Mode: {mode:<12} Strategy: {strategy:<20}|
|  Symbol: {symbol:<11} Capital: ${capital:<18}|
|                                                            |
|  Architecture: Platform-agnostic via ExchangeAdapter       |
|  Supported: Binance, Bybit, KuCoin, + 100 more via CCXT   |
+============================================================+
        """.format(
            mode=config["trading"]["mode"].upper(),
            strategy=config["strategy"]["name"],
            symbol=config["trading"]["symbol"],
            capital=config["trading"]["initial_capital"],
        ))

        if config["trading"]["mode"] == "live":
            print("  You are about to trade with REAL MONEY!")
            confirm = input("  Type 'YES' to confirm: ")
            if confirm != "YES":
                print("  Aborted.")
                return

        bot = TradingBot(config)
        bot.start()


if __name__ == "__main__":
    main()
