"""
Crypto Trading Bot - Main Entry Point
======================================
Usage:
  python -m src                      # Run bot (paper trading)
  python -m src --backtest           # Run backtest
  python -m src --backtest-all       # Backtest all strategies
  python -m src --config path        # Custom config file
  python -m src --strategy rsi_macd  # Override strategy
"""

import argparse
import os
import sys
import yaml
from src.bot import TradingBot, STRATEGY_MAP
from src.backtesting.engine import BacktestEngine
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
    """Run walk-forward validation to detect overfitting."""
    from src.backtesting.walk_forward import WalkForwardValidator

    name = strategy_name or config["strategy"]["name"]
    if name not in STRATEGY_MAP:
        print(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
        return None

    df = fetch_backtest_data(config)
    validator = WalkForwardValidator(
        config, STRATEGY_MAP[name],
        train_days=30, test_days=10, step_days=10
    )
    return validator.run(df)


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest mode")
    parser.add_argument("--backtest-all", action="store_true",
                        help="Backtest all strategies and compare")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward validation (detect overfitting)")
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

    if args.walk_forward:
        run_walk_forward(config)
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
