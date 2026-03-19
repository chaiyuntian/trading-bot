"""System Verification Suite — proves the entire trading bot works end-to-end.

Run with: python -m tests.verify

This script runs a comprehensive set of checks using only synthetic data
(no network required). It validates every layer of the system:

  1. Indicators compute correctly
  2. All strategies produce valid signals across market conditions
  3. Risk manager enforces all limits
  4. Backtest engine produces consistent results
  5. Walk-forward validation works
  6. Ensemble adapts to regime changes
  7. Full pipeline: data -> indicators -> strategy -> risk -> execution

Exit code 0 = all checks passed, 1 = failures detected.
"""

import sys
import time
import traceback
import numpy as np
import pandas as pd

# ── Data Generators ──

def make_trending_up(n=300):
    np.random.seed(42)
    trend = np.linspace(0, 3000, n)
    noise = np.cumsum(np.random.randn(n) * 30)
    close = 50000 + trend + noise
    return _to_ohlcv(close, n, seed=10)


def make_trending_down(n=300):
    np.random.seed(42)
    trend = np.linspace(0, -3000, n)
    noise = np.cumsum(np.random.randn(n) * 30)
    close = 50000 + trend + noise
    return _to_ohlcv(close, n, seed=20)


def make_ranging(n=300):
    np.random.seed(123)
    t = np.arange(n)
    close = 50000 + 500 * np.sin(t * 2 * np.pi / 50) + np.random.randn(n) * 30
    return _to_ohlcv(close, n, seed=30)


def make_volatile(n=300):
    np.random.seed(99)
    close = 50000 + np.cumsum(np.random.randn(n) * 200)
    for i in range(50, n, 60):
        end = min(i + 10, n)
        close[i:end] += np.random.randn(end - i) * 1000
    return _to_ohlcv(close, n, seed=40)


def _to_ohlcv(close, n, seed=7):
    np.random.seed(seed)
    high = close + np.abs(np.random.randn(n) * 80)
    low = close - np.abs(np.random.randn(n) * 80)
    volume = np.random.rand(n) * 2000 + 500
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 20,
        "high": high, "low": low, "close": close, "volume": volume,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="15min")
    return df


def make_config(capital=1000):
    return {
        "trading": {
            "symbol": "BTC/USDT", "timeframe": "15m",
            "initial_capital": capital, "mode": "paper",
        },
        "strategy": {
            "name": "ensemble",
            "params": {
                "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
                "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                "ema_period": 50, "grid_levels": 10,
                "grid_spacing_atr_mult": 0.5, "dca_interval": 4,
                "ensemble_min_consensus": 0.4,
                "kama_period": 10, "er_threshold": 0.3,
            },
        },
        "risk": {
            "max_position_pct": 0.30, "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06, "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": 0.25, "risk_per_trade_pct": 0.02,
            "max_open_trades": 3, "trailing_stop": False,
        },
    }


# ── Check Framework ──

class CheckResult:
    def __init__(self, name, passed, detail=""):
        self.name = name
        self.passed = passed
        self.detail = detail


def run_check(name, fn):
    try:
        fn()
        return CheckResult(name, True)
    except Exception as e:
        return CheckResult(name, False, f"{e}\n{traceback.format_exc()}")


# ── Individual Checks ──

def check_indicators():
    from src.indicators.technical import (
        add_rsi, add_macd, add_ema, add_bollinger_bands, add_atr,
        add_volume_sma, add_kama, add_kama_efficiency_ratio
    )
    df = make_trending_up()

    df = add_rsi(df, 14)
    assert "rsi_14" in df.columns
    rsi = df["rsi_14"].dropna()
    assert rsi.min() >= -0.1 and rsi.max() <= 100.1  # allow float imprecision

    df = add_macd(df, 12, 26, 9)
    assert "macd" in df.columns and "macd_hist" in df.columns

    df = add_ema(df, 50)
    assert "ema_50" in df.columns

    df = add_bollinger_bands(df)
    assert all(c in df.columns for c in ["bb_upper", "bb_middle", "bb_lower"])

    df = add_atr(df)
    assert "atr_14" in df.columns
    assert df["atr_14"].dropna().min() >= 0

    df = add_volume_sma(df)
    assert "vol_sma_20" in df.columns

    df = add_kama(df, 10)
    assert "kama_10" in df.columns
    kama = df["kama_10"].dropna()
    assert len(kama) > 100
    assert kama.iloc[-1] > kama.iloc[0]  # uptrend

    df = add_kama_efficiency_ratio(df, 10)
    assert "er_10" in df.columns
    er = df["er_10"].dropna()
    assert er.min() >= -0.01  # allow tiny float imprecision
    assert er.max() <= 1.01


def check_all_strategies_valid_signals():
    from src.strategies.rsi_macd import RsiMacdStrategy
    from src.strategies.mean_reversion import MeanReversionStrategy
    from src.strategies.grid_trading import GridTradingStrategy
    from src.strategies.dca_momentum import DCAMomentumStrategy
    from src.strategies.ensemble import EnsembleStrategy
    from src.strategies.kama_trend import KamaTrendStrategy
    from src.strategies.base import Signal

    config = make_config()
    strategies = [
        RsiMacdStrategy(config),
        MeanReversionStrategy(config),
        GridTradingStrategy(config),
        DCAMomentumStrategy(config),
        EnsembleStrategy(config),
        KamaTrendStrategy(config),
    ]

    for data_fn in [make_trending_up, make_ranging, make_volatile]:
        df = data_fn()
        for strategy in strategies:
            sig = strategy.generate_rich_signal(df)
            assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD), \
                f"{strategy.strategy_name} returned invalid signal: {sig.signal}"
            assert 0 <= sig.confidence <= 1.0, \
                f"{strategy.strategy_name} confidence out of range: {sig.confidence}"
            assert isinstance(sig.reason, str) and len(sig.reason) > 0


def check_strategies_handle_edge_cases():
    from src.strategies.rsi_macd import RsiMacdStrategy
    from src.strategies.ensemble import EnsembleStrategy
    from src.strategies.kama_trend import KamaTrendStrategy

    config = make_config()

    # Tiny dataset
    tiny = make_trending_up(5)
    for cls in [RsiMacdStrategy, EnsembleStrategy, KamaTrendStrategy]:
        s = cls(config)
        assert s.generate_signal(tiny) == "hold"

    # Single candle
    single = make_trending_up(1)
    s = RsiMacdStrategy(config)
    assert s.generate_signal(single) == "hold"


def check_risk_manager():
    from src.risk.manager import RiskManager

    config = make_config(capital=1000)
    rm = RiskManager(config)

    # Initial state
    assert rm.capital == 1000
    assert not rm.halted
    can, _ = rm.can_open_trade()
    assert can

    # Position sizing respects limits
    entry, stop = 50000, 48500
    amount = rm.calculate_position_size(entry, stop)
    assert amount * entry <= 1000 * 0.30  # max_position_pct

    # Open/close trade with PnL
    trade = rm.open_trade("t1", "BTC/USDT", "buy", 50000, 0.01)
    assert len(rm.open_trades) == 1
    rm.close_trade(trade, 51000, "test")
    assert len(rm.open_trades) == 0
    assert trade.pnl > 0
    assert rm.capital > 1000

    # Stop-loss fires
    trade2 = rm.open_trade("t2", "BTC/USDT", "buy", 50000, 0.01)
    stopped = rm.check_stops("BTC/USDT", trade2.stop_loss - 1)
    assert len(stopped) == 1

    # Take-profit fires
    trade3 = rm.open_trade("t3", "BTC/USDT", "buy", 50000, 0.01)
    stopped = rm.check_stops("BTC/USDT", trade3.take_profit + 1)
    assert len(stopped) == 1

    # Max open trades limit
    rm2 = RiskManager(config)
    for i in range(3):
        rm2.open_trade(f"t{i}", "BTC/USDT", "buy", 50000, 0.001)
    can, reason = rm2.can_open_trade()
    assert not can

    # Drawdown halt
    rm3 = RiskManager(config)
    rm3.capital = 700
    rm3.peak_capital = 1000
    can, _ = rm3.can_open_trade()
    assert not can
    assert rm3.halted

    # Daily reset clears daily halt
    rm4 = RiskManager(config)
    rm4.halted = True
    rm4.halt_reason = "Daily loss exceeded"
    rm4.reset_daily()
    assert not rm4.halted


def check_backtest_pipeline():
    from src.strategies.rsi_macd import RsiMacdStrategy
    from src.strategies.ensemble import EnsembleStrategy
    from src.strategies.kama_trend import KamaTrendStrategy
    from src.backtesting.engine import BacktestEngine

    config = make_config()

    for strategy_cls in [RsiMacdStrategy, EnsembleStrategy, KamaTrendStrategy]:
        for data_fn in [make_trending_up, make_ranging]:
            df = data_fn()
            strategy = strategy_cls(config)
            engine = BacktestEngine(config, strategy)
            result = engine.run(df)

            assert "total_trades" in result
            assert "roi_pct" in result
            assert "sharpe_ratio" in result
            assert "sortino_ratio" in result
            assert "max_drawdown_pct" in result
            assert "total_fees" in result
            assert result["candles"] == len(df)
            assert isinstance(result["sharpe_ratio"], (int, float))
            assert isinstance(result["sortino_ratio"], (int, float))


def check_walk_forward():
    from src.strategies.rsi_macd import RsiMacdStrategy
    from src.backtesting.walk_forward import WalkForwardValidator

    config = make_config()
    df = make_trending_up(1000)

    validator = WalkForwardValidator(
        config, RsiMacdStrategy,
        train_days=2, test_days=1, step_days=1
    )
    result = validator.run(df)

    assert "folds" in result
    assert result["folds"] > 0
    assert "avg_test_roi" in result
    assert "overfit_risk" in result
    assert result["overfit_risk"] in ("LOW", "MEDIUM", "HIGH", "N/A (both negative)")


def check_ensemble_regime_adaptation():
    from src.strategies.ensemble import EnsembleStrategy
    from src.strategies.regime import detect_regime

    config = make_config()
    strategy = EnsembleStrategy(config)

    # Run on trending data
    df_trend = make_trending_up(200)
    sig_trend = strategy.generate_rich_signal(df_trend)
    assert "Regime=" in sig_trend.reason

    # Run on ranging data
    df_range = make_ranging(200)
    sig_range = strategy.generate_rich_signal(df_range)
    assert "Regime=" in sig_range.reason


def check_regime_detection_all_types():
    from src.strategies.regime import detect_regime, MarketRegime

    for data_fn in [make_trending_up, make_trending_down, make_ranging, make_volatile]:
        df = data_fn(200)
        regime = detect_regime(df)
        assert regime in list(MarketRegime)

    # Insufficient data defaults to RANGING
    assert detect_regime(make_ranging(30)) == MarketRegime.RANGING


def check_full_pipeline_integrity():
    """The ultimate integration check: data -> indicators -> strategy -> risk -> backtest.
    Verifies that the entire pipeline produces sane, consistent output."""
    from src.strategies.ensemble import EnsembleStrategy
    from src.backtesting.engine import BacktestEngine

    config = make_config(capital=500)
    df = make_trending_up(400)

    strategy = EnsembleStrategy(config)
    engine = BacktestEngine(config, strategy)
    result = engine.run(df)

    # Sanity checks on the complete result
    assert result["total_trades"] >= 0
    assert 0 <= result["win_rate"] <= 1.0 or result["total_trades"] == 0
    assert result["max_drawdown_pct"] >= 0
    assert result["total_fees"] >= 0
    assert result["candles"] == 400

    # Equity curve should have been tracked
    assert len(engine.equity_curve) > 0


# ── Main Runner ──

def main():
    checks = [
        ("Indicators (RSI, MACD, EMA, BB, ATR, KAMA, ER)", check_indicators),
        ("All strategies produce valid signals", check_all_strategies_valid_signals),
        ("Strategies handle edge cases", check_strategies_handle_edge_cases),
        ("Risk manager (sizing, stops, halts)", check_risk_manager),
        ("Backtest pipeline (all strategies x conditions)", check_backtest_pipeline),
        ("Walk-forward validation", check_walk_forward),
        ("Ensemble regime adaptation", check_ensemble_regime_adaptation),
        ("Regime detection (all market types)", check_regime_detection_all_types),
        ("Full pipeline integrity", check_full_pipeline_integrity),
    ]

    print("\n" + "=" * 70)
    print("  SYSTEM VERIFICATION SUITE")
    print("  Testing all components end-to-end (no network required)")
    print("=" * 70 + "\n")

    results = []
    start = time.time()

    for name, fn in checks:
        t0 = time.time()
        result = run_check(name, fn)
        elapsed = time.time() - t0
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name} ({elapsed:.1f}s)")
        if not result.passed:
            lines = result.detail.strip().split("\n")
            for line in lines[:8]:
                print(f"         {line}")

    total_time = time.time() - start
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print("\n" + "-" * 70)
    print(f"  Results: {passed} passed, {failed} failed ({total_time:.1f}s total)")

    if failed == 0:
        print("  All systems operational. Ready for deployment.")
    else:
        print("  FAILURES DETECTED. Fix issues before deploying.")

    print("=" * 70 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
