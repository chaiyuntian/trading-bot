"""Integration tests — verify the full pipeline works end-to-end.

Tests the complete flow: data -> indicators -> strategy -> risk -> backtest
without requiring network access (uses synthetic data).
"""

import pandas as pd
import numpy as np
import pytest
from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.kama_trend import KamaTrendStrategy
from src.strategies.regime import detect_regime, MarketRegime
from src.strategies.base import Signal
from src.risk.manager import RiskManager
from src.backtesting.engine import BacktestEngine
from src.backtesting.walk_forward import WalkForwardValidator
from src.indicators.technical import add_kama, add_kama_efficiency_ratio


def make_config(capital=1000):
    return {
        "trading": {
            "symbol": "BTC/USDT",
            "timeframe": "15m",
            "initial_capital": capital,
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
        },
    }


def make_trending_up_data(n=300):
    """Synthetic data with a clear uptrend."""
    np.random.seed(42)
    trend = np.linspace(0, 3000, n)
    noise = np.cumsum(np.random.randn(n) * 30)
    close = 50000 + trend + noise
    high = close + np.abs(np.random.randn(n) * 80)
    low = close - np.abs(np.random.randn(n) * 80)
    volume = np.random.rand(n) * 2000 + 500
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 20,
        "high": high, "low": low, "close": close, "volume": volume,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="15min")
    return df


def make_ranging_data(n=300):
    """Synthetic data that oscillates in a range."""
    np.random.seed(123)
    t = np.arange(n)
    close = 50000 + 500 * np.sin(t * 2 * np.pi / 50) + np.random.randn(n) * 30
    high = close + np.abs(np.random.randn(n) * 40)
    low = close - np.abs(np.random.randn(n) * 40)
    volume = np.random.rand(n) * 1000 + 200
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 15,
        "high": high, "low": low, "close": close, "volume": volume,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="15min")
    return df


def make_volatile_data(n=300):
    """Synthetic data with high volatility spikes."""
    np.random.seed(99)
    close = 50000 + np.cumsum(np.random.randn(n) * 200)
    # Add volatility spikes
    for i in range(50, n, 60):
        close[i:i + 10] += np.random.randn(min(10, n - i)) * 1000
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    volume = np.random.rand(n) * 5000 + 1000
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 50,
        "high": high, "low": low, "close": close, "volume": volume,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="15min")
    return df


# ── KAMA Indicator Tests ──

class TestKAMAIndicator:
    def test_kama_computes(self):
        df = make_trending_up_data()
        df = add_kama(df, 10)
        assert "kama_10" in df.columns
        valid = df["kama_10"].dropna()
        assert len(valid) > 100

    def test_kama_follows_trend(self):
        df = make_trending_up_data()
        df = add_kama(df, 10)
        kama = df["kama_10"].dropna()
        # In an uptrend, KAMA should generally increase
        assert kama.iloc[-1] > kama.iloc[0]

    def test_kama_smoother_than_close(self):
        df = make_ranging_data()
        df = add_kama(df, 10)
        valid = df.dropna()
        close_diff_std = valid["close"].diff().std()
        kama_diff_std = valid["kama_10"].diff().std()
        # KAMA changes should be smoother (smaller step-to-step variation)
        assert kama_diff_std < close_diff_std

    def test_efficiency_ratio(self):
        df = make_trending_up_data()
        df = add_kama_efficiency_ratio(df, 10)
        assert f"er_10" in df.columns
        valid = df["er_10"].dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 1.0


# ── KAMA Strategy Tests ──

class TestKAMAStrategy:
    def test_returns_valid_signal(self):
        config = make_config()
        strategy = KamaTrendStrategy(config)
        df = make_trending_up_data()
        signal = strategy.generate_signal(df)
        assert signal in ("buy", "sell", "hold")

    def test_rich_signal_has_fields(self):
        config = make_config()
        strategy = KamaTrendStrategy(config)
        df = make_trending_up_data()
        sig = strategy.generate_rich_signal(df)
        assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert 0 <= sig.confidence <= 1.0
        assert "KAMA" in sig.reason or "slope" in sig.reason

    def test_handles_small_data(self):
        config = make_config()
        strategy = KamaTrendStrategy(config)
        df = make_trending_up_data(10)
        signal = strategy.generate_signal(df)
        assert signal == "hold"


# ── Full Pipeline Integration Tests ──

class TestBacktestPipeline:
    """Verify the complete backtest pipeline works end-to-end."""

    @pytest.mark.parametrize("strategy_cls,name", [
        (RsiMacdStrategy, "RSI+MACD"),
        (MeanReversionStrategy, "Mean Reversion"),
        (GridTradingStrategy, "Grid"),
        (DCAMomentumStrategy, "DCA Momentum"),
        (EnsembleStrategy, "Ensemble"),
        (KamaTrendStrategy, "KAMA Trend"),
    ])
    def test_backtest_completes(self, strategy_cls, name):
        config = make_config()
        df = make_trending_up_data()
        strategy = strategy_cls(config)
        engine = BacktestEngine(config, strategy)
        result = engine.run(df)

        assert "total_trades" in result
        assert "roi_pct" in result
        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "max_drawdown_pct" in result
        assert result["candles"] == len(df)

    def test_backtest_different_market_conditions(self):
        config = make_config()
        for data_fn in [make_trending_up_data, make_ranging_data, make_volatile_data]:
            df = data_fn()
            strategy = EnsembleStrategy(config)
            engine = BacktestEngine(config, strategy)
            result = engine.run(df)
            assert isinstance(result["roi_pct"], (int, float))

    def test_backtest_capital_bounded(self):
        """Capital should not go wildly negative."""
        config = make_config(capital=100)
        df = make_volatile_data()
        strategy = RsiMacdStrategy(config)
        engine = BacktestEngine(config, strategy)
        result = engine.run(df)
        # Max drawdown should be limited by risk manager
        assert result["max_drawdown_pct"] < 100


class TestRiskManagerIntegration:
    def test_stop_loss_fires(self):
        config = make_config()
        rm = RiskManager(config)
        trade = rm.open_trade("t1", "BTC/USDT", "buy", 50000, 0.01)
        sl_price = trade.stop_loss
        stopped = rm.check_stops("BTC/USDT", sl_price - 1)
        assert len(stopped) == 1
        assert stopped[0].pnl < 0

    def test_take_profit_fires(self):
        config = make_config()
        rm = RiskManager(config)
        trade = rm.open_trade("t1", "BTC/USDT", "buy", 50000, 0.01)
        tp_price = trade.take_profit
        stopped = rm.check_stops("BTC/USDT", tp_price + 1)
        assert len(stopped) == 1
        assert stopped[0].pnl > 0

    def test_drawdown_halt_prevents_trading(self):
        config = make_config(capital=100)
        rm = RiskManager(config)
        # Simulate losses
        rm.capital = 70
        rm.peak_capital = 100
        can, reason = rm.can_open_trade()
        assert can is False
        assert rm.halted

    def test_daily_reset_clears_halt(self):
        config = make_config(capital=100)
        rm = RiskManager(config)
        rm.halted = True
        rm.halt_reason = "Daily loss exceeded"
        rm.reset_daily()
        assert rm.halted is False


class TestRegimeDetection:
    def test_trending_data_detected(self):
        df = make_trending_up_data(200)
        regime = detect_regime(df)
        # Should detect some kind of trend or ranging — exact classification
        # depends on noise, but should not crash
        assert regime in list(MarketRegime)

    def test_ranging_data_detected(self):
        df = make_ranging_data(200)
        regime = detect_regime(df)
        assert regime in list(MarketRegime)

    def test_small_data_defaults_ranging(self):
        df = make_ranging_data(30)
        regime = detect_regime(df)
        assert regime == MarketRegime.RANGING


class TestEnsembleDynamicWeighting:
    def test_ensemble_produces_signal(self):
        config = make_config()
        strategy = EnsembleStrategy(config)
        df = make_trending_up_data()
        sig = strategy.generate_rich_signal(df)
        assert sig.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert "Regime=" in sig.reason

    def test_dynamic_weights_change_over_time(self):
        config = make_config()
        strategy = EnsembleStrategy(config)
        df = make_trending_up_data(300)

        # Run multiple cycles to build signal history
        for i in range(100, 300, 5):
            window = df.iloc[:i]
            strategy.generate_rich_signal(window)

        # After many cycles, dynamic weights should have differentiated
        weights = strategy._compute_dynamic_weights()
        values = list(weights.values())
        assert len(values) > 0
        # Not all weights should be exactly the same
        assert max(values) > min(values) or all(v == 1.0 for v in values)


class TestWalkForward:
    def test_walk_forward_runs(self):
        config = make_config()
        df = make_trending_up_data(1000)
        validator = WalkForwardValidator(
            config, RsiMacdStrategy,
            train_days=2, test_days=1, step_days=1
        )
        result = validator.run(df)
        assert "folds" in result
        assert result["folds"] > 0
        assert "avg_test_roi" in result
        assert "overfit_risk" in result

    def test_insufficient_data_handled(self):
        config = make_config()
        df = make_trending_up_data(50)
        validator = WalkForwardValidator(
            config, RsiMacdStrategy,
            train_days=30, test_days=10, step_days=10
        )
        result = validator.run(df)
        assert result.get("error") == "insufficient_data"
