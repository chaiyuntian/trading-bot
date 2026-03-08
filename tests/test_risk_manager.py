import pytest
from src.risk.manager import RiskManager


def make_config(capital=100, **overrides):
    cfg = {
        "trading": {"initial_capital": capital},
        "risk": {
            "max_position_pct": 0.30,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
            "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": 0.25,
            "risk_per_trade_pct": 0.02,
            "max_open_trades": 3,
            "trailing_stop": False,
        }
    }
    cfg["risk"].update(overrides)
    return cfg


def test_initial_state():
    rm = RiskManager(make_config())
    assert rm.capital == 100
    assert rm.initial_capital == 100
    assert len(rm.open_trades) == 0
    assert not rm.halted


def test_position_sizing():
    rm = RiskManager(make_config())
    # With 3% stop loss, 2% risk = $2 risk
    # stop_distance = 3% = 0.03
    # position_value = $2 / 0.03 = $66.67
    entry = 50000
    stop = entry * 0.97  # 3% below
    amount = rm.calculate_position_size(entry, stop)
    position_value = amount * entry
    assert position_value <= 100 * 0.30  # Max 30% of capital


def test_can_open_trade():
    rm = RiskManager(make_config())
    can, reason = rm.can_open_trade()
    assert can is True


def test_max_open_trades():
    rm = RiskManager(make_config())
    for i in range(3):
        rm.open_trade(f"t{i}", "BTC/USDT", "buy", 50000, 0.001)
    can, reason = rm.can_open_trade()
    assert can is False
    assert "Max open trades" in reason


def test_stop_loss():
    rm = RiskManager(make_config())
    entry = 50000
    sl = rm.get_stop_loss(entry, "buy")
    assert sl == entry * 0.97  # 3% below


def test_take_profit():
    rm = RiskManager(make_config())
    entry = 50000
    tp = rm.get_take_profit(entry, "buy")
    assert tp == entry * 1.06  # 6% above


def test_trade_pnl():
    rm = RiskManager(make_config())
    trade = rm.open_trade("t1", "BTC/USDT", "buy", 50000, 0.001)
    rm.close_trade(trade, 51000, "test")
    assert trade.pnl == pytest.approx(1.0)  # (51000-50000)*0.001
    assert rm.capital == pytest.approx(101.0)


def test_drawdown_halt():
    rm = RiskManager(make_config())
    rm.capital = 74  # 26% drawdown
    rm.peak_capital = 100
    can, reason = rm.can_open_trade()
    assert can is False
    assert rm.halted


def test_stats():
    rm = RiskManager(make_config())
    t1 = rm.open_trade("t1", "BTC/USDT", "buy", 50000, 0.001)
    rm.close_trade(t1, 51000, "tp")
    t2 = rm.open_trade("t2", "BTC/USDT", "buy", 51000, 0.001)
    rm.close_trade(t2, 50500, "sl")

    stats = rm.get_stats()
    assert stats["total_trades"] == 2
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["win_rate"] == 0.5


def test_atr_based_stop_loss():
    """ATR-based stops adapt to volatility."""
    rm = RiskManager(make_config())
    entry = 50000
    atr = 500  # $500 ATR

    sl = rm.get_stop_loss(entry, "buy", atr=atr)
    assert sl == entry - (atr * 2.0)  # 2x ATR below entry

    tp = rm.get_take_profit(entry, "buy", atr=atr)
    assert tp == entry + (atr * 3.0)  # 3x ATR above entry (1.5:1 R/R)


def test_drawdown_scale_factor():
    """Position size shrinks as drawdown deepens."""
    rm = RiskManager(make_config())
    # No drawdown -> full size
    assert rm._drawdown_scale_factor() == 1.0

    # 10% drawdown (40% of max 25%) -> reduced
    rm.capital = 90
    rm.peak_capital = 100
    scale = rm._drawdown_scale_factor()
    assert 0.2 < scale < 1.0

    # At max drawdown -> minimum scale
    rm.capital = 75
    rm.peak_capital = 100
    scale = rm._drawdown_scale_factor()
    assert scale == 0.2


def test_loss_cooldown():
    """After consecutive losses, trading pauses briefly."""
    rm = RiskManager(make_config(loss_cooldown_after=2))
    rm.max_consecutive_losses_cooldown = 2

    # Two consecutive losses -> cooldown triggers
    t1 = rm.open_trade("t1", "BTC/USDT", "buy", 50000, 0.001)
    rm.close_trade(t1, 49000)  # Loss 1
    t2 = rm.open_trade("t2", "BTC/USDT", "buy", 50000, 0.001)
    rm.close_trade(t2, 49000)  # Loss 2 -> cooldown

    assert rm.cooldown_trades_remaining > 0
    can, reason = rm.can_open_trade()
    assert can is False
    assert "cooldown" in reason.lower()


def test_win_resets_loss_streak():
    """A winning trade resets the consecutive loss counter."""
    rm = RiskManager(make_config())
    t1 = rm.open_trade("t1", "BTC/USDT", "buy", 50000, 0.001)
    rm.close_trade(t1, 49000)  # Loss
    assert rm.consecutive_losses == 1

    t2 = rm.open_trade("t2", "BTC/USDT", "buy", 50000, 0.001)
    rm.close_trade(t2, 51000)  # Win
    assert rm.consecutive_losses == 0
