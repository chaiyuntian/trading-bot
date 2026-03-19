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
            "min_hold_candles": 0,
            "sl_cooldown_candles": 0,
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
