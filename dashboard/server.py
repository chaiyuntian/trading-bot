"""Live Trading Dashboard — real-time simulation viewer.

Usage:
  cd dashboard && python server.py

Then open http://localhost:8765 in your browser.

Streams simulation data via WebSocket to a live TradingView chart.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.kama_trend import KamaTrendStrategy
from src.strategies.regime import detect_regime
from src.strategies.base import Signal
from src.risk.manager import RiskManager
from src.indicators.technical import add_kama, add_kama_efficiency_ratio

app = FastAPI(title="Trading Bot Dashboard")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

STRATEGY_MAP = {
    "rsi_macd": RsiMacdStrategy,
    "mean_reversion": MeanReversionStrategy,
    "grid": GridTradingStrategy,
    "dca_momentum": DCAMomentumStrategy,
    "ensemble": EnsembleStrategy,
    "kama_trend": KamaTrendStrategy,
}

CONFIG = {
    "trading": {
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "initial_capital": 1000,
        "mode": "paper",
    },
    "strategy": {
        "name": "ensemble",
        "params": {
            "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "ema_period": 50, "grid_levels": 10, "grid_spacing_atr_mult": 0.5,
            "dca_interval": 4, "ensemble_min_consensus": 0.4,
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


def fetch_data():
    """Fetch historical OHLCV data from Binance."""
    import ccxt
    symbol = CONFIG["trading"]["symbol"]
    timeframe = CONFIG["trading"]["timeframe"]

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
    return df


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        # Wait for config from client
        msg = await websocket.receive_text()
        client_cfg = json.loads(msg)
        strategy_name = client_cfg.get("strategy", "ensemble")
        speed = client_cfg.get("speed", 50)  # ms between candles

        await websocket.send_json({"type": "status", "msg": "Fetching historical data..."})
        df = fetch_data()
        await websocket.send_json({
            "type": "status",
            "msg": f"Loaded {len(df)} candles. Starting simulation...",
        })

        # Send all historical candles for the chart
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time": int(ts.timestamp()),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
            })

        # Send candles in chunks for speed
        chunk_size = 500
        for i in range(0, len(candles), chunk_size):
            await websocket.send_json({
                "type": "candles",
                "data": candles[i:i + chunk_size],
            })

        # ── Run simulation ──
        config = CONFIG.copy()
        config["strategy"]["name"] = strategy_name

        strategy_cls = STRATEGY_MAP.get(strategy_name, EnsembleStrategy)
        strategy = strategy_cls(config)
        risk_manager = RiskManager(config)

        lookback = 60
        fee_rate = 0.001
        slippage = 0.0005
        trade_counter = 0
        equity_curve = []

        await websocket.send_json({"type": "sim_start", "total": len(df) - lookback})

        for i in range(lookback, len(df)):
            window = df.iloc[:i + 1]
            ts = df.index[i]
            current_price = float(df.iloc[i]["close"])

            # Track equity
            open_pnl = sum(
                (current_price - t.entry_price) * t.amount
                if t.side == "buy" else
                (t.entry_price - current_price) * t.amount
                for t in risk_manager.open_trades
            )
            equity = risk_manager.capital + open_pnl
            equity_curve.append(equity)

            # Check stops
            stopped_trades = risk_manager.check_stops(CONFIG["trading"]["symbol"], current_price)
            for st in stopped_trades:
                await websocket.send_json({
                    "type": "trade",
                    "action": "close",
                    "time": int(ts.timestamp()),
                    "price": round(st.exit_price, 2),
                    "pnl": round(st.pnl, 4),
                    "reason": "stop" if st.exit_price <= st.entry_price else "take_profit",
                })

            # Can trade?
            can_trade, reason = risk_manager.can_open_trade()

            # Generate signal
            signal = strategy.generate_signal(window)

            # Detect regime
            regime = detect_regime(window)

            if signal == "buy" and can_trade:
                stop_price = risk_manager.get_stop_loss(current_price, "buy")
                amount = risk_manager.calculate_position_size(current_price, stop_price)
                if amount * current_price >= 5:
                    exec_price = current_price * (1 + slippage)
                    fee = amount * exec_price * fee_rate
                    risk_manager.capital -= fee
                    trade_counter += 1
                    risk_manager.open_trade(
                        f"sim_{trade_counter}", CONFIG["trading"]["symbol"],
                        "buy", exec_price, amount,
                    )
                    await websocket.send_json({
                        "type": "trade",
                        "action": "buy",
                        "time": int(ts.timestamp()),
                        "price": round(exec_price, 2),
                        "amount": round(amount, 6),
                        "fee": round(fee, 4),
                    })

            elif signal == "sell" and risk_manager.open_trades:
                for trade in list(risk_manager.open_trades):
                    exec_price = current_price * (1 - slippage)
                    fee = trade.amount * exec_price * fee_rate
                    risk_manager.capital -= fee
                    risk_manager.close_trade(trade, exec_price, "signal")
                    await websocket.send_json({
                        "type": "trade",
                        "action": "sell",
                        "time": int(ts.timestamp()),
                        "price": round(exec_price, 2),
                        "pnl": round(trade.pnl, 4),
                        "fee": round(fee, 4),
                    })

            # Send periodic updates
            if (i - lookback) % 5 == 0:
                stats = risk_manager.get_stats()
                peak = max(equity_curve) if equity_curve else equity
                dd = (peak - equity) / peak if peak > 0 else 0

                await websocket.send_json({
                    "type": "update",
                    "candle_idx": i - lookback,
                    "time": int(ts.timestamp()),
                    "price": round(current_price, 2),
                    "equity": round(equity, 2),
                    "capital": round(risk_manager.capital, 2),
                    "drawdown": round(dd * 100, 2),
                    "open_trades": len(risk_manager.open_trades),
                    "total_trades": stats["total_trades"],
                    "win_rate": round(stats["win_rate"] * 100, 1),
                    "total_pnl": round(stats["total_pnl"], 2),
                    "regime": regime.value,
                    "signal": signal,
                })

            # Control playback speed
            if speed > 0:
                await asyncio.sleep(speed / 1000.0)

        # ── Final stats ──
        last_price = float(df.iloc[-1]["close"])
        for trade in list(risk_manager.open_trades):
            risk_manager.close_trade(trade, last_price, "sim_end")

        stats = risk_manager.get_stats()

        # Calculate Sharpe/Sortino
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0])
        std = np.std(returns)
        sharpe = float((np.mean(returns) / std) * np.sqrt(365 * 24 * 4)) if std > 0 else 0
        neg_ret = returns[returns < 0]
        ds = np.std(neg_ret) if len(neg_ret) > 0 else 1
        sortino = float((np.mean(returns) / ds) * np.sqrt(365 * 24 * 4)) if ds > 0 else 0

        peak = max(equity_curve) if equity_curve else CONFIG["trading"]["initial_capital"]
        trough = min(equity_curve) if equity_curve else CONFIG["trading"]["initial_capital"]
        max_dd = (peak - trough) / peak if peak > 0 else 0

        await websocket.send_json({
            "type": "done",
            "stats": {
                "total_trades": stats["total_trades"],
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "win_rate": round(stats["win_rate"] * 100, 1),
                "total_pnl": round(stats["total_pnl"], 2),
                "roi_pct": round(stats["roi_pct"], 2),
                "profit_factor": round(stats.get("profit_factor", 0), 2),
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(sortino, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "final_capital": round(stats["capital"], 2),
                "total_fees": round(
                    CONFIG["trading"]["initial_capital"] - stats["capital"] + stats["total_pnl"], 2
                ),
            },
        })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\n  Trading Bot Dashboard")
    print("  http://localhost:8765\n")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
