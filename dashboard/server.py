"""Live Trading Dashboard — real-time simulation viewer with v2 risk engine.

Usage:
  python -m src.main --backtest --viz              # backtest with live viz
  python -m src.main --backtest-all --viz          # compare all with viz
  cd dashboard && python server.py                 # standalone

Features:
  - TradingView candlestick chart with buy/sell markers
  - Real-time equity, PnL, drawdown, regime display
  - ATR-adaptive stops, trailing stops, min hold period
  - Alpha combiner with 21 signals
  - Configurable strategy, timeframe, capital from UI
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.bot import STRATEGY_MAP
from src.strategies.regime import detect_regime
from src.strategies.base import Signal
from src.risk.manager import RiskManager
from src.indicators.technical import add_atr

app = FastAPI(title="Trading Bot Dashboard")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def make_config(strategy="ensemble", timeframe="4h", capital=10000):
    return {
        "trading": {
            "symbol": "BTC/USDT", "timeframe": timeframe,
            "initial_capital": capital, "mode": "paper",
        },
        "strategy": {
            "name": strategy,
            "params": {
                "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
                "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                "ema_period": 50, "grid_levels": 10, "grid_spacing_atr_mult": 0.5,
                "dca_interval": 4, "ensemble_min_consensus": 0.4,
                "kama_period": 10, "er_threshold": 0.3,
                "combiner_threshold": 0.12,
            },
        },
        "risk": {
            "max_position_pct": 0.40, "stop_loss_pct": 0.03,
            "take_profit_pct": 0.08, "max_daily_loss_pct": 0.05,
            "max_drawdown_pct": 0.25, "risk_per_trade_pct": 0.03,
            "max_open_trades": 2, "trailing_stop": True,
            "trailing_stop_pct": 0.025, "min_hold_candles": 6,
            "sl_cooldown_candles": 4,
        },
    }


def fetch_data(symbol="BTC/USDT", timeframe="4h"):
    import ccxt
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


@app.get("/api/strategies")
async def list_strategies():
    return {"strategies": list(STRATEGY_MAP.keys())}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        msg = await websocket.receive_text()
        client_cfg = json.loads(msg)
        strategy_name = client_cfg.get("strategy", "alpha")
        speed = client_cfg.get("speed", 10)
        timeframe = client_cfg.get("timeframe", "4h")
        capital = client_cfg.get("capital", 10000)

        config = make_config(strategy_name, timeframe, capital)

        await websocket.send_json({"type": "status", "msg": f"Fetching {timeframe} data..."})
        df = fetch_data("BTC/USDT", timeframe)
        await websocket.send_json({
            "type": "status",
            "msg": f"Loaded {len(df)} candles ({timeframe}). Simulating...",
        })

        # Send candles
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time": int(ts.timestamp()),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
            })
        chunk_size = 500
        for i in range(0, len(candles), chunk_size):
            await websocket.send_json({"type": "candles", "data": candles[i:i + chunk_size]})

        # Pre-compute ATR
        df_atr = add_atr(df.copy(), 14)
        atr_values = df_atr["atr_14"].values

        # Setup
        strategy_cls = STRATEGY_MAP.get(strategy_name)
        if not strategy_cls:
            await websocket.send_json({"type": "error", "msg": f"Unknown strategy: {strategy_name}"})
            return
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
            price = float(df.iloc[i]["close"])
            atr = float(atr_values[i]) if not np.isnan(atr_values[i]) else None

            risk_manager.set_candle_index(i)

            open_pnl = sum(
                (price - t.entry_price) * t.amount if t.side == "buy"
                else (t.entry_price - price) * t.amount
                for t in risk_manager.open_trades
            )
            equity = risk_manager.capital + open_pnl
            equity_curve.append(equity)

            stopped = risk_manager.check_stops(config["trading"]["symbol"], price)
            for st in stopped:
                await websocket.send_json({
                    "type": "trade", "action": "close",
                    "time": int(ts.timestamp()),
                    "price": round(st.exit_price, 2),
                    "pnl": round(st.pnl, 4),
                    "reason": "stop_loss" if st.pnl < 0 else "take_profit",
                })

            can_trade, _ = risk_manager.can_open_trade()
            signal = strategy.generate_signal(window)
            regime = detect_regime(window)

            if signal == "buy" and can_trade:
                stop_price = risk_manager.get_stop_loss(price, "buy", atr)
                tp_price = risk_manager.get_take_profit(price, "buy", atr)
                amount = risk_manager.calculate_position_size(price, stop_price)
                if amount * price >= 5:
                    exec_price = price * (1 + slippage)
                    fee = amount * exec_price * fee_rate
                    risk_manager.capital -= fee
                    trade_counter += 1
                    risk_manager.open_trade(
                        f"sim_{trade_counter}", config["trading"]["symbol"],
                        "buy", exec_price, amount,
                        stop_loss=stop_price, take_profit=tp_price, atr=atr,
                    )
                    await websocket.send_json({
                        "type": "trade", "action": "buy",
                        "time": int(ts.timestamp()),
                        "price": round(exec_price, 2),
                        "amount": round(amount, 6),
                        "fee": round(fee, 4),
                    })

            elif signal == "sell" and risk_manager.open_trades:
                for trade in list(risk_manager.open_trades):
                    if not risk_manager.can_sell_trade(trade):
                        continue
                    exec_price = price * (1 - slippage)
                    fee = trade.amount * exec_price * fee_rate
                    risk_manager.capital -= fee
                    risk_manager.close_trade(trade, exec_price, "signal")
                    await websocket.send_json({
                        "type": "trade", "action": "sell",
                        "time": int(ts.timestamp()),
                        "price": round(exec_price, 2),
                        "pnl": round(trade.pnl, 4),
                        "fee": round(fee, 4),
                    })

            if (i - lookback) % 3 == 0:
                stats = risk_manager.get_stats()
                peak = max(equity_curve) if equity_curve else equity
                dd = (peak - equity) / peak if peak > 0 else 0
                await websocket.send_json({
                    "type": "update",
                    "candle_idx": i - lookback,
                    "time": int(ts.timestamp()),
                    "price": round(price, 2),
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

            if speed > 0:
                await asyncio.sleep(speed / 1000.0)

        # Final
        last_price = float(df.iloc[-1]["close"])
        for trade in list(risk_manager.open_trades):
            risk_manager.close_trade(trade, last_price, "sim_end")
        stats = risk_manager.get_stats()
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0])
        std = np.std(returns)
        sharpe = float((np.mean(returns) / std) * np.sqrt(365 * 24 * 4)) if std > 0 else 0
        neg_ret = returns[returns < 0]
        ds = np.std(neg_ret) if len(neg_ret) > 0 else 1
        sortino = float((np.mean(returns) / ds) * np.sqrt(365 * 24 * 4)) if ds > 0 else 0
        peak = max(equity_curve) if equity_curve else capital
        trough = min(equity_curve) if equity_curve else capital
        max_dd = (peak - trough) / peak if peak > 0 else 0

        btc_start = float(df.iloc[0]["close"])
        btc_end = float(df.iloc[-1]["close"])
        btc_roi = (btc_end - btc_start) / btc_start * 100

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
                "btc_buy_hold_roi": round(btc_roi, 2),
            },
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass


def start_server(port=8765, open_browser=True):
    if open_browser:
        import webbrowser
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    print(f"\n  Trading Bot Dashboard: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
