# Crypto Trading Bot

自动化加密货币交易机器人，支持多种策略、回测引擎和风险管理。

> **⚠️ 风险警告**: 交易涉及资金损失风险。本项目仅供学习和研究用途。请勿投入无法承受损失的资金。过往回测表现不代表未来收益。

## 功能特性

- **4种交易策略**: RSI+MACD组合、均值回归、网格交易、DCA动量
- **风险管理**: 仓位控制、止损/止盈、最大回撤限制、每日亏损限制
- **回测引擎**: 历史数据回测，考虑手续费和滑点
- **模拟交易**: Paper trading 模式，不需要真金白银
- **多交易所**: 通过 ccxt 支持 Binance、Bybit、KuCoin 等 100+ 交易所

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cp config/config.example.yaml config/config.yaml
# 编辑 config/config.yaml，填入你的设置

# 3. 回测（推荐先回测！）
python -m src.main --backtest

# 4. 回测所有策略对比
python -m src.main --backtest-all

# 5. 模拟交易（Paper Trading）
python -m src.main

# 6. 实盘交易（谨慎！）
# 在 config.yaml 中设置 mode: "live" 并填入 API 密钥
python -m src.main
```

## 策略说明

| 策略 | 适用场景 | 原理 |
|------|---------|------|
| `rsi_macd` | 趋势+动量 | RSI超卖回升 + MACD金叉 + EMA趋势过滤 |
| `mean_reversion` | 震荡市 | 价格触及布林带下轨 + RSI超卖 = 买入 |
| `grid` | 横盘整理 | 在价格区间内设置网格，低买高卖 |
| `dca_momentum` | 长期积累 | 定投 + 动量过滤，只在有利时机买入 |

## 风险管理参数

```yaml
risk:
  max_position_pct: 0.30    # 单次最大仓位：30%
  stop_loss_pct: 0.03       # 止损：3%
  take_profit_pct: 0.06     # 止盈：6% (2:1 盈亏比)
  max_daily_loss_pct: 0.05  # 日最大亏损：5% → 停止交易
  max_drawdown_pct: 0.25    # 最大回撤：25% → 停止交易
  risk_per_trade_pct: 0.02  # 每笔风险：2%
```

## 项目结构

```
├── config/
│   └── config.example.yaml   # 配置模板
├── src/
│   ├── main.py                # 入口
│   ├── bot.py                 # 交易机器人主逻辑
│   ├── exchange/client.py     # 交易所连接 (ccxt)
│   ├── strategies/            # 交易策略
│   │   ├── rsi_macd.py        # RSI + MACD 策略
│   │   ├── mean_reversion.py  # 均值回归策略
│   │   ├── grid_trading.py    # 网格交易策略
│   │   └── dca_momentum.py    # DCA 动量策略
│   ├── risk/manager.py        # 风险管理
│   ├── indicators/technical.py # 技术指标
│   └── backtesting/engine.py  # 回测引擎
├── tests/                     # 单元测试
├── logs/                      # 运行日志
└── data/                      # 数据存储
```

## 使用建议

1. **先回测** → 用 `--backtest-all` 找到最适合当前市场的策略
2. **再模拟** → Paper trading 至少运行 1-2 周观察
3. **小额实盘** → 确认策略可行后，用小额资金测试
4. **持续监控** → 自动化≠无人值守，定期检查和调整
