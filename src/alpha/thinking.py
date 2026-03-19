"""Thinking Engine — structured market reasoning framework.

Instead of blindly summing signal scores, the thinking engine:

1. OBSERVE  — What is the market doing right now?
2. ORIENT   — What regime/context are we in? What happened recently?
3. HYPOTHESIZE — Form a directional thesis with evidence
4. EVALUATE — How strong is the evidence? What contradicts it?
5. DECIDE   — Act only when conviction exceeds uncertainty
6. REFLECT  — After the trade, was the thesis correct? Learn.

This is an OODA-inspired decision loop adapted for trading.
Each step produces structured output that feeds the next.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from src.alpha.base import AlphaSignal, AlphaOutput
from src.alpha.combiner import SignalCombiner
from src.strategies.base import TradeSignal, Signal
from src.strategies.regime import detect_regime, MarketRegime
from src.utils.logger import setup_logger

logger = setup_logger("thinking")


@dataclass
class MarketObservation:
    """Step 1: What is the market doing right now?"""
    price: float
    price_change_1: float    # 1-candle return
    price_change_5: float    # 5-candle return
    price_change_20: float   # 20-candle return
    volatility: float        # current ATR / price
    volume_ratio: float      # current vol / avg vol
    trend_strength: float    # ADX or ER
    bb_position: float       # 0=lower band, 1=upper band


@dataclass
class MarketContext:
    """Step 2: What regime/context are we in?"""
    regime: MarketRegime
    regime_duration: int     # how many candles in this regime
    recent_trades_won: int
    recent_trades_lost: int
    recent_pnl: float
    drawdown: float
    is_recovering: bool      # are we bouncing back from a drawdown?


@dataclass
class Hypothesis:
    """Step 3: A directional thesis with supporting/contradicting evidence."""
    direction: str           # "long", "short", "neutral"
    thesis: str              # human-readable reasoning
    supporting: list[str]    # signals that support this view
    contradicting: list[str] # signals that contradict this view
    conviction: float        # 0.0 to 1.0
    edge_ratio: float        # supporting_weight / contradicting_weight


@dataclass
class Decision:
    """Step 5: Final decision with full reasoning chain."""
    action: Signal
    confidence: float
    reasoning: str
    observation: MarketObservation
    context: MarketContext
    hypothesis: Hypothesis
    should_size_up: bool = False   # high conviction = bigger position
    should_size_down: bool = False # contradictions present = smaller position


@dataclass
class TradeReflection:
    """Step 6: Post-trade analysis — was the thesis correct?"""
    trade_id: str
    hypothesis_direction: str
    actual_outcome: str      # "win" or "loss"
    thesis_correct: bool     # did the thesis predict the outcome?
    pnl: float
    lesson: str              # what to learn from this


class ThinkingEngine:
    """Structured reasoning framework for trading decisions.

    Wraps the alpha signal combiner with a reasoning layer that:
    - Forms hypotheses instead of just computing scores
    - Tracks whether hypotheses were correct (learning)
    - Adjusts conviction based on recent accuracy
    - Provides transparent reasoning for every decision
    """

    def __init__(self, config: dict):
        self.combiner = SignalCombiner(config)
        self.config = config
        params = config.get("strategy", {}).get("params", {})

        # Thinking parameters
        self.min_conviction = params.get("min_conviction", 0.4)
        self.min_edge_ratio = params.get("min_edge_ratio", 1.5)

        # State
        self._regime_history: list[MarketRegime] = []
        self._reflections: list[TradeReflection] = []
        self._pending_hypotheses: dict[str, Hypothesis] = {}  # trade_id -> hypothesis
        self._accuracy_window = 20
        self._consecutive_losses = 0
        self._recent_accuracy = 0.5  # start neutral

    def _observe(self, df: pd.DataFrame, ind: dict) -> MarketObservation:
        """Step 1: Gather raw market observations."""
        close = df["close"].values
        price = float(close[-1])
        n = len(close)

        price_change_1 = (close[-1] - close[-2]) / close[-2] if n >= 2 else 0
        price_change_5 = (close[-1] - close[-6]) / close[-6] if n >= 6 else 0
        price_change_20 = (close[-1] - close[-21]) / close[-21] if n >= 21 else 0

        atr = ind.get("atr_14", 0)
        volatility = atr / price if price > 0 and atr else 0

        vol = ind.get("volume", 0)
        vol_avg = ind.get("vol_sma_20", 1)
        volume_ratio = vol / vol_avg if vol_avg > 0 else 1

        er = ind.get("er_10", 0) or 0
        adx = ind.get("adx_14", 0) or 0
        trend_strength = max(er, adx / 50)  # normalize ADX to 0-1 range

        bbu = ind.get("bb_upper", price)
        bbl = ind.get("bb_lower", price)
        bb_range = (bbu - bbl) if bbu and bbl else 1
        bb_position = (price - (bbl or price)) / bb_range if bb_range > 0 else 0.5

        return MarketObservation(
            price=price,
            price_change_1=price_change_1,
            price_change_5=price_change_5,
            price_change_20=price_change_20,
            volatility=volatility,
            volume_ratio=volume_ratio,
            trend_strength=trend_strength,
            bb_position=bb_position,
        )

    def _orient(self, df: pd.DataFrame, obs: MarketObservation) -> MarketContext:
        """Step 2: Understand the current context and regime."""
        regime = detect_regime(df)
        self._regime_history.append(regime)

        # How long have we been in this regime?
        duration = 1
        for i in range(len(self._regime_history) - 2, -1, -1):
            if self._regime_history[i] == regime:
                duration += 1
            else:
                break

        # Recent trade performance
        recent = self._reflections[-self._accuracy_window:]
        wins = sum(1 for r in recent if r.actual_outcome == "win")
        losses = sum(1 for r in recent if r.actual_outcome == "loss")
        recent_pnl = sum(r.pnl for r in recent)

        # Drawdown state
        drawdown = 0
        if recent_pnl < 0:
            drawdown = abs(recent_pnl)

        is_recovering = (len(recent) >= 3 and
                         recent[-1].actual_outcome == "win" if recent else False)

        return MarketContext(
            regime=regime,
            regime_duration=duration,
            recent_trades_won=wins,
            recent_trades_lost=losses,
            recent_pnl=recent_pnl,
            drawdown=drawdown,
            is_recovering=is_recovering,
        )

    def _hypothesize(self, df: pd.DataFrame, ind: dict,
                     obs: MarketObservation, ctx: MarketContext) -> Hypothesis:
        """Step 3: Form a directional hypothesis with evidence."""
        # Run all alpha signals
        outputs: list[AlphaOutput] = []
        for signal in self.combiner.signals:
            try:
                output = signal.compute(df, ind)
                outputs.append(output)
            except Exception:
                continue

        # Categorize evidence
        supporting_long = []
        supporting_short = []
        contradicting_long = []
        contradicting_short = []

        long_weight = 0.0
        short_weight = 0.0

        for o in outputs:
            if o.score == 0 and o.confidence == 0:
                continue
            weight = abs(o.score * o.confidence)
            if o.score > 0.05:
                supporting_long.append(f"{o.name}:{o.score:+.2f}")
                long_weight += weight
            elif o.score < -0.05:
                supporting_short.append(f"{o.name}:{o.score:+.2f}")
                short_weight += weight

        # Multi-timeframe context (use price changes as proxy)
        if obs.price_change_20 > 0.03:  # 3%+ up over 20 candles
            supporting_long.append("trend_20:bullish")
            long_weight += 0.3
        elif obs.price_change_20 < -0.03:
            supporting_short.append("trend_20:bearish")
            short_weight += 0.3

        # Regime context
        if ctx.regime == MarketRegime.TRENDING_UP:
            supporting_long.append("regime:uptrend")
            long_weight += 0.2
        elif ctx.regime == MarketRegime.TRENDING_DOWN:
            supporting_short.append("regime:downtrend")
            short_weight += 0.2

        # Determine direction
        if long_weight > short_weight:
            direction = "long"
            supporting = supporting_long
            contradicting = supporting_short
            edge_ratio = long_weight / max(short_weight, 0.01)
            conviction = min(1.0, long_weight / max(long_weight + short_weight, 0.01))

            thesis = (f"Long bias: {len(supporting)} supporting signals "
                      f"(weight={long_weight:.2f}) vs {len(contradicting)} "
                      f"contradicting (weight={short_weight:.2f}). "
                      f"Regime={ctx.regime.value}, trend_20={obs.price_change_20:+.1%}")

        elif short_weight > long_weight:
            direction = "short"
            supporting = supporting_short
            contradicting = supporting_long
            edge_ratio = short_weight / max(long_weight, 0.01)
            conviction = min(1.0, short_weight / max(long_weight + short_weight, 0.01))

            thesis = (f"Short bias: {len(supporting)} supporting signals "
                      f"(weight={short_weight:.2f}) vs {len(contradicting)} "
                      f"contradicting (weight={long_weight:.2f}). "
                      f"Regime={ctx.regime.value}")
        else:
            direction = "neutral"
            supporting = []
            contradicting = []
            edge_ratio = 1.0
            conviction = 0.0
            thesis = "No clear directional edge"

        # Adjust conviction based on recent accuracy
        if self._recent_accuracy < 0.4:
            conviction *= 0.6  # less confident when we've been wrong
        elif self._recent_accuracy > 0.6:
            conviction *= 1.2  # more confident when we've been right
            conviction = min(1.0, conviction)

        # Reduce conviction during high volatility
        if obs.volatility > 0.02:  # ATR > 2% of price
            conviction *= 0.7

        # Reduce conviction after consecutive losses
        if self._consecutive_losses >= 3:
            conviction *= 0.5
            thesis += f" [CAUTION: {self._consecutive_losses} consecutive losses]"

        return Hypothesis(
            direction=direction,
            thesis=thesis,
            supporting=supporting,
            contradicting=contradicting,
            conviction=conviction,
            edge_ratio=edge_ratio,
        )

    def _decide(self, obs: MarketObservation, ctx: MarketContext,
                hyp: Hypothesis) -> Decision:
        """Step 5: Make a decision based on the hypothesis."""

        # Decision rules:
        # 1. Must have minimum conviction
        # 2. Must have minimum edge ratio (supporting >> contradicting)
        # 3. Regime must not strongly oppose the direction
        # 4. Not in a losing streak cooldown

        should_act = (
            hyp.conviction >= self.min_conviction and
            hyp.edge_ratio >= self.min_edge_ratio and
            self._consecutive_losses < 5
        )

        # Size adjustments
        should_size_up = hyp.conviction > 0.7 and hyp.edge_ratio > 2.5
        should_size_down = (
            len(hyp.contradicting) > 3 or
            obs.volatility > 0.015 or
            self._consecutive_losses >= 2
        )

        if should_act and hyp.direction == "long":
            action = Signal.BUY
            confidence = hyp.conviction
            reasoning = f"BUY: {hyp.thesis}"
        elif should_act and hyp.direction == "short":
            action = Signal.SELL
            confidence = hyp.conviction
            reasoning = f"SELL: {hyp.thesis}"
        else:
            action = Signal.HOLD
            confidence = 0.0
            if not should_act and hyp.direction != "neutral":
                reasoning = (f"HOLD (insufficient conviction): {hyp.thesis} "
                             f"[conv={hyp.conviction:.2f} edge={hyp.edge_ratio:.1f}]")
            else:
                reasoning = f"HOLD: {hyp.thesis}"

        return Decision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            observation=obs,
            context=ctx,
            hypothesis=hyp,
            should_size_up=should_size_up,
            should_size_down=should_size_down,
        )

    def reflect(self, trade_id: str, pnl: float, hypothesis: Hypothesis = None):
        """Step 6: Post-trade reflection — learn from the outcome."""
        hyp = hypothesis or self._pending_hypotheses.pop(trade_id, None)
        if hyp is None:
            return

        outcome = "win" if pnl > 0 else "loss"
        thesis_correct = (
            (hyp.direction == "long" and pnl > 0) or
            (hyp.direction == "short" and pnl < 0)
        )

        if outcome == "loss":
            self._consecutive_losses += 1
            lesson = (f"Loss on {hyp.direction} thesis. "
                      f"Contradicting signals were: {hyp.contradicting[:3]}. "
                      f"Consider: were these warnings ignored?")
        else:
            self._consecutive_losses = 0
            lesson = (f"Win on {hyp.direction} thesis. "
                      f"Key supporting signals: {hyp.supporting[:3]}")

        reflection = TradeReflection(
            trade_id=trade_id,
            hypothesis_direction=hyp.direction,
            actual_outcome=outcome,
            thesis_correct=thesis_correct,
            pnl=pnl,
            lesson=lesson,
        )
        self._reflections.append(reflection)

        # Update accuracy tracker
        recent = self._reflections[-self._accuracy_window:]
        correct = sum(1 for r in recent if r.thesis_correct)
        self._recent_accuracy = correct / len(recent) if recent else 0.5

        logger.info(
            f"REFLECT: {outcome} | thesis_correct={thesis_correct} | "
            f"accuracy={self._recent_accuracy:.0%} | {lesson[:80]}"
        )

    def think(self, df: pd.DataFrame) -> TradeSignal:
        """Main entry point: run the full thinking loop."""
        ind = self.combiner.build_indicator_snapshot(df)
        if not ind:
            return TradeSignal(Signal.HOLD, 0.0, "Insufficient data")

        # 1. OBSERVE
        obs = self._observe(df, ind)

        # 2. ORIENT
        ctx = self._orient(df, obs)

        # 3. HYPOTHESIZE
        hyp = self._hypothesize(df, ind, obs, ctx)

        # 5. DECIDE (step 4 is implicit in hypothesis evaluation)
        decision = self._decide(obs, ctx, hyp)

        # Build TradeSignal
        if decision.action == Signal.BUY:
            # Store hypothesis for post-trade reflection
            trade_key = f"think_{len(self._reflections)}"
            self._pending_hypotheses[trade_key] = hyp

        atr = ind.get("atr_14", 0)
        return TradeSignal(
            signal=decision.action,
            confidence=decision.confidence,
            reason=decision.reasoning,
            entry_price=obs.price,
            metadata={
                "regime": ctx.regime.value,
                "conviction": hyp.conviction,
                "edge_ratio": hyp.edge_ratio,
                "supporting": len(hyp.supporting),
                "contradicting": len(hyp.contradicting),
                "accuracy": self._recent_accuracy,
                "consecutive_losses": self._consecutive_losses,
                "size_up": decision.should_size_up,
                "size_down": decision.should_size_down,
                "atr": atr,
            },
        )

    def get_thinking_stats(self) -> dict:
        """Get statistics about the thinking process."""
        if not self._reflections:
            return {"total_reflections": 0, "accuracy": 0.5}

        total = len(self._reflections)
        correct = sum(1 for r in self._reflections if r.thesis_correct)
        wins = sum(1 for r in self._reflections if r.actual_outcome == "win")

        return {
            "total_reflections": total,
            "thesis_accuracy": correct / total,
            "win_rate": wins / total,
            "recent_accuracy": self._recent_accuracy,
            "consecutive_losses": self._consecutive_losses,
            "total_pnl": sum(r.pnl for r in self._reflections),
        }
