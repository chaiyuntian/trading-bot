"""Tests for walk-forward and Monte Carlo validation."""

import numpy as np
from src.backtesting.monte_carlo import MonteCarloSimulator


def test_monte_carlo_profitable_strategy():
    """Strategy with positive expectancy should be robust."""
    np.random.seed(42)
    # Simulate a strategy with slight edge (55% win rate, 1:1 R/R)
    pnls = []
    for _ in range(100):
        if np.random.random() < 0.55:
            pnls.append(2.0)  # Win $2
        else:
            pnls.append(-2.0)  # Lose $2

    mc = MonteCarloSimulator(initial_capital=100, n_simulations=500)
    result = mc.run(pnls)

    assert result["valid"]
    assert result["profitable_pct"] > 0.5


def test_monte_carlo_losing_strategy():
    """Strategy with negative expectancy should fail."""
    np.random.seed(42)
    pnls = []
    for _ in range(100):
        if np.random.random() < 0.35:
            pnls.append(1.5)
        else:
            pnls.append(-2.0)

    mc = MonteCarloSimulator(initial_capital=100, n_simulations=500)
    result = mc.run(pnls)

    assert result["valid"]
    assert result["profitable_pct"] < 0.5


def test_monte_carlo_insufficient_trades():
    mc = MonteCarloSimulator()
    result = mc.run([1.0, -0.5])
    assert not result["valid"]
