"""Evolution Loop — self-improving parameter optimization.

The evolver runs a genetic algorithm over strategy parameters:

1. POPULATION: N parameter sets (genomes)
2. FITNESS: Backtest each genome, score by Sortino ratio
3. SELECTION: Keep top performers
4. MUTATION: Randomly mutate parameters
5. CROSSOVER: Combine traits from two successful genomes
6. REPEAT: Each generation improves on the last

This replaces manual parameter tuning with automated evolution.
Run with: python -m src.evolution.evolver

The evolution loop produces a `data/evolved_params.json` file
with the best-performing parameter set found.
"""

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.bot import STRATEGY_MAP
from src.utils.logger import setup_logger

logger = setup_logger("evolution")


@dataclass
class Genome:
    """A set of strategy + risk parameters."""
    params: dict
    risk: dict
    fitness: float = 0.0
    generation: int = 0
    id: str = ""

    def to_config(self, base_config: dict) -> dict:
        config = copy.deepcopy(base_config)
        config["strategy"]["params"].update(self.params)
        config["risk"].update(self.risk)
        return config


# Parameter ranges for mutation
PARAM_RANGES = {
    # Strategy params
    "rsi_period": (7, 21, int),
    "rsi_oversold": (20, 40, int),
    "rsi_overbought": (60, 80, int),
    "macd_fast": (8, 16, int),
    "macd_slow": (20, 32, int),
    "macd_signal": (6, 12, int),
    "ema_period": (20, 100, int),
    "kama_period": (5, 20, int),
    "er_threshold": (0.15, 0.5, float),
    "combiner_threshold": (0.08, 0.25, float),
    "combiner_sell_threshold": (0.15, 0.40, float),
    "combiner_buy_persistence": (1, 4, int),
    "combiner_sell_persistence": (1, 5, int),
    "combiner_trend_bias": (0.0, 0.10, float),
    "ensemble_min_consensus": (0.3, 0.6, float),
    "min_conviction": (0.25, 0.6, float),
    "min_edge_ratio": (1.2, 3.0, float),
    # Risk params
    "stop_loss_pct": (0.02, 0.06, float),
    "take_profit_pct": (0.04, 0.15, float),
    "risk_per_trade_pct": (0.01, 0.05, float),
    "max_position_pct": (0.20, 0.50, float),
    "trailing_stop_pct": (0.015, 0.04, float),
    "min_hold_candles": (3, 15, int),
    "sl_cooldown_candles": (2, 8, int),
}

STRATEGY_PARAMS = {
    "rsi_period", "rsi_oversold", "rsi_overbought", "macd_fast", "macd_slow",
    "macd_signal", "ema_period", "kama_period", "er_threshold",
    "combiner_threshold", "combiner_sell_threshold", "combiner_buy_persistence",
    "combiner_sell_persistence", "combiner_trend_bias", "ensemble_min_consensus",
    "min_conviction", "min_edge_ratio",
}

RISK_PARAMS = {
    "stop_loss_pct", "take_profit_pct", "risk_per_trade_pct",
    "max_position_pct", "trailing_stop_pct", "min_hold_candles",
    "sl_cooldown_candles",
}


def random_genome(gen: int = 0) -> Genome:
    """Create a random genome within parameter ranges."""
    params = {}
    risk = {}
    for name, (lo, hi, typ) in PARAM_RANGES.items():
        val = random.uniform(lo, hi)
        val = typ(round(val, 4) if typ == float else int(val))
        if name in STRATEGY_PARAMS:
            params[name] = val
        else:
            risk[name] = val

    return Genome(params=params, risk=risk, generation=gen,
                  id=f"g{gen}_{random.randint(1000,9999)}")


def default_genome() -> Genome:
    """The current default parameters as a genome."""
    return Genome(
        params={
            "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "ema_period": 50, "kama_period": 10, "er_threshold": 0.3,
            "combiner_threshold": 0.12, "combiner_sell_threshold": 0.25,
            "combiner_buy_persistence": 2, "combiner_sell_persistence": 3,
            "combiner_trend_bias": 0.05, "ensemble_min_consensus": 0.4,
            "min_conviction": 0.4, "min_edge_ratio": 1.5,
        },
        risk={
            "stop_loss_pct": 0.03, "take_profit_pct": 0.08,
            "risk_per_trade_pct": 0.03, "max_position_pct": 0.40,
            "trailing_stop_pct": 0.025, "min_hold_candles": 6,
            "sl_cooldown_candles": 4,
        },
        generation=0, id="default",
    )


def mutate(genome: Genome, mutation_rate: float = 0.3, gen: int = 0) -> Genome:
    """Mutate a genome — small random changes to some parameters."""
    new_params = copy.deepcopy(genome.params)
    new_risk = copy.deepcopy(genome.risk)

    for name, (lo, hi, typ) in PARAM_RANGES.items():
        if random.random() > mutation_rate:
            continue

        target = new_params if name in STRATEGY_PARAMS else new_risk
        current = target.get(name, (lo + hi) / 2)

        # Small perturbation: +-15% of range
        range_size = hi - lo
        delta = random.gauss(0, range_size * 0.15)
        new_val = current + delta
        new_val = max(lo, min(hi, new_val))
        target[name] = typ(round(new_val, 4) if typ == float else int(new_val))

    return Genome(params=new_params, risk=new_risk, generation=gen,
                  id=f"g{gen}_{random.randint(1000,9999)}")


def crossover(a: Genome, b: Genome, gen: int = 0) -> Genome:
    """Combine two genomes — take some params from each parent."""
    new_params = {}
    new_risk = {}

    for name in PARAM_RANGES:
        if name in STRATEGY_PARAMS:
            src = a.params if random.random() < 0.5 else b.params
            new_params[name] = src.get(name, a.params.get(name))
        else:
            src = a.risk if random.random() < 0.5 else b.risk
            new_risk[name] = src.get(name, a.risk.get(name))

    return Genome(params=new_params, risk=new_risk, generation=gen,
                  id=f"g{gen}_{random.randint(1000,9999)}")


def evaluate_fitness(genome: Genome, df: pd.DataFrame,
                     base_config: dict, strategy_name: str = "alpha") -> float:
    """Backtest a genome and return fitness score (Sortino ratio)."""
    config = genome.to_config(base_config)
    config["strategy"]["name"] = strategy_name

    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if not strategy_cls:
        return -999

    try:
        strategy = strategy_cls(config)
        engine = BacktestEngine(config, strategy)
        result = engine.run(df)

        # Fitness = Sortino ratio (primary) + bonus for positive ROI + penalty for high DD
        sortino = result.get("sortino_ratio", 0)
        roi = result.get("roi_pct", 0)
        max_dd = result.get("max_drawdown_pct", 100)
        win_rate = result.get("win_rate", 0)
        trades = result.get("total_trades", 0)

        # Penalty for too few or too many trades
        trade_penalty = 0
        if trades < 5:
            trade_penalty = -2  # need enough trades to be meaningful
        elif trades > 200:
            trade_penalty = -1  # probably churning

        # Penalty for high drawdown
        dd_penalty = 0
        if max_dd > 20:
            dd_penalty = -(max_dd - 20) * 0.1

        fitness = sortino + roi * 0.02 + trade_penalty + dd_penalty

        return fitness

    except Exception as e:
        logger.debug(f"Fitness eval failed for {genome.id}: {e}")
        return -999


class EvolutionLoop:
    """Runs generations of parameter evolution."""

    def __init__(self, strategy_name: str = "alpha",
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.3,
                 elite_pct: float = 0.2):
        self.strategy_name = strategy_name
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_count = max(2, int(population_size * elite_pct))

        self.history: list[dict] = []

    def run(self, df: pd.DataFrame, base_config: dict) -> Genome:
        """Run the evolution loop. Returns the best genome found."""

        logger.info(
            f"Evolution starting: {self.generations} generations, "
            f"population={self.population_size}, strategy={self.strategy_name}"
        )

        # Initialize population: default + random genomes
        population = [default_genome()]
        for i in range(self.population_size - 1):
            population.append(random_genome(gen=0))

        best_ever = default_genome()
        best_ever.fitness = -999

        for gen in range(self.generations):
            t0 = time.time()

            # Evaluate fitness for each genome
            for genome in population:
                if genome.fitness == 0:  # not yet evaluated
                    genome.fitness = evaluate_fitness(
                        genome, df, base_config, self.strategy_name
                    )

            # Sort by fitness (descending)
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            gen_best = population[0]
            if gen_best.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(gen_best)

            gen_time = time.time() - t0
            avg_fitness = np.mean([g.fitness for g in population])

            gen_record = {
                "generation": gen,
                "best_fitness": gen_best.fitness,
                "avg_fitness": avg_fitness,
                "best_id": gen_best.id,
                "time_s": gen_time,
            }
            self.history.append(gen_record)

            logger.info(
                f"Gen {gen}: best={gen_best.fitness:.2f} avg={avg_fitness:.2f} "
                f"id={gen_best.id} ({gen_time:.1f}s)"
            )

            # Print top genome params
            if gen == self.generations - 1 or gen % 3 == 0:
                self._print_genome(gen_best, gen)

            # Selection + reproduction for next generation
            if gen < self.generations - 1:
                next_pop = []

                # Elitism: keep top N unchanged
                for i in range(self.elite_count):
                    elite = copy.deepcopy(population[i])
                    elite.generation = gen + 1
                    elite.fitness = 0  # re-evaluate (data might shift)
                    next_pop.append(elite)

                # Fill rest with mutations and crossovers
                while len(next_pop) < self.population_size:
                    if random.random() < 0.3:
                        # Crossover between two top performers
                        a = random.choice(population[:self.elite_count * 2])
                        b = random.choice(population[:self.elite_count * 2])
                        child = crossover(a, b, gen + 1)
                        child = mutate(child, self.mutation_rate * 0.5, gen + 1)
                    elif random.random() < 0.7:
                        # Mutate a top performer
                        parent = random.choice(population[:self.elite_count * 2])
                        child = mutate(parent, self.mutation_rate, gen + 1)
                    else:
                        # Fresh random genome (maintain diversity)
                        child = random_genome(gen + 1)

                    next_pop.append(child)

                population = next_pop

        logger.info(f"Evolution complete. Best fitness: {best_ever.fitness:.2f}")
        return best_ever

    def _print_genome(self, genome: Genome, gen: int):
        print(f"\n  === Gen {gen} Best (fitness={genome.fitness:.2f}) ===")
        print(f"  Strategy params: {json.dumps(genome.params, indent=2)}")
        print(f"  Risk params: {json.dumps(genome.risk, indent=2)}")

    def save_result(self, genome: Genome, output_dir: str = "data"):
        """Save the best genome to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "evolved_params.json")

        result = {
            "fitness": genome.fitness,
            "generation": genome.generation,
            "id": genome.id,
            "strategy_params": genome.params,
            "risk_params": genome.risk,
            "evolution_history": self.history,
        }

        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Best genome saved to {path}")
        return path


def fetch_evolution_data(timeframe="4h"):
    """Fetch data for evolution."""
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()
    all_data = []
    since = exchange.parse8601("2024-06-01T00:00:00Z")
    while True:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df[~df.index.duplicated(keep="first")]


def main():
    """Run the evolution loop from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Trading Bot Evolver")
    parser.add_argument("--strategy", default="alpha", help="Strategy to evolve")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=15)
    parser.add_argument("--timeframe", default="4h")
    args = parser.parse_args()

    print(f"\n  Trading Bot Evolution Loop")
    print(f"  Strategy: {args.strategy} | TF: {args.timeframe}")
    print(f"  Generations: {args.generations} | Population: {args.population}\n")

    base_config = {
        "trading": {
            "symbol": "BTC/USDT", "timeframe": args.timeframe,
            "initial_capital": 10000, "mode": "paper",
        },
        "strategy": {
            "name": args.strategy,
            "params": default_genome().params,
        },
        "risk": default_genome().risk,
    }

    print("Fetching data...", flush=True)
    df = fetch_evolution_data(args.timeframe)
    print(f"Loaded {len(df)} candles ({args.timeframe})\n", flush=True)

    evolver = EvolutionLoop(
        strategy_name=args.strategy,
        population_size=args.population,
        generations=args.generations,
    )

    best = evolver.run(df, base_config)
    path = evolver.save_result(best)

    print(f"\n  Best genome saved to: {path}")
    print(f"  Fitness: {best.fitness:.2f}")
    print(f"  Use these params in your config.yaml for improved performance.\n")


if __name__ == "__main__":
    main()
