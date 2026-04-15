import sys
import os

from cilpy.problem.de_benchmark_funcs import (
    Rastrigin, Ackley, Griewank, Schwefel, Michalewicz, Levy, Salomon, Alpine1
)
from cilpy.solver.de.de_rand_2_bin import DE_bin
from cilpy.runner import ExperimentRunner

def main():
    problems = [
        Rastrigin(dimension=30),
        Ackley(dimension=30),
        Griewank(dimension=30),
        Schwefel(dimension=30),
        Michalewicz(dimension=30),
        Levy(dimension=30),
        Salomon(dimension=30),
        Alpine1(dimension=30)
    ]

    solver_configs = [
        {
            "class": DE_bin,
            "params": {
                "name": "DE_rand_2_bin",
                "population_size": 100,
                "crossover_rate": 0.8,
                "f_weight": 0.5,
            }
        }
    ]

    runner = ExperimentRunner(
        problems=problems,
        solver_configurations=solver_configs,
        num_runs=30,
        max_iterations=3000
    )
    runner.run_experiments()

if __name__ == "__main__":
    main()
