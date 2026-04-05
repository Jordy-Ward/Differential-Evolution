# cilpy/solver/de.py
import math
import random
from typing import List, Tuple

from cilpy.problem import Problem, Evaluation
from cilpy.solver import Solver


class DE(Solver[List[float], float]):
    """
    A canonical Differential Evolution (DE) solver for single-objective
    optimization.

    This is a `DE/rand/2/PCX` implementation. It creates a trial vector for
    each member of the population and replaces the member if the trial vector
    has better or equal fitness.

    The algorithm uses:
    - `rand` strategy for selecting vectors for mutation.
    - `2` difference vectors in the mutation step.
    - `PCX` cross over
    """

    def __init__(
        self,
        problem: Problem[List[float], float],
        name: str,
        population_size: int,
        crossover_rate: float,
        f_weight: float,
        **kwargs,
    ):
        """
        Initializes the Differential Evolution solver.

        Args:
            problem: The optimization problem to solve.
            name: the name of the solver
            population_size: The number of individuals (ns) in the population.
            crossover_rate: The crossover probability (CR) in the range [0, 1].
            f_weight: The differential weight (F) for mutation, typically in the
                range [0, 2].
            **kwargs: Additional keyword arguments (not used in this canonical
                DE).
        """
        super().__init__(problem, name)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.f_weight = f_weight

        # Initialize population
        self.population = self._initialize_population()
        self.evaluations = [self.problem.evaluate(i) for i in self.population]

    def _initialize_population(self) -> List[List[float]]:
        """Creates the initial population with random solutions."""
        population = []
        lower_bounds, upper_bounds = self.problem.bounds
        for _ in range(self.population_size):
            individual = [
                random.uniform(lower_bounds[i], upper_bounds[i])
                for i in range(self.problem.dimension)
            ]
            population.append(individual)
        return population

    def step(self) -> None:
        """Performs one generation of the Differential Evolution algorithm."""
        lower_bounds, upper_bounds = self.problem.bounds

        for i in range(self.population_size):
            target_vector = self.population[i]
            target_eval = self.evaluations[i]

            # 1. Mutation (Create Donor Vector) - DE/rand/2
            # Select five distinct individuals other than the target
            indices = list(range(self.population_size))
            indices.remove(i)
            r1, r2, r3, r4, r5 = random.sample(indices, 5)

            x_r1 = self.population[r1]
            x_r2 = self.population[r2]
            x_r3 = self.population[r3]
            x_r4 = self.population[r4]
            x_r5 = self.population[r5]
            

            donor_vector = [
                x_r1[j] + self.f_weight * (x_r2[j] - x_r3[j]) + self.f_weight * (x_r4[j] - x_r5[j])
                for j in range(self.problem.dimension)
            ]

            # 2. Recombination (Create Trial Vector) - PCX
            # Parent, mutant, and a random selected vector != to parent
            x3 = x_r1
            dim = self.problem.dimension
            sigma_xi = 0.1
            sigma_eta = 0.1
            
            # calculate distance between target vector and mutation vector (donor)
            d = [donor_vector[j] - target_vector[j] for j in range(dim)] # cycle over every dmension for this target vector. Where the bell curv will gen new off spring
            # in pcx the center of the bell is dropped right on the parent
            m = target_vector #center gravity. Mean normal distr. Where the algo aims throwing darts. Spreads them out along d.
            
            # calculate the euclidean length magnitude of d
            d_mag_sq = sum(val**2 for val in d)
            d_mag = math.sqrt(d_mag_sq)
            
            # calculate the orthogonal distance D from x3 to the primary d vector
            v = [x3[j] - target_vector[j] for j in range(dim)]
            
            if d_mag_sq > 1e-10: #if parent and mutant are very close it can cause a div zero error
                v_dot_d = sum(v[j] * d[j] for j in range(dim))
                proj_v_on_d = [(v_dot_d / d_mag_sq) * d[j] for j in range(dim)]
                orth_v = [v[j] - proj_v_on_d[j] for j in range(dim)]
                D = math.sqrt(sum(val**2 for val in orth_v)) # how fat to make the bell curve
            else:
                D = 0.0
                
            # create the child vectors. The trial vectors using normal distribution
            trial_vector = [0.0] * dim
            xi = random.gauss(0, sigma_xi) # primary axis noise. How far up down the vector d to move
            
            # generate random gauss vector. How far left perp to move
            # so we need to extract the perp comp again, else we would add extra up down movement, already
            # handled in the xi
            z = [random.gauss(0, 1) for _ in range(dim)]
            if d_mag_sq > 1e-10:
                z_dot_d = sum(z[j] * d[j] for j in range(dim))
                proj_z_on_d = [(z_dot_d / d_mag_sq) * d[j] for j in range(dim)]
                z_orth = [z[j] - proj_z_on_d[j] for j in range(dim)]
            else:
                z_orth = z
                
            # assemble the off spring
            for j in range(dim):
                trial_vector[j] = m[j] + (xi * d[j]) + (z_orth[j] * sigma_eta * D)
            
            # Ensure trial vector is within bounds
            for j in range(self.problem.dimension):
                trial_vector[j] = max(
                    lower_bounds[j], min(trial_vector[j], upper_bounds[j])
                )
                
            
            trial_eval = self.problem.evaluate(trial_vector)

            # If the trial vector is better or equal, it replaces the target
            # vector
            if not self.comparator.is_better(target_eval, trial_eval):
                self.population[i] = trial_vector
                self.evaluations[i] = trial_eval

    def get_result(self) -> List[Tuple[List[float], Evaluation[float]]]:
        """Returns the best solution found in the current population."""
        best_idx = 0
        for i in range(1, self.population_size):
            if self.comparator.is_better(
                self.evaluations[i], self.evaluations[best_idx]
            ):
                best_idx = i

        best_solution = self.population[best_idx]
        best_evaluation = self.evaluations[best_idx]
        return [(best_solution, best_evaluation)]

    def get_population(self) -> List[List[float]]:
        """
        Returns the entire current DE population.

        This overrides the default Solver method to provide statistics on all
        individuals in the current generation.

        Returns:
            A list containing every individual.
        """
        return self.population

    def get_population_evaluations(self) -> List[Evaluation[float]]:
        """
        Returns the evaluations of the entire current DE population.

        This overrides the default Solver method to provide statistics on all
        individuals in the current generation.

        Returns:
            A list containing the `Evaluation` object for every individual.
        """
        return self.evaluations
