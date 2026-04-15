# cilpy/problem/benchmark_problems.py
import math
from typing import List, Tuple

from cilpy.problem import Problem, Evaluation

class Rastrigin(Problem[List[float], float]):
    """The Rastrigin Function.
    Highly multi-modal with a massive number of local minima.
    Global optimum is f(x) = 0 at x = (0, ..., 0).
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [-5.12] * dimension
        upper_bounds = [5.12] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Rastrigin")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        fitness = 10.0 * self.dimension + sum(x**2 - 10.0 * math.cos(2 * math.pi * x) for x in solution)
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class Ackley(Problem[List[float], float]):
    """The Ackley Function.
    Nearly flat outer region with a steep drop at the center.
    Global optimum is f(x) = 0 at x = (0, ..., 0).
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [-32.0] * dimension
        upper_bounds = [32.0] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Ackley")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        sum_sq = sum(x**2 for x in solution)
        sum_cos = sum(math.cos(2 * math.pi * x) for x in solution)
        
        term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / self.dimension))
        term2 = -math.exp(sum_cos / self.dimension)
        
        fitness = term1 + term2 + 20.0 + math.e
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class Griewank(Problem[List[float], float]):
    """The Griewank Function.
    Similar to Rastrigin, but waves are created by a product of cosines.
    Global optimum is f(x) = 0 at x = (0, ..., 0).
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [-600.0] * dimension
        upper_bounds = [600.0] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Griewank")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        sum_sq = sum(x**2 for x in solution) / 4000.0
        
        prod_cos = 1.0
        for i, x in enumerate(solution):
            # i + 1 because the mathematical formula relies on a 1-based index
            prod_cos *= math.cos(x / math.sqrt(i + 1))
            
        fitness = sum_sq - prod_cos + 1.0
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class Schwefel(Problem[List[float], float]):
    """The Schwefel Function.
    Deceptive landscape where the second-best minimum is far from the global minimum.
    Global optimum is f(x) = 0 at x = (420.9687, ..., 420.9687).
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [-500.0] * dimension
        upper_bounds = [500.0] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Schwefel")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        fitness = 418.9829 * self.dimension - sum(x * math.sin(math.sqrt(abs(x))) for x in solution)
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class Michalewicz(Problem[List[float], float]):
    """The Michalewicz Function.
    Steep valleys and ridges, punishing to discrete linear searchers.
    Global optimum varies based on dimension.
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [0.0] * dimension
        upper_bounds = [math.pi] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Michalewicz")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        m = 10.0 # Standard parameter for steepness
        fitness = 0.0
        for i, x in enumerate(solution):
            # i + 1 for 1-based mathematical indexing
            fitness -= math.sin(x) * (math.sin(((i + 1) * x**2) / math.pi)) ** int(2 * m)
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class Levy(Problem[List[float], float]):
    """The Levy Function.
    Highly multimodal with irregular trap placements.
    Global optimum is f(x) = 0 at x = (1, ..., 1).
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [-10.0] * dimension
        upper_bounds = [10.0] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Levy")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        # Transformation: w_i = 1 + (x_i - 1)/4
        w = [1.0 + (x - 1.0) / 4.0 for x in solution]
        
        term1 = math.sin(math.pi * w[0])**2
        term3 = (w[-1] - 1.0)**2 * (1.0 + math.sin(2.0 * math.pi * w[-1])**2)
        
        term2 = 0.0
        for i in range(self.dimension - 1):
            term2 += (w[i] - 1.0)**2 * (1.0 + 10.0 * math.sin(math.pi * w[i] + 1.0)**2)
            
        fitness = term1 + term2 + term3
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class Salomon(Problem[List[float], float]):
    """The Salomon Function.
    Features concentric hypersphere local minima rings. 
    Global optimum is f(x) = 0 at x = (0, ..., 0).
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [-100.0] * dimension
        upper_bounds = [100.0] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Salomon")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        sum_sq = sum(x**2 for x in solution)
        sqrt_sum_sq = math.sqrt(sum_sq)
        
        fitness = 1.0 - math.cos(2.0 * math.pi * sqrt_sum_sq) + 0.1 * sqrt_sum_sq
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False


class Alpine1(Problem[List[float], float]):
    """The Alpine No. 1 Function.
    Separable but highly rugged with sharp 'V-shaped' local minima.
    Global optimum is f(x) = 0 at x = (0, ..., 0).
    """
    def __init__(self, dimension: int = 30):
        lower_bounds = [-10.0] * dimension
        upper_bounds = [10.0] * dimension
        super().__init__(dimension=dimension, bounds=(lower_bounds, upper_bounds), name="Alpine1")

    def evaluate(self, solution: List[float]) -> Evaluation[float]:
        fitness = sum(abs(x * math.sin(x) + 0.1 * x) for x in solution)
        return Evaluation(fitness=fitness, constraints_inequality=[])

    def is_dynamic(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_multi_objective(self) -> bool:
        return False