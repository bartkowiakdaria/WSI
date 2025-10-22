import numpy as np
from solver import Solver
from genetic_algorithm import population, genetic_algorithm


'''
Klasa naszego algorytmu zgodnie z szablonem wymaganym w zadaniu (solver.py)
'''

class GASolver(Solver):
    def __init__(self, population_size, mu, pm, pc, t_max, genome_len, ran_seed=None):
        self.population_size = int(population_size)
        self.mu, self.pm, self.pc = int(mu), float(pm), float(pc)
        self.t_max, self.genome_len = int(t_max), int(genome_len)
        self.ran_seed = ran_seed

    def get_parameters(self):
        return {
            "population_size": self.population_size,
            "mu": self.mu,
            "pm": self.pm,
            "pc": self.pc,
            "t_max": self.t_max,
            "genome_len": self.genome_len,
            "ran_seed": self.ran_seed,
        }

    def solve(self, problem, x0=None, *_, **__):
        '''
        Rozwiązuje zadanie optymalizacji za pomocą algorytmu genetycznego.

        PARAMETRY:
        problem : funkcja celu,
        x0 : punkt początkowy (u nas populacja początkowa),
        *_ , **__ : nieużywane parametry zachowane dla zgodności interfejsu.

        ZWRACA:
        best_x : najlepszy znaleziony osobnik,
        info : słownik z danymi:
            - best_score: wartość funkcji celu dla best_x,
            - last_population: ostatnia populacja,
            - parameters: parametry algorytmu.
        '''
        ran = np.random.default_rng(self.ran_seed)
        P0 = x0 if x0 is not None else population(self.population_size, self.genome_len, ran) # jeśli populacja początkowa nie jest podana - generujemy ją
        q_fn = problem  # = funkcja celu (u nas: calc_target)
        best_x, best_score, lastP = genetic_algorithm(q_fn, P0, self.mu, self.pm, self.pc, self.t_max, ran)
        return best_x, {"best_score": best_score, "last_population": lastP, "parameters": self.get_parameters()}
