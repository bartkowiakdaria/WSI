import pandas as pd
import numpy as np
from genetic_algorithm import calc_target
from ga_solver import GASolver
import matplotlib.pyplot as plt

TIME = 200
GENOME_LEN = TIME * 2

# --- TRYWIALNE ROZWIĄZANIE ---
# rakieta nic nie robi, więc ląduje od razu w punkcie startowym
zeros = np.zeros((1, GENOME_LEN), dtype=np.uint8)
score_zeros = calc_target(zeros)[0]
print(f"Trywialne rozwiązanie (oba silniki wyłączone): {score_zeros}")
zeros = np.ones((1, GENOME_LEN), dtype=np.uint8)
score_ones = calc_target(zeros)[0]
print(f"Trywialne rozwiązanie (oba silniki włączone): {score_ones}")

# --- GÓRNE OGRANICZENIE WYNIKU ---
# najlepszy możliwy wynik to 0, czyli idealne lądowanie 350 m od punktu startowego
print("Górne ograniczenie wyniku:", 0)

# --- LOSOWY WYNIK ---
# wynik bez użycia naszego algorytmu genetycznego
random_rocket = np.random.default_rng(0).integers(0, 2, size=(1, GENOME_LEN))
score_random = calc_target(random_rocket)[0]
print(f"Losowe rozwiązanie: {score_random:.3f}")

# --- ALGORYTM GENETYCZNY (własny zestaw hiperparametrów) ---
mean = 0 # średnia wartość wyniku dla różnych seed
for i in range(10): # testujemy dla 10 losowych random_seed
    solver = GASolver(
        population_size=200,
        mu=200,
        pm=0.005,
        pc=0.8,
        t_max=100,
        genome_len=GENOME_LEN,
        ran_seed = np.random.randint(1, 100),
    )
    best_x, info = solver.solve(problem=calc_target)
    best_score = info["best_score"]
    mean += best_score
    print(f"Najlepszy wynik GA z własnymi parametrami dla ziarna {solver.ran_seed}: {best_score:.5f}")
print("Średni wynik:", mean/10)
# --- EKSPERYMENTOWANIE Z PRAWDOPODOBIEŃSTWEM MUTACJI ----
print("Testowanie hiperparametru pm:")
# Ustalone ziarno generatora
pm_values = [0.00005, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
results = []
for pm in pm_values:
    solver = GASolver(
        population_size=200,
        mu=200,
        pm=pm,
        pc=0.8,
        t_max=100,
        genome_len=GENOME_LEN,
        ran_seed=13,
    )
    _, info = solver.solve(problem=calc_target)
    results.append({
        "pm": pm,
        "best_score": info["best_score"]
    })
df = pd.DataFrame(results)
print("\nTabela wyników dla ustalonego ziarna generatora:\n", df)


# Średnia dla 10 różnych ziarn generatora
seeds = np.random.randint(1, 1000, size=10)
print("Użyte ziarna generatora:", seeds)
results = []
for pm in pm_values:
    scores = []
    for seed in seeds:
        solver = GASolver(
            population_size=200,
            mu=200,
            pm=pm,
            pc=0.8,
            t_max=100,
            genome_len=GENOME_LEN,
            ran_seed=seed  # różne seedy dla losowości
        )
        best_x, info = solver.solve(problem=calc_target)
        score = info["best_score"]
        scores.append(score)

    mean_score = np.mean(scores)
    results.append({"pm": pm, "mean_score": mean_score})

df = pd.DataFrame(results)
print("\nTabela wyników dla 10 różnych ziaren generatora:\n", df)

plt.figure(figsize=(8,5))
plt.plot(df["pm"], df["mean_score"], marker='o')
plt.xscale("log")
plt.xlabel("Prawdopodobieństwo mutacji (pm)")
plt.ylabel("Średni najlepszy wynik (funkcja celu)")
plt.title("Wpływ pm na wynik algorytmu genetycznego")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()