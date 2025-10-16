import numpy as np

def calc_path(control: np.ndarray, time: int = 200, dt: float = 0.1):
    pathx = []
    pathz = []

    posx = np.zeros(control.shape[:-1])
    posz = np.zeros(control.shape[:-1])
    velx = np.zeros(control.shape[:-1])
    velz = np.zeros(control.shape[:-1])
    control = control.reshape(*control.shape[:-1], time, 2)

    t = 0
    pathx.append(posx)
    pathz.append(posz)

    while (posz >= 0).any():
        if t < time:
            cx = control[..., t, 0]
            cz = control[..., t, 1]
        else:
            cx = 0
            cz = 0

        velx = velx + (cx * 15 - 0.5 * velx) * dt
        velz = velz + (cz * 15 - 9.8 - 0.5 * velz) * dt

        velx = velx * (posz >= 0)
        velz = velz * (posz >= 0)

        posx = posx + velx * dt
        posz = posz + velz * dt

        pathx.append(posx)
        pathz.append(posz)
        t += 1

    return pathx, pathz

def calc_target(control: np.ndarray):
    # nasza funckja celu
    pathx, pathz = calc_path(control)
    return -(pathx[-1] - 350) ** 2

def population(population_size: int, genome_len: int, ran: np.random.Generator | None = None) -> np.ndarray:
    """
    Tworzy populację losowych osobników (bity 0/1).
    - population_size: liczba osobników w populacji
    - genome_len: długość chromosomu (liczba bitów na osobnika)
    - ran: random generator
    Zwraca ndarray shape = (pop_size, genome_len), dtype=uint8
    """
    if population_size <= 0:
        raise ValueError("wielkość populacji musi być dodatnia")
    if genome_len <= 0:
        raise ValueError("długość chromosomu musi być dodatnia")
    if ran is None:
        ran = np.random.default_rng()
    return ran.integers(0, 2, size=(population_size, genome_len), dtype=np.uint8)



def evaluate_population(population, q_fn):
    """
    Liczy wartości funkcji celu dla wszystkich osobników w populacji.
    population – tablica (N, L): N osobników po L bitów
    q_fn – funkcja celu (u nas będzie: calc_target), zwraca wyniki dla całej populacji
    Zwraca wektor wyników (N,)
    """
    scores = q_fn(population)
    return np.asarray(scores).reshape(-1)   # zwracamy jako 1D wektor - ujednolicenie postaci

def select_parents(population, scores, parents_number, ran=None):
    """
    Wybiera rodziców metodą ruletki.
    - population – tablica (N, L): populacja
    - scores – wektor wyników (N,) z evaluate_population
    - parents_number – ile osobników wylosować (rozmiar puli rodziców)
    - ran – opcjonalny generator losowy
    Zwraca tablicę (parents_number, L) – wybrane osobniki (z powtórzeniami)
    """
    if ran is None:
        ran = np.random.default_rng()

    # przesuwamy wyniki tak, żeby były nieujemne (bo funkcja celu (calc_target) zwraca też ujemne liczby)
    final_score = scores - scores.min()
    if np.all(final_score == 0):
        # jeśli wszyscy mają taki sam wynik — losujemy równomiernie
        probs = np.ones_like(final_score) / len(final_score)
    else:
        probs = final_score / final_score.sum()

    # losujemy indeksy z odpowiednimi prawdopodobieństwami, przy czym jeden osobnik może być wylosowany kilka razy
    indexes = ran.choice(len(population), size=parents_number, replace=True, p=probs)
    return population[indexes]

def crossover(parents, pc, ran=None):
    """
    Wykonuje krzyżowanie jednopunktowe.
    - parents: (M, L) – pula M rodziców o chromosomie długości L
    - pc: prawdopodobieństwo krzyżowania pary
    - ran: random generator
    Zwraca tablicę potomków (M, L).
    """
    if ran is None:
        ran = np.random.default_rng()
    M, L = parents.shape
    children = parents.copy()

    # dobieramy rodziców parami: (0,1), (2,3), ...
    for i in range(0, M - 1, 2):
        if ran.random() < pc: # z szansą pc krzyżujemy daną parę
            cut = ran.integers(1, L)  # punkt cięcia w [1..L-1]
            a = children[i].copy()
            b = children[i+1].copy()
            children[i, cut:], children[i+1, cut:] = b[cut:], a[cut:]
    return children

def mutation(population, pm, ran=None):
    """
    Mutacja bitowa populacji.
    population – tablica (N, L): populacja po krzyżowaniu (reprodukcji)
    pm – prawdopodobieństwo mutacji pojedynczego bitu
    ran – opcjonalny generator losowy
    Zwraca nową populację (N, L) po mutacji
    """
    if ran is None:
        ran = np.random.default_rng()

    # tworzymy tablicę True/False - porównujemy tablicę losowych liczb z zakresu [0, 1) o takim samym rozmiarze jak populacja
    # z wartością prawdopodobieństwa populacji (pm)
    mutation_indexes = ran.random(population.shape) < pm

    # odwracamy bity tam, gdzie mutation_indexes == True
    mutated = population.copy()
    mutated[mutation_indexes] = 1 - mutated[mutation_indexes]

    return mutated

def best_of(population, scores):
    """
    Zwraca najlepszego osobnika z populacji i jego wynik funkcji celu.
    populacja – tablica (N, L): populacja
    scores – wektor wyników (N,)
    Zwraca (najlepszy_osobnik, jego_wynik)
    """
    j = int(np.argmax(scores))           # indeks osobnika z najwyższym wynikiem
    return population[j].copy(), float(scores[j])


# GA: zgodnie z pseudo-kodem z wykładu
def genetic_algorithm(q_fn, P0, mu, pm, pc, t_max, ran=None):
    """
    q_fn – funkcja celu, np. calc_target
    P0 – populacja początkowa (N, L)
    mu – liczba osobników wybieranych do reprodukcji (M w puli rodziców)
    pm – prawdopodobieństwo mutacji genu
    pc – prawdopodobieństwo krzyżowania pary
    t_max – liczba pokoleń
    ran - generator losowy
    """
    if ran is None:
        ran = np.random.default_rng()

    t = 0
    P = P0.copy()
    o = evaluate_population(P, q_fn)
    x_hat, o_hat = best_of(P, o)  # najlepszy globalny

    while t < t_max:
        # R ← reprodukcja
        # (upewniamy się, że μ jest parzyste do krzyżowania)
        if mu % 2 == 1:
            mu += 1
        R = select_parents(P, o, mu, ran)

        # M ← krzyżowanie i mutacja
        M = crossover(R, pc, ran)
        M = mutation(M, pm, ran)

        # o ← ocena nowej populacji
        o = evaluate_population(M, q_fn)

        # x_t*, o_t* ← znajdujemy najlepszego i podmieniamy
        x_t, o_t = best_of(M, o)
        if o_t > o_hat:
            o_hat = o_t
            x_hat = x_t

        # dalej będziemy działać na populacji zmutowanej
        P = M
        t += 1

    return x_hat, o_hat, P
