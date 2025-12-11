import csv
import math
import random

from matplotlib import pyplot as plt

from zad2.MiniMax import MinimaxPlayer
from zad2.two_player_games.player import Player
from zad2.two_player_games.games.dots_and_boxes import DotsAndBoxesState
from zad2.two_player_games.games.dots_and_boxes import DotsAndBoxesMove
from typing import cast, Optional


def single_game(starter: str, depthN: int, depthM: int, size: int, seed: Optional[int] = None):
    pA = Player('A')   # A = głębokość N
    pB = Player('B')   # B = głębokość M

    # kto zaczyna
    if starter == "A":
        state = DotsAndBoxesState(pA, pB, size=size)
    else:
        state = DotsAndBoxesState(pB, pA, size=size)

    rng = random.Random(seed) if seed is not None else random.Random()
    smart_players = {
        pA: MinimaxPlayer(depth=depthN, ran=rng),
        pB: MinimaxPlayer(depth=depthM, ran=rng),
    }

    while not state.is_finished(): # koniec gry
        current = state._current_player
        move = cast(DotsAndBoxesMove, smart_players[current].choose_move(state))
        state = state.make_move(move)

    scores = state.get_scores()
    return scores[pA], scores[pB]  # zawsze (A_score, B_score)




if __name__ == "__main__":

    # turniej
    size = 4
    max_depth = 4
    no_games = 3 # liczba partii rozegranych dla każdego seeda
    no_seeds = 10  # liczba różnych seedów

    A_wins_total = 0
    B_wins_total = 0
    draws_total = 0

    # macierze do heatmapy: zwycięstwa A i liczba gier
    winsA = [[0 for _ in range(max_depth)] for __ in range(max_depth)]
    totals = [[0 for _ in range(max_depth)] for __ in range(max_depth)] # bez remisów bo interesuje nas tylko przewaga

    # surowe wiersze do CSV: (A_depth, B_depth, A_score, B_score, seed)
    rows = []

    # turniej: dla każdej pary (N, M) po no_games gier na 10 seedach
    for N in range(1, max_depth + 1):
        for M in range(1, max_depth + 1):
            for s in range(no_seeds):
                seed_base = 100_000 * s  # rozdziela pule losowań między seedy
                for r in range(no_games):
                    # A zaczyna (A ma depth=N, B ma depth=M)
                    seed1 = seed_base + 10_000 + 97 * N + 131 * M + 17 * r
                    a1, b1 = single_game("A", N, M, size, seed=seed1)
                    rows.append((N, M, a1, b1, seed1))
                    if a1 > b1:
                        winsA[N - 1][M - 1] += 1
                        A_wins_total += 1
                        totals[N - 1][M - 1] += 1
                    elif b1 > a1:
                        B_wins_total += 1
                        totals[N - 1][M - 1] += 1
                    else:
                        draws_total += 1

                    # B zaczyna (DALEJ A ma depth=N, B ma depth=M, tylko kolejność ruchu inna - bo podejrzewam, że kolejność ruchu ma znaczenie)
                    seed2 = seed_base + 20_000 + 97 * N + 131 * M + 19 * r
                    a2, b2 = single_game("B", N, M, size, seed=seed2)
                    rows.append((N, M, a2, b2, seed2))
                    if a2 > b2:
                        winsA[N - 1][M - 1] += 1
                        A_wins_total += 1
                        totals[N - 1][M - 1] += 1
                    elif b2 > a2:
                        B_wins_total += 1
                        totals[N - 1][M - 1] += 1
                    else:
                        draws_total += 1


    # macierz % zwycięstw
    pctA = [[(winsA[i][j] / totals[i][j]) if totals[i][j] else math.nan
             for j in range(max_depth)] for i in range(max_depth)]

    # CSV: wszystkie gry z turnieju
    with open("results_games.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["A_depth", "B_depth", "A_score", "B_score", "seed"])
        w.writerows(rows)

    # CSV: macierz % zwycięstw
    pctN = [[(winsA[i][j] / totals[i][j]) if totals[i][j] else 0.0
             for j in range(max_depth)] for i in range(max_depth)]

    with open("pct_wins_depthN.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"M={d}" for d in range(1, max_depth + 1)])
        for i in range(max_depth):
            w.writerow([f"N={i + 1}"] + [f"{pctN[i][j]:.3f}" for j in range(max_depth)])

    plt.figure()
    plt.imshow(pctN, interpolation='nearest', aspect='auto')
    plt.title("% zwycięstw głębokości N nad M (uśrednione po obu starterach)")
    plt.xlabel("M (głębokość przeciwnika)")
    plt.ylabel("N (głębokość testowana)")
    plt.xticks(ticks=range(max_depth), labels=[str(d) for d in range(1, max_depth + 1)])
    plt.yticks(ticks=range(max_depth), labels=[str(d) for d in range(1, max_depth + 1)])
    plt.colorbar(label="P(win depth N)")
    for i in range(max_depth):
        for j in range(max_depth):
            plt.text(j, i, f"{pctN[i][j] * 100:.0f}%", ha="center", va="center", fontsize=9, color="black")
    plt.tight_layout()
    plt.show()

    print(f"Podsumowanie turnieju — A wygrane: {A_wins_total}, B wygrane: {B_wins_total}, remisy: {draws_total}")

    # ----------------------------------------------------------------
    # Czy mamy przewagę zaczynającego?
    size2 = 4
    max_depth2 = 4
    no_games2 = 6
    no_seeds2 = 10

    depths = list(range(1, max_depth2 + 1))
    starter_win_rate = []
    starter_rows = []

    for d in depths:
        wins = 0
        total = 0
        for s in range(no_seeds2):
            for r in range(no_games2):
                seed = s
                a_score, b_score = single_game("A", d, d, size2, seed=seed)  # A zaczyna
                starter_rows.append((d, a_score, b_score, seed))
                wins += 1 if a_score > b_score else 0
                total += 1
        starter_win_rate.append(wins / total if total else 0.0)


    # wykres: P(zaczynający wygrywa) vs głębokość
    plt.figure()
    plt.plot(depths, starter_win_rate, marker="o")
    plt.title("Przewaga gracza zaczynającego (A)")
    plt.xlabel("Głębokość przeszukiwania obu graczy")
    plt.ylabel("P(zaczynający wygrywa)")
    plt.ylim(0, 1)
    plt.xticks(depths, [str(d) for d in depths])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


