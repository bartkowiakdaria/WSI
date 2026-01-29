import math
import random
from typing import Optional, Tuple, List

from zad2.two_player_games.games.dots_and_boxes import DotsAndBoxesState
from zad2.two_player_games.state import State
from zad2.two_player_games.move import Move
from zad2.two_player_games.player import Player


class MinimaxPlayer:
    """
    Gracz oparty o minimax z obcinaniem alfa-beta.
    - depth: maksymalna głębokość przeszukiwania (liczba kroków do przodu)
    - ran: generator losowy do tie-breaku między równie dobrymi ruchami
    """

    def __init__(self, depth: int, ran: Optional[random.Random] = None):
        self.depth = depth
        self.ran = ran if ran is not None else random.Random()

    def choose_move(self, state: DotsAndBoxesState) -> Move:
        """
        Zwraca ruch, który aktualny gracz powinien teraz wykonać (najlepszy możliwy dla niego ruch)
        """
        player = state._current_player  # gracz, dla którego optymalizujemy
        value, best_move = self.minimax(
            state=state,
            depth=self.depth,
            alpha=-math.inf,
            beta=math.inf,
            player=player
        )
        return best_move

    def evaluate(self, state: DotsAndBoxesState, player: Player) -> float:
        """
        Heurystyka - różnica punktowa.
        """
        scores = state.get_scores()  # wyniki graczy = {Player: int}
        # ustalamy przeciwnika
        players = list(state.get_players())  # [current, other]
        if players[0] is player:
            opp = players[1]
        else:
            opp = players[0]

        my_score = scores[player]
        opp_score = scores[opp]
        return my_score - opp_score

    def minimax(self, state: DotsAndBoxesState, depth: int, alpha: float, beta: float, player: Player) -> Tuple[float, Optional[Move]]:
        """
        ALgorytm MiniMax z obcinaniem alfa-beta.
        Zwraca (ocena (heurystyka), ruch).
        Jeśli depth==0 albo stan końcowy -> ruch = None.
        """

        # warunki stopu

        # stan końcowy lub nie możemy dalej przeszukiwać
        if depth == 0 or state.is_finished():
            return self.evaluate(state, player), None

        legal_moves: List[Move] = list(state.get_moves())
        # brak ruchów = koniec dla nas
        if not legal_moves:
            return self.evaluate(state, player), None

        # sprawdzamy, czy aktualny gracz w tym stanie to ten, dla którego optymalizujemy
        maximizing = (state._current_player is player)

        if maximizing:
            best_value = -math.inf
            best_moves: List[Move] = [] # pusta lista z najlepszymi ruchami -> później wypełniana

            for move in legal_moves:
                next_state = state.make_move(move)

                # działamy rekurencyjnie - przeszukujemy coraz to głebsze poziomy
                value, _ = self.minimax(
                    state=next_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    player=player
                )

                if value > best_value: # lepszy ruch niż poprzedni -> podmieniamy
                    best_value = value
                    best_moves = [move]
                elif value == best_value: # równie dobry ruch -> dodajemy do listy
                    best_moves.append(move)

                # aktualizujemy alpha i sprawdzamy, czy zatrzymać dalsze przeszukiwanie
                alpha = max(alpha, best_value) # jeśli znaleźliśmy ruch lepszy niż wszystko dotąd, to podnosimy nasze minimum gwarantowane.
                if beta <= alpha:
                    break  # obcięcie beta (przeciwnik nie dopuści lepszego wyniku)

            # losowy tie-break między ruchami równie dobrymi
            chosen_move = self.ran.choice(best_moves)
            return best_value, chosen_move

        else:
            # węzeł minimalizujący analogicznie: zakładamy, że przeciwnik gra perfekcyjnie przeciwko nam
            worst_value = math.inf
            worst_moves: List[Move] = []

            for move in legal_moves:
                next_state = state.make_move(move)

                val, _ = self.minimax(
                    state=next_state,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    player=player
                )

                if val < worst_value:
                    worst_value = val
                    worst_moves = [move]
                elif val == worst_value:
                    worst_moves.append(move)

                # aktualizacja beta + sprawdzenie czy szukamy dalej
                beta = min(beta, worst_value)
                if beta <= alpha:
                    break  # obcięcie alpha (my nie pozwolimy na gorszy wynik)

            chosen_move = self.ran.choice(worst_moves)
            return worst_value, chosen_move
