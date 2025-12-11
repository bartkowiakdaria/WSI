from typing import Iterable, Optional
from zad2.two_player_games.move import Move
from zad2.two_player_games.player import Player
from zad2.two_player_games.state import State


class Game:
    """Game interface."""
    def __init__(self, state: State):
        """ Initializes game. Don't use directly, instead use the desired game.

        Args:
            state (State): initial game state
        """
        self.state = state

    def get_moves(self) -> Iterable[Move]:
        """ Returns possible moves in the current state.

        Returns:
            Iterable[Move]: An iterable of game-specific Move objects
                that are valid for the current player in the current state.
        """
        return self.state.get_moves()

    def get_current_player(self) -> Player:
        """ Returns the current player.

        Returns:
            Player: the object that represents the current player.
        """
        return self.state.get_current_player()

    def make_move(self, move: Move):
        """ Makes move and changes the underlying state of the game.

        Args:
            move (Move): the move to make.
        """
        self.state = self.state.make_move(move)

    def is_finished(self) -> bool:
        """ Checks if the game is finished.

        Returns:
            bool: if the game is finished.
        """
        return self.state.is_finished()

    def get_winner(self) -> Optional[Player]:
        """ Checks which player is the winner.

        Returns:
            Optional[Player]: Player object that represents the winner or None if not finished or draw.
        """
        return self.state.get_winner()

    def get_players(self) -> Iterable[Player]:
        """ Retrieves players. Their order may not be consistent between different states.

        Returns:
            Iterable[Player]: the players in the game.
        """
        return self.state.get_players()

    def __str__(self) -> str:
        """ Returns string representation of the current game's state.

        Returns:
            str: printable text represenation of the game's state.
        """
        return str(self.state)
