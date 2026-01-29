from typing import Iterable, Optional

from zad2.two_player_games.move import Move
from zad2.two_player_games.player import Player


class State:
    """Immutable game state object."""
    def __init__(self, current_player, other_player) -> None:
        """ Initializes game state. Don't use directly, instead use the desired game.

        Args:
            state (State): initial game state
        """
        self._current_player = current_player
        self._other_player = other_player

    def get_moves(self) -> Iterable[Move]:
        """ Returns possible moves in the current state.

        Returns:
            Iterable[Move]: An iterable of game-specific Move objects
                that are valid for the current player in the current state.
        """
        raise NotImplementedError

    def get_current_player(self) -> Player:
        """ Returns the current player.

        Returns:
            Player: the object that represents the current player.
        """
        return self._current_player

    def make_move(self, move: Move) -> 'State':
        """ Makes move without changing this object - returns a new object with given state.

        Args:
            move (Move): the move to make.

        Returns:
            State: New state after the move
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """ Checks if the game is finished.

        Returns:
            bool: if the game is finished.
        """
        raise NotImplementedError

    def get_winner(self) -> Optional[Player]:
        """ Checks which player is the winner.

        Returns:
            Optional[Player]: Player object that represents the winner or None if not finished or draw.
        """
        raise NotImplementedError

    def get_players(self) -> Iterable[Player]:
        """ Retrieves players. Their order may not be consistent between different states.

        Returns:
            Iterable[Player]: the players in the game.
        """
        return [self._current_player, self._other_player]

    def __str__(self) -> str:
        """ Returns string representation of the current game's state.

        Returns:
            str: printable text represenation of the game's state.
        """
        raise NotImplementedError
