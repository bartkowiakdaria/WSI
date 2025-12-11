from abc import ABC, abstractmethod


class Solver(ABC):
    """A solver. Parameters may be passed during initialization."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def fit(self, X, y):
        """
        A method that fits the solver to the given data.
        X is the dataset without the class attribute.
        y contains the class attribute for each sample from X.
        It may return anything.
        """
        ...

    def predict(self, X):
        """
        A method that returns predicted class for each row of X
        """
