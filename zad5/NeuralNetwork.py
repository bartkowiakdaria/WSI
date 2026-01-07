from __future__ import annotations
from typing import Iterable, List, Tuple, Callable, Dict
import numpy as np



def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Pochodna sigmoidy liczona względem z
    """
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def drelu(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)

def loss_mse(A: np.ndarray, Y: np.ndarray) -> float:
    # L = (1/(2m)) * sum ||A - Y||^2
    m = Y.shape[1]
    return float(0.5 * np.sum((A - Y) ** 2) / m)

def dloss_mse(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # dL/dA = (A - Y) / m
    m = Y.shape[1]
    return (A - Y) / m


LOSSES: Dict[str, Tuple[Callable[[np.ndarray, np.ndarray], float],
                        Callable[[np.ndarray, np.ndarray], np.ndarray]]] = {"mse": (loss_mse, dloss_mse)}

ACTIVATIONS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray],
                             Callable[[np.ndarray], np.ndarray]]] = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, drelu),
}


class NeuralNetwork:
    def __init__(self, layer_sizes: Iterable[int], seed: int = 0, loss: str = "mse", activation: str = "sigmoid"):
        """
        layer_sizes: np. [n_in, 8, 4, n_out] - n_in cech wejściowych, warstwa ukryta z 8 neuronami,
        warstwa ukryta z 4 neuronami, n_out wyjść

        Tworzymy wagi dla każdej pary warstw:
        W[l] ma rozmiar (n_l, n_{l-1} + 1)
        +1 to kolumna na bias (bo do wejścia doklejamy wiersz jedynek).
        """
        self.layer_sizes = list(layer_sizes)
        if len(self.layer_sizes) < 2: # musimy mieć minimum warstwę wyjściową i wejściową
            raise ValueError("Brak wejścia i wyjścia")

        rng = np.random.default_rng(seed)

        self.weights: List[np.ndarray] = [] # macierz z wagami dla każdej warstwy

        # tworzenie wag:
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # inicjalizacja wag:
            # W (wagi): (n_out, n_in + 1) bo +1 to bias (zamiast trzymac osobno bias b - robimy z niego dodatkową kolumnę wag))
            W = rng.normal(0.0, 1.0, size=(n_out, n_in + 1)) * np.sqrt(1.0 / n_in) # skalujemy aby liczby nie były za duże
            self.weights.append(W)

        if loss not in LOSSES:
            raise ValueError("Nie ma takiej straty.")
        if activation not in ACTIVATIONS:
            raise ValueError(f"Nie ma takiej funkcji aktywacji.")


        self.loss_fn, self.dloss_fn = LOSSES[loss]
        self.act_f, self.act_df = ACTIVATIONS[activation]


    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Zwraca:
          activations: [A0, A1, ..., AL]
          pre_acts:    [Z1, Z2, ..., ZL]
        gdzie:
          A0 = X
          Z(l) = W(l) @ [A(l-1); 1]
          A(l) = sigmoid(Z(l))
        """
        n_in, m = X.shape # n_in - liczbe wejść (cech); m - liczba próbek
        if n_in != self.layer_sizes[0]:
            raise ValueError(f"Zły rozmiar wejścia.") # liczba cech wejściowych musi pasować do architektury sici neuropnowej

        activations: List[np.ndarray] = [X]   # A0 = wejście, A1 - wynik po 1 warstwie itp... (juz po f. aktywacji)
        pre_activ: List[np.ndarray] = []       # wyniki neuronu przed użyciem f. aktywacji

        A = X
        ones = np.ones((1, m))  # wiersz jedynek, do biasu

        for W in self.weights:
            # doklejamy bias: A_bias = [A; 1]
            A_bias = np.vstack([A, ones])

            Z = W @ A_bias # suma ważona
            pre_activ.append(Z)

            # aktywacja: A = activation_fun(Z)
            A = self.act_f(Z)
            activations.append(A)

        return activations, pre_activ

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Zwraca ostatnią aktywację A_L."""
        activations, _ = self.forward(X)
        return activations[-1]


    def loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Liczy bład sieci na danych X względem prawdziwej wartości - u:
        Zakładamy Y ma ten sam kształt co A_L: (n_out, m)
        """
        A_L = self.predict(X)
        if A_L.shape != Y.shape:
            raise ValueError(f"Zły rozmiar targetów.") # zabezpieczenie
        m = Y.shape[1] #lb próbek
        return self.loss_fn(A_L, Y) # liczymy mse



    def backward(self, X: np.ndarray, Y: np.ndarray) -> List[np.ndarray]:
        """
        PROPAGACJA WSTECZNA
        Liczy gradienty wag dW dla każdej warstwy
        Zwraca listę grads (jak zmienic wagi zeby zmniejszyć loss)

        Dla MSE:
          dL/dA_L = (A_L - Y) / m : pochodna straty po A_L
        """
        activations, pre_activ = self.forward(X)
        A_L = activations[-1]
        if A_L.shape != Y.shape:
            raise ValueError(f"Zły rozmiar targetów.")

        m = Y.shape[1] # lb próbek
        ones = np.ones((1, m)) # bias

        # start propagacji wstecznej: gradient po wyjściu
        dA = self.dloss_fn(A_L, Y)


        grads: List[np.ndarray] = [np.zeros_like(W) for W in self.weights] # każdy gradient musi mieć kształt jak odpowiadająca mu lista wag W

        # idziemy od ostatniej warstwy do pierwszej
        for l in reversed(range(len(self.weights))): #L-1, L-2, ..., 0
            W = self.weights[l]
            Z = pre_activ[l] # Z_l+1 bo pre_activ[0] = Z1
            A_prev = activations[l]  # A_l, bo activations[0]=A0

            # pochodna Z:
            # dZ = dA ∘ sigmoid'(Z)
            dZ = dA * self.act_df(Z)

            # doklejamy bias do A_prev, żeby policzyć dW w jednym kroku
            A_prev_bias = np.vstack([A_prev, ones])

            # gradient wag:
            # dW = dZ @ A_prev_bias^T
            grads[l] = dZ @ A_prev_bias.T

            # gradient dla poprzedniej warstwy:
            W_no_bias = W[:, :-1]
            dA = W_no_bias.T @ dZ  # dA idzie do następnej iteracji pętli

        return grads


    def train_SGD(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.1, epochs: int = 1000,
        batch_size: int | None = None, shuffle: bool = True) -> List[float]:
        """
        Uczenie STOCHASTIC GRADIENT DESCENT - aktualizowanie wag.
        - jeśli batch_size=None -> full batch
        - jeśli batch_size ustawione -> losowe mini-batche
        """
        n_in, m = X.shape
        if Y.shape[1] != m:
            raise ValueError("X i Y muszą mieć tyle samo próbek.")

        rng = np.random.default_rng(0)
        history: List[float] = [] # zapisuje błąd po każdej epoce

        for ep in range(1, epochs + 1): # epoka = jedno przejście przez cały zbiór
            if batch_size is None: # nie robimy mini-batch
                # pełny batch
                grads = self.backward(X, Y)
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * grads[i] # klasyczny gardient descent - idziemy kierunku spadku gradientu
            else:
                # mini-batch: tasujemy indeksy i idziemy po kawałkach a nie całości na raz
                idx = np.arange(m) # lista indeksów próbek
                if shuffle:
                    rng.shuffle(idx) # mieszamy kolejność próbek aby batche były różne w każdej epoce

                for start in range(0, m, batch_size): # dzielimy zbiór na kawałki - start = 0, batch_size, 2*batch_size, ...
                    batch_idx = idx[start:start + batch_size]
                    Xb = X[:, batch_idx] # wybieramy wylosowane kolumny
                    Yb = Y[:, batch_idx]

                    # to samo co wcześniej ale dla każdej porcji danych
                    grads = self.backward(Xb, Yb)
                    for i in range(len(self.weights)):
                        self.weights[i] -= learning_rate * grads[i]

            # zapis loss po epoce
            L = self.loss(X, Y)
            history.append(L)



        return history

    def train_Adam(self, X: np.ndarray, Y: np.ndarray, learning_rate: float = 0.001, beta1: float = 0.9,
                   beta2: float = 0.999, epsilon: float = 1e-8, epochs: int = 1000, batch_size: int | None = None,
                   shuffle: bool = True) -> List[float]:

        n_in, m = X.shape
        if Y.shape[1] != m:
            raise ValueError("X i Y muszą mieć tyle samo próbek.")

        rng = np.random.default_rng(0)

        # momenty
        M = [np.zeros_like(W) for W in self.weights]
        V = [np.zeros_like(W) for W in self.weights]

        history: List[float] = []

        for ep in range(1, epochs + 1):
            if batch_size is None:
                batches = [(X, Y)]
            else:
                idx = np.arange(m)
                if shuffle:
                    rng.shuffle(idx)
                batches = []
                for start in range(0, m, batch_size):
                    b = idx[start:start + batch_size]
                    batches.append((X[:, b], Y[:, b]))

            for Xb, Yb in batches:
                grads = self.backward(Xb, Yb)

                for i in range(len(self.weights)):
                    # aktualizacja momentów
                    M[i] = beta1 * M[i] + (1 - beta1) * grads[i]
                    V[i] = beta2 * V[i] + (1 - beta2) * (grads[i] ** 2)

                    # korekta biasu
                    m_hat = M[i] / (1 - beta1)
                    v_hat = V[i] / (1 - beta2)

                    # update wag
                    self.weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            history.append(self.loss(X, Y))

        return history


