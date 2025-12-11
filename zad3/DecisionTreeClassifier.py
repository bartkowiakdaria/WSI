import math
from zad3.solver import Solver

class Node:
    def __init__(self, leaf, attribute=None, label=None):
        self.leaf = leaf  # czy jest liściem
        self.attribute = attribute  # wartość
        self.label = label  # jaką klase wybierzemy na tym poziomie (większościową)
        self.children = {}  # wartość atrybutu ->  poddrzewo


class DecisionTreeClassifier(Solver):

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None

    def get_parameters(self):
        return self.max_depth, self.tree

    @staticmethod
    def entropy(labels):
        '''
        Entropia zbioru uczącego zgodnie ze wzorem z wykładu.
        I(U) = - sum_i f_i * ln(f_i), gdzie f_i to częstość i-tej klasy w labels.
        '''
        n = len(labels)
        if n == 0:
            return 0.0

        unique_classes = list(set(labels))  # unikalne wartości y
        entropy = 0.0
        for class_ in unique_classes:
            count = labels.count(class_)
            p = float(count) / n  # częstość klasy
            entropy -= p * math.log(p)
        return entropy

    def inf_gain(self, X, y, attribute):
        """
        InfGain(d, U) = I(U) - Inf(d, U), gdzie:
        Inf(d, U) = sum_v (|U_{d=v}| / |U|) * I(U_{d=v})
        """
        n = len(X)
        if n == 0:
            return 0.0

        entropy_before = self.entropy(y)
        values = list(set(row[attribute] for row in X))  # wyliczamy wszystkie różne wartości atrybutu d

        info_after = 0.0
        for value in values:
            #podzbiory X i y
            X_sub = []
            y_sub = []
            for row, label in zip(X, y):
                if row[attribute] == value:
                    X_sub.append(row)
                    y_sub.append(label)

            if len(X_sub) == 0:
                continue

            weight = float(len(X_sub)) / n
            info_after += weight * self.entropy(y_sub)

        return entropy_before - info_after

    @staticmethod
    def majority_class(y):
        """
        Zwraca najczęściej występującą klasę w y.
        """
        if not y:
            return None
        best_class = None
        best_count = -1
        for class_ in set(y):
            c = y.count(class_)  # zliczamy ile razy class_ występuje w y
            if c > best_count:
                best_count = c
                best_class = class_
        return best_class

    def ID3(self, X, y, attributes, depth):
        """
        Rekurencyjnie buduje drzewo algorytmem ID3.
        attributes – lista indeksów atrybutów, które nie zostały jeszcze wykorzystane.
        depth – aktualna głębokość.
        """

        # Jeśli zbiór pusty, to zwracamy liść z None
        if not X or not y:
            return Node(leaf=True, label=None)

        # Jeśli wszystkie etykiety są takie same, to zwracamy liść z tą etykietą
        if len(set(y)) == 1:  # tylko jedna unikalna wartość
            return Node(leaf=True, label=y[0])

        # Jeśli osiągnęliśmy max_depth, to zwracamy liść z najczęstszą klasą
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(leaf=True, label=self.majority_class(y))

        # Jeśli nie ma już dostępnych atrybutów, to zwracamy liść z najczęstszą klasą
        if not attributes:
            return Node(leaf=True, label=self.majority_class(y))

        # Inny przypadek: wybieramy najlepszy atrybut - dający najwięcej informacji na temat y
        best_attribute = None
        best_inf_gain = -1.0
        for attribute in attributes:
            gain = self.inf_gain(X, y, attribute)
            if gain > best_inf_gain:
                best_inf_gain = gain
                best_attribute = attribute

        # Jeśli żaden atrybut nie daję nam informacji - nie opłaca się żadnego rozważać - kończymy na liściu z najczęstszą wartością.
        if best_inf_gain <= 0 or best_attribute is None:
            return Node(leaf=True, label=self.majority_class(y))

        # tworzymy węzeł wewnętrzny
        node = Node(
            leaf=False,
            attribute=best_attribute,
            label=self.majority_class(y)  # przyda się przy nieznanej wartości atrybutu
        )

        # Dla każdej wartości wybranego atrybutu budujemy poddrzewo
        values = list(set(row[best_attribute] for row in X))
        new_attributes = [a for a in attributes if a != best_attribute]  # teraz już w kolejnych ruchach nie wykorzystamy wybranego atrybutu

        for val in values:
            X_sub = []
            y_sub = []
            for row, label in zip(X, y):
                if row[best_attribute] == val:
                    X_sub.append(row)
                    y_sub.append(label)

            # rekurencyjnie budujemy poddrzewo
            child = self.ID3(X_sub, y_sub, new_attributes, depth + 1)
            node.children[val] = child

        return node

    def fit(self, X, y):
        """
        Buduje drzewo decyzyjne ID3 z ograniczeniem głębokości.
        X - zbiór z nasyzmi obserwacjami
        y - lista etykiet (które przewidujemy)
        """
        # ile jest cech (atrybutów) w jednym przykładzie
        n_features = len(X[0])

        # lista indeksów atrybutów
        attributes = list(range(n_features))

        # budujemy drzewo od korzenia, na głębokości 0
        self.tree = self.ID3(X, y, attributes, depth=0)

        # zwracamy siebie, żeby można było pisać: clf.fit(X, y)
        return self

    def predict(self, X):
        """
        Zwraca listę przewidywanych klas dla listy przykładów X.
        """
        if self.tree is None:
            raise ValueError("Model nie został jeszcze wytrenowany.")

        predictions = []

        for row in X:
            node = self.tree  # startujemy od korzenia

            # schodzimy w dół drzewa, aż trafimy na liść
            while not node.leaf:
                attr = node.attribute  # po której cesze dzielimy
                val = row[attr]  # wartość tej cechy w naszym przykładzie

                # jeśli mamy gałąź dla tej wartości → idziemy niżej
                if val in node.children:
                    node = node.children[val]
                else:
                    # jeśli nie ma takiej gałęzi → przerywamy, używamy label z aktualnego węzła
                    break

            # na końcu node to albo liść, albo węzeł bez odpowiedniej gałęzi
            predictions.append(node.label)

        return predictions
