from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zad3.DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("cardio_train.csv", sep=";")

# rzućmy okiem na dane
print(df.head())
print(df.columns)
# zmienna id nic nie wnosi - to tylko oznaczenie kolejnej obserswacji - usuńmy ją
df = df.drop(columns=["id"])

# dodatkowo możemy zauważyć, że w mamy dużo zmiennych ciągłych (age, height, weight, ap_hi, ap_lo)
# podzielimy te wartości na przedziały (kategorie), a następnie zastosujemy one-hot encoding


# 1. wiek
df["age_years"] = df["age"] / 365.25
df["age_sec"] = pd.cut(
    df["age_years"],
    bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    labels=False,
    include_lowest=True
)
df = df.drop(columns=["age", "age_years"])

# 2. wzrost
df["height_sec"] = pd.cut(
    df["height"],
    bins=[0, 140, 150, 160, 170, 180, 190, 250],
    labels=False,
    include_lowest=True
)
df = df.drop(columns=["height"])

# 3. waga
df["weight_sec"] = pd.cut(
    df["weight"],
    bins=[0, 40, 50, 60, 70, 80, 90, 100, 300],
    labels=False,
    include_lowest=True
)
df = df.drop(columns=["weight"])

# 4. ap
df["ap_hi_sec"] = pd.cut(
    df["ap_hi"],
    bins=[0, 120, 140, 160, 300],
    labels=False,
    include_lowest=True
)
df["ap_lo_sec"] = pd.cut(
    df["ap_lo"],
    bins=[0, 80, 90, 100, 200],
    labels=False,
    include_lowest=True
)
df = df.drop(columns=["ap_hi", "ap_lo"])
print(df.columns)


# teraz robimy one-hot encoding na wszystkich kolumnach z więcej niż 2 kategoriami
target = "cardio"

X = df.drop(columns=[target])
y = df[target]

cat_cols = ["age_sec", "height_sec", "weight_sec", "ap_hi_sec", "ap_lo_sec", "gluc", "cholesterol"]

X_cat = X[cat_cols].astype("category")

# one-hot na wybranych kolumnach
X_oh = pd.get_dummies(X_cat, drop_first=False, dtype=int)
other_cols = [c for c in X.columns if c not in cat_cols]

X_final = pd.concat([X[other_cols].reset_index(drop=True), X_oh.reset_index(drop=True)], axis=1)

print(X_final.head())
print(X_final.columns)

# podział na zmienne objaśniające i objaśnianą
X = X_final.values      # numpy array
y = df[target].values   # numpy array


# ESPERYMENTY

def accuracy(y_true, y_pred):
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)

max_depth_values = range(1, 21)
n_seeds = 10
seed_values = range(10)  # 0..9

train_ac_per_depth = {d: [] for d in max_depth_values}
val_ac_per_depth   = {d: [] for d in max_depth_values}
seed_best_depths = []   # najlepsza głębokość dla każdego seeda

for seed in seed_values:
    print(f"\nRANDOM SEED = {seed}")

    # podział: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    # po 15% na test i val
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,test_size=0.5,random_state=seed, stratify=y_temp)

    X_train_list = X_train.tolist()
    X_val_list   = X_val.tolist()
    y_train_list = y_train.tolist()
    y_val_list   = y_val.tolist()

    results_seed = []  # (depth, train_acc, val_acc) dla tego seeda

    for depth in max_depth_values:
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train_list, y_train_list)

        y_pred_train = model.predict(X_train_list)
        ac_train = accuracy(y_train_list, y_pred_train)

        y_pred_val = model.predict(X_val_list)
        ac_val = accuracy(y_val_list, y_pred_val)

        train_ac_per_depth[depth].append(ac_train)
        val_ac_per_depth[depth].append(ac_val)
        results_seed.append((depth, ac_train, ac_val))

        print(f"seed={seed:2d} | max depth={depth:2d} | "
              f"train accuracy={ac_train:.4f} | val accuracy={ac_val:.4f}")

    # wyznaczmy najlepsza głębokość dla tego konkretnego seeda
    depth, _, best_val = max(results_seed, key=lambda t: t[2])
    seed_best_depths.append(depth)
    print(f"--> seed={seed}: best_depth={depth}, val_acc={best_val:.4f}")

# średnie accuracy po wszystkich seedach
depths = list(max_depth_values)
mean_train_acs = [np.mean(train_ac_per_depth[d]) for d in depths]
mean_val_acs   = [np.mean(val_ac_per_depth[d]) for d in depths]

plt.figure()
plt.plot(depths, mean_train_acs, marker="o", label="średnie train accuracy")
plt.plot(depths, mean_val_acs, marker="o", label="średnie validation accuracy")
plt.xlabel("Maksymalna głębokość drzewa")
plt.ylabel("Accuracy")
plt.title(f"Train vs validation accuracy (uśrednione po {n_seeds} seedach)")
plt.grid(True)
plt.legend()
plt.show()

# najczęściej występująca najlepsza głębokość (moda)
counts = Counter(seed_best_depths)
mode_depth, mode_count = counts.most_common(1)[0]
print(f"\nNajczęściej najlepszą głębokością jest: {mode_depth} (wystąpiła {mode_count} razy na {n_seeds})")


test_acs = []

for seed in seed_values:
    print(f"\ntestowanie na seed = {seed} (depth={mode_depth})")

    # ten sam schemat podziału
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        random_state=seed,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp
    )

    X_train_full = X_train.tolist()
    X_test_list  = X_test.tolist()
    y_train_full = y_train.tolist()
    y_test_list  = y_test.tolist()

    final_model = DecisionTreeClassifier(max_depth = mode_depth)
    final_model.fit(X_train_full, y_train_full)

    y_pred_test = final_model.predict(X_test_list)
    acc_test = accuracy(y_test_list, y_pred_test)
    test_acs.append(acc_test)

    print(f"Test accuracy (seed={seed}): {acc_test:.4f}")

# podsumowanie po wszystkich seedach
mean_test_acc = np.mean(test_acs)
std_test_acc  = np.std(test_acs)

print(f"\nŚrednie test accuracy po {n_seeds} seedach (depth={mode_depth}): {mean_test_acc:.4f} ± {std_test_acc:.4f}")
