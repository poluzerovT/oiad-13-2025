#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

TRAIN_PATH = "../../datasets/insurance_train.csv"
TEST_PATH = "../../datasets/insurance_test.csv"






def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize(X):
    """Стандартизация признаков с защитой от скаляров"""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def apply_standardization(X, mean, std):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    std_adj = std.copy()
    std_adj[std_adj == 0] = 1.0
    return (X - mean) / std_adj


def load_and_prepare(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cat_cols = ["sex", "smoker", "region"]
    num_cols = ["age", "bmi", "children"]
    target_col = "charges"

    print("\n=== 1. Подготовка данных ===")
    print(f"\nИсходные данные (первые 5 строк):")
    print(train_df[cat_cols + num_cols + [target_col]].head())
    
    print(f"\nКатегориальные признаки: {cat_cols}")
    print(f"Числовые признаки: {num_cols}")
    print(f"Целевая переменная: {target_col}")

    # one-hot кодирование
    train_X = pd.get_dummies(train_df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
    test_X = pd.get_dummies(test_df[cat_cols + num_cols], columns=cat_cols, drop_first=True)

    print(f"\nПосле one-hot кодирования (признаки):")
    print(train_X.columns.tolist())
    print(f"Количество признаков: {len(train_X.columns)}")

    # выравнивание признаков
    train_X, test_X = train_X.align(test_X, join="outer", axis=1, fill_value=0)

    # гарантируем numpy float массивы
    X_train = train_X.to_numpy(dtype=float)
    X_test = test_X.to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    # корреляции
    corr_df = pd.get_dummies(train_df[cat_cols + num_cols + [target_col]], columns=cat_cols, drop_first=True)
    corr_matrix = corr_df.corr(numeric_only=True)

    print(f"\nМатрица корреляций (с целевой переменной '{target_col}'):")
    print(corr_matrix[target_col].sort_values(ascending=False))

    return X_train, y_train, X_test, y_test, corr_matrix, train_X.columns.tolist()


def normal_equation(X, y):
    XT_X = X.T @ X
    XT_y = X.T @ y
    w = np.linalg.pinv(XT_X) @ XT_y
    return w


def gradient_descent(X, y, lr=0.05, epochs=20000):
    n_samples, n_features = X.shape
    rng = np.random.default_rng()
    w = rng.normal(0, 0.01, size=n_features)

    history = []  # список для ошибок

    for _ in range(epochs):
        y_pred = X @ w
        grad = (2.0 / n_samples) * (X.T @ (y_pred - y))
        w -= lr * grad

        # сохраняем MSE на каждой итерации
        loss = np.mean((y - y_pred) ** 2)
        history.append(loss)

    return w, history


def run(train_path=TRAIN_PATH, test_path=TEST_PATH, lr=0.05, epochs=2000):
    X_train_raw, y_train, X_test_raw, y_test, corr_matrix, feature_names = load_and_prepare(train_path, test_path)

    # стандартизация
    X_train_std, mean, std = standardize(X_train_raw)
    X_test_std = apply_standardization(X_test_raw, mean, std)
    X_train = add_bias(X_train_std)
    X_test = add_bias(X_test_std)

    # baseline
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = mse(y_test, baseline_pred)

    print("\n=== 2. Многомерная линейная регрессия ===")
    
    # аналитическое решение
    print("\n2.1. Аналитическое решение (Normal Equation)")
    print("Формула: w = (X^T X)^-1 X^T y")
    w_analytical = normal_equation(X_train, y_train)
    analytical_mse = mse(y_test, X_test @ w_analytical)
    
    print(f"\nПолученные параметры модели (w):")
    print(f"  Bias (w0): {w_analytical[0]:.4f}")
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {w_analytical[i+1]:.4f}")
    print(f"\nMSE на train: {mse(y_train, X_train @ w_analytical):.4f}")
    print(f"MSE на test: {analytical_mse:.4f}")

    # градиентный спуск
    print("\n2.2. Градиентный спуск")
    print(f"Параметры: learning_rate={lr}, epochs={epochs}")
    w_gd, history = gradient_descent(X_train, y_train, lr=lr, epochs=epochs)
    gd_mse = mse(y_test, X_test @ w_gd)
    
    print(f"\nПолученные параметры модели (w):")
    print(f"  Bias (w0): {w_gd[0]:.4f}")
    for i, feature in enumerate(feature_names):
        print(f"  {feature}: {w_gd[i+1]:.4f}")
    
    print(f"\nСходимость:")
    print(f"  Начальная MSE (epoch 1): {history[0]:.4f}")
    print(f"  Конечная MSE (epoch {epochs}): {history[-1]:.4f}")
    print(f"  Улучшение: {history[0] - history[-1]:.4f}")
    print(f"\nMSE на train: {mse(y_train, X_train @ w_gd):.4f}")
    print(f"MSE на test: {gd_mse:.4f}")

    y_pred_gd = X_test @ w_gd



    print("\n=== 3. Оценка обобщающей способности ===")
    print("\nСравнение моделей на тестовых данных (MSE):")
    print(f"Baseline (mean):        {baseline_mse:.4f}")
    print(f"Analytical solution:    {analytical_mse:.4f}")
    print(f"Gradient descent:       {gd_mse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=TRAIN_PATH)
    parser.add_argument("--test", type=str, default=TEST_PATH)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=2000)
    args = parser.parse_args()

    run(train_path=args.train, test_path=args.test, lr=args.lr, epochs=args.epochs)
