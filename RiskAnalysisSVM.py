# Analisi del Rischio con SVM Kernelizzate

Questo notebook implementa un'analisi del rischio utilizzando Support Vector Machines (SVM) con kernel **lineare** e **polinomiale**. L'obiettivo è confrontare le due metodologie nella classificazione del rischio.

# 1. Caricamento e Controllo dei Dati

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import os

## Percorsi dei file
dataset_path = "dataset/"
file_discrimination = dataset_path + "Discrimination.csv"
file_target = dataset_path + "y_origin_bin.csv"

## Controllo dell'esistenza dei file prima del caricamento
if not os.path.exists(file_discrimination) or not os.path.exists(file_target):
    raise FileNotFoundError("Uno o entrambi i file richiesti non sono stati trovati.")

## Caricamento dei dataset
X = pd.read_csv(file_discrimination)
y = pd.read_csv(file_target)

# 2. Preprocessing dei Dati

## Selezione delle feature più rilevanti
selected_features = ['Age', 'RaceDesc', 'Sex', 'Pay Rate']
X_subset = X[selected_features]

## Verifica della presenza di valori mancanti
if X_subset.isnull().sum().sum() > 0 or y.isnull().sum().sum() > 0:
    raise ValueError("Il dataset contiene valori mancanti. ")

## Identificazione delle colonne categoriche e numeriche
categorical_cols = X_subset.select_dtypes(include=['object']).columns
numerical_cols = X_subset.select_dtypes(include=['int64', 'float64']).columns

## Preprocessing con standardizzazione e one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
X_preprocessed = preprocessor.fit_transform(X_subset)
X_train, _, y_train, _ = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# 3. Assegnazione dei Livelli di Rischio

def assign_risk_levels(model, X, y_train, max_levels=5):
    risk_levels = np.zeros(X.shape[0], dtype=int)
    current_risk = 1
    X_remaining = X.copy()
    y_remaining = y_train[:X.shape[0]].values.ravel()

    while X_remaining.shape[0] > 0 and current_risk < max_levels:
        model.fit(X_remaining, y_remaining)
        support_vectors_indices = model.support_
        original_indices = np.where(risk_levels == 0)[0]

        if len(support_vectors_indices) == 0:
            break

        risk_levels[original_indices[support_vectors_indices]] = current_risk
        X_remaining = np.delete(X_remaining, support_vectors_indices, axis=0)
        y_remaining = np.delete(y_remaining, support_vectors_indices)
        current_risk += 1

    risk_levels[risk_levels == 0] = max_levels
    return risk_levels

# 4. Visualizzazione dei Risultati

def plot_risk_map(X, risk_levels, title, file_name):
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X)
    color_map = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'lightgreen', 5: 'green'}
    risk_colors = [color_map[level] for level in risk_levels]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2D[:, 0], X_2D[:, 1], c=risk_colors, edgecolor='k', alpha=0.7)
    legend_elements = [Patch(facecolor=color, label=f"Livello {i}") for i, color in color_map.items()]
    plt.legend(handles=legend_elements, title="Livelli di Rischio", loc='upper right')
    plt.title(title)
    plt.xlabel("Componente Principale 1")
    plt.ylabel("Componente Principale 2")
    plt.savefig(file_name)
    plt.close()

def plot_risk_histogram(risk_levels, title, file_name):
    color_map = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'lightgreen', 5: 'green'}
    counts = np.bincount(risk_levels)[1:]
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, 6), counts, color=[color_map[i] for i in range(1, 6)])
    plt.title(title)
    plt.xlabel("Livelli di Rischio")
    plt.ylabel("Numero di Punti")
    plt.xticks(range(1, 6))
    plt.savefig(file_name)
    plt.close()

# 5. Addestramento e Confronto tra Kernel Lineare e Polinomiale

## Addestramento modelli
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
svm_poly = SVC(kernel='poly', degree=3, probability=True, random_state=42)

## Assegnazione dei livelli di rischio
risk_levels_linear = assign_risk_levels(svm_linear, X_train, y_train)
risk_levels_poly = assign_risk_levels(svm_poly, X_train, y_train)

## Visualizzazione risultati
plot_risk_map(X_train, risk_levels_linear, "Mappa dei Rischi SVM Lineare", "linear_risk_map.png")
plot_risk_histogram(risk_levels_linear, "Istogramma dei Livelli di Rischio (Lineare)", "linear_istogramma.png")

plot_risk_map(X_train, risk_levels_poly, "Mappa dei Rischi SVM Polinomiale", "poly_risk_map.png")
plot_risk_histogram(risk_levels_poly, "Istogramma dei Livelli di Rischio (Polinomiale)", "poly_istogramma.png")

## Confronto tra i modelli
changed_indices = np.where(risk_levels_linear != risk_levels_poly)[0]
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
plt.scatter(X_2D[:, 0], X_2D[:, 1], c='lightgray', edgecolor='k', alpha=0.5, label="Punti invariati")
plt.scatter(X_2D[changed_indices, 0], X_2D[changed_indices, 1], c='red', edgecolor='k', label="Punti con rischio cambiato")
plt.title("Confronto dei Livelli di Rischio (Lineare vs Polinomiale)")
plt.xlabel("Componente Principale 1")
plt.ylabel("Componente Principale 2")
plt.legend()
plt.savefig("risk_comparison.png")
plt.close()

print("Grafici salvati nella cartella dello script.")
