import csv
import os
from typing import Dict

import numpy as np


def save_evaluation_results(results: dict, file_name: str, output_dir: str):
    """
    Salva i risultati della valutazione in un file CSV.

    Args:
        results (dict): Dizionario contenente i risultati della valutazione.
        file_name (str): Nome del file di output.
        output_dir (str): Directory di output.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    print(f"Risultati della valutazione salvati in {file_path}")


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcola le metriche di clustering usando l'approccio basato su coppie di istanze.
    """

    def count_pairs_in_group(n: int) -> int:
        return (n * (n - 1)) // 2

    tp = 0
    fp = 0
    fn = 0

    for cluster_id in np.unique(y_pred):
        cluster_mask = y_pred == cluster_id

        class_counts = {}
        for class_id in np.unique(y_true):
            count = np.sum((y_true == class_id) & cluster_mask)
            if count > 0:
                class_counts[class_id] = count

        for count in class_counts.values():
            if count > 1:
                tp += count_pairs_in_group(count)

        classes = list(class_counts.keys())
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                fp += class_counts[classes[i]] * class_counts[classes[j]]

    for class_id in np.unique(y_true):
        class_distribution = {}
        class_mask = y_true == class_id

        for cluster_id in np.unique(y_pred):
            count = np.sum((y_pred == cluster_id) & class_mask)
            if count > 0:
                class_distribution[cluster_id] = count

        clusters = list(class_distribution.keys())
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                fn += class_distribution[clusters[i]] * class_distribution[clusters[j]]

    total_pairs = count_pairs_in_group(len(y_true))
    tn = total_pairs - (tp + fp + fn)

    rand_index = (tp + tn) / total_pairs
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'rand_index': rand_index,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }


def print_cluster_statistics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Stampa statistiche dettagliate sui cluster per debug.
    """
    print("\nStatistiche dei cluster:")
    for cluster in np.unique(y_pred):
        cluster_mask = y_pred == cluster
        print(f"\nCluster {cluster}:")
        for label in np.unique(y_true):
            count = np.sum((y_true == label) & cluster_mask)
            if count > 0:
                print(f"  Classe {label}: {count} elementi")


def print_contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    canonical_pred = [f"C{i + 1}" for i in range(len(unique_pred))]
    pred_mapping = {canonical_pred[i]: unique_pred[i] for i in range(len(unique_pred))}

    matrix = np.zeros((len(unique_true), len(unique_pred)), dtype=int)

    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            mask = (y_true == true_label) & (y_pred == pred_label)
            matrix[i, j] = np.sum(mask)

    print("\nMatrice di Contingenza:")
    print(" " * 10, end="")
    for pred in canonical_pred:
        print(f"{pred:>6}", end=" ")
    print("\n" + "-" * (10 + 5 * len(unique_pred)))

    for i, true_label in enumerate(unique_true):
        print(f"{true_label:8} |", end=" ")
        for j in range(len(unique_pred)):
            print(f"{matrix[i, j]:4}", end=" ")
        print()

    print("Mappatura cluster predetti:", ", ".join([f"{k}=raw {v}" for k, v in pred_mapping.items()]))

    return matrix
