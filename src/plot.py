import numpy as np
from scipy.cluster.hierarchy import dendrogram
import os
import matplotlib.pyplot as plt
from typing import Callable
import pandas as pd
# Determina il percorso della cartella "assets/plot"


# def save_dendrogram(linkage_matrix, file_name="dendrogram.png"):
#     # Percorso completo per il file
#     file_path = os.path.join(plot_dir, file_name)
#     # Plot del dendrogramma
#     plt.figure(figsize=(10, 7))
#     dendrogram(linkage_matrix)
#     plt.title("Dendrogram")
#     plt.xlabel("Samples")
#     plt.ylabel("Distance")
#     plt.savefig(file_path)  # Salva l'immagine
#     plt.close()  # Chiude la figura per liberare memoria
#     print(f"Dendrogramma salvato in {file_path}")
#
#
# def save_silhouette_plot(X, labels, file_name="silhouette_plot.png"):
#     from sklearn.metrics import silhouette_samples
#     import numpy as np
#
#     file_path = os.path.join(plot_dir, file_name)
#
#     # Calcolo dei campioni della silhouette
#     silhouette_vals = silhouette_samples(X, labels)
#     y_ticks = []
#     y_lower, y_upper = 0, 0
#
#     # Plot delle silhouette
#     plt.figure(figsize=(10, 7))
#     for i, cluster in enumerate(np.unique(labels)):
#         cluster_silhouette_vals = silhouette_vals[labels == cluster]
#         cluster_silhouette_vals.sort()
#         y_upper += len(cluster_silhouette_vals)
#         plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
#         y_ticks.append((y_lower + y_upper) / 2)
#         y_lower += len(cluster_silhouette_vals)
#
#     plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")  # Linea verticale per il valore medio
#     plt.yticks(y_ticks, np.unique(labels))
#     plt.xlabel("Silhouette Coefficient")
#     plt.ylabel("Cluster")
#     plt.title("Silhouette Plot")
#     plt.savefig(file_path)  # Salva l'immagine
#     plt.close()
#     print(f"Plot della silhouette salvato in {file_path}")



# def save_plot(plot, file_name: str, plot_dir: str):
#     """
#     Salva il plot corrente nella directory specificata.
#
#     Args:
#         plot: Oggetto matplotlib.pyplot.
#         file_name (str): Nome del file di output.
#         plot_dir (str): Directory di output per i plot.
#     """
#     os.makedirs(plot_dir, exist_ok=True)
#     file_path = os.path.join(plot_dir, file_name)
#     plot.savefig(file_path)
#     plot.close()
#     print(f"Plot salvato in {file_path}")
#
# def save_dendrogram(linkage_matrix: np.ndarray, plot_dir: str):
#     """
#     Crea e salva il dendrogramma.
#
#     Args:
#         linkage_matrix (List[List[float]]): Matrice di linkage per il dendrogramma.
#         k (int): Numero di cluster.
#         plot_dir (str): Directory di output per i plot.
#     """
#     plt.figure(figsize=(10, 7))
#     dendrogram(linkage_matrix)
#     plt.title(f"Dendrogram")
#     plt.xlabel("Samples")
#     plt.ylabel("Distance")
#     save_plot(plt, f"dendrogram.png", plot_dir)
#
# def save_silhouette_plot(X: np.ndarray, labels: np.ndarray, k: int, plot_dir: str):
#     """
#     Crea e salva il plot della silhouette.
#
#     Args:
#         X (np.ndarray): Dati di input.
#         labels (np.ndarray): Etichette dei cluster.
#         k (int): Numero di cluster.
#         plot_dir (str): Directory di output per i plot.
#     """
#     from sklearn.metrics import silhouette_samples
#     silhouette_vals = silhouette_samples(X, labels)
#     y_ticks = []
#     y_lower, y_upper = 0, 0
#
#     plt.figure(figsize=(10, 7))
#     for i, cluster in enumerate(np.unique(labels)):
#         cluster_silhouette_vals = silhouette_vals[labels == cluster]
#         cluster_silhouette_vals.sort()
#         y_upper += len(cluster_silhouette_vals)
#         plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
#         y_ticks.append((y_lower + y_upper) / 2)
#         y_lower += len(cluster_silhouette_vals)
#
#     plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
#     plt.yticks(y_ticks, np.unique(labels))
#     plt.xlabel("Silhouette Coefficient")
#     plt.ylabel("Cluster")
#     plt.title(f"Silhouette Plot (k={k})")
#     save_plot(plt, f"silhouette_plot_k={k}.png", plot_dir)
def save_plot(plot, file_name: str, plot_dir: str):
    """
    Salva il plot corrente nella directory specificata.

    Args:
        plot: Oggetto matplotlib.pyplot.
        file_name (str): Nome del file di output.
        plot_dir (str): Directory di output per i plot.
    """
    os.makedirs(plot_dir, exist_ok=True)
    file_path = os.path.join(plot_dir, file_name)
    plot.savefig(file_path)
    plot.close()
    print(f"Plot salvato in {file_path}")


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    plot_dir: str,
    n_clusters: int,
    max_display_branches: int = 10):
    """
    Crea e salva il dendrogramma con un numero specifico di cluster colorati.

    Args:
        linkage_matrix (np.ndarray): Matrice di linkage per il dendrogramma.
        plot_dir (str): Directory di output per i plot.
        n_clusters (int): Numero desiderato di cluster da visualizzare.
        max_display_branches (int): Numero massimo di rami terminali mostrati.
    """
    from scipy.cluster.hierarchy import dendrogram
    plt.figure(figsize=(10, 7))

    n_clusters = max(2, min(int(n_clusters), len(linkage_matrix) + 1))
    # Calcola la soglia di taglio per ottenere il numero desiderato di cluster
    threshold = linkage_matrix[-(n_clusters - 1), 2]

    max_display_branches = max(2, int(max_display_branches))
    dendrogram(
        linkage_matrix,
        color_threshold=threshold,
        truncate_mode='lastp',
        p=max_display_branches,
        show_leaf_counts=True,
    )

    plt.title(f"Dendrogram with {n_clusters} clusters (last {max_display_branches} branches)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")

    # Aggiungi una linea orizzontale per indicare il taglio
    plt.axhline(y=float(threshold), color='r', linestyle='--')

    save_plot(plt, f"dendrogram_{n_clusters}_clusters_lastp{max_display_branches}.png", plot_dir)

    return threshold

def save_silhouette_plot(X: np.ndarray, labels: np.ndarray, k: int, plot_dir: str):
    """
    Crea e salva il plot della silhouette.

    Args:
        X (np.ndarray): Dati di input.
        labels (np.ndarray): Etichette dei cluster.
        k (int): Numero di cluster.
        plot_dir (str): Directory di output per i plot.
    """
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, labels)
    y_ticks = []
    y_lower, y_upper = 0, 0

    plt.figure(figsize=(10, 7))
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
    plt.yticks(y_ticks, np.unique(labels))
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.title(f"Silhouette Plot (k={k})")
    save_plot(plt, f"silhouette_plot_k{k}.png", plot_dir)


def save_elbow_plot(X: np.ndarray, max_clusters: int, clustering_func: Callable, plot_dir: str):
    """
    Crea e salva il grafico del gomito.

    Args:
        X (np.ndarray): Dati di input.
        max_clusters (int): Numero massimo di cluster da provare.
        clustering_func (Callable): Funzione che esegue il clustering e restituisce le etichette.
        plot_dir (str): Directory di output per i plot.
    """
    from sklearn.metrics import silhouette_score

    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        labels = clustering_func(n_clusters)
        silhouette_scores.append(silhouette_score(X, labels))

    plt.figure(figsize=(10, 7))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel("Numero di cluster")
    plt.ylabel("Silhouette Score")
    plt.title("Metodo del gomito usando Silhouette Score")
    save_plot(plt, "elbow_plot.png", plot_dir)


def _load_comparison_row(comparison_csv_path: str) -> pd.Series:
    df = pd.read_csv(comparison_csv_path)
    if df.empty:
        raise ValueError(f"Il file {comparison_csv_path} e vuoto")
    return df.iloc[0]


def plot_comparison_metrics_bar(comparison_csv_path: str, output_path: str | None = None) -> str:
    """
    Plotta un confronto a barre tra metriche custom e sklearn.

    Args:
        comparison_csv_path (str): Path al file comparison_with_sklearn.csv.
        output_path (str | None): Path completo del file immagine di output.

    Returns:
        str: Path del file salvato.
    """
    row = _load_comparison_row(comparison_csv_path)
    metrics = ['precision', 'recall', 'f1_score', 'rand_index']

    custom_vals = [float(row[f'custom_{m}']) for m in metrics]
    sklearn_vals = [float(row[f'sklearn_{m}']) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, custom_vals, width, label='Custom', color='#5B8FF9')
    plt.bar(x + width / 2, sklearn_vals, width, label='Scikit-learn', color='#5AD8A6')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Custom vs Scikit-learn: metriche principali')
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(comparison_csv_path), 'comparison_metrics_bar.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot salvato in {output_path}")
    return output_path


def plot_comparison_deltas(comparison_csv_path: str, output_path: str | None = None) -> str:
    """
    Plotta la differenza (custom - sklearn) sulle metriche principali.

    Args:
        comparison_csv_path (str): Path al file comparison_with_sklearn.csv.
        output_path (str | None): Path completo del file immagine di output.

    Returns:
        str: Path del file salvato.
    """
    row = _load_comparison_row(comparison_csv_path)
    metrics = ['precision', 'recall', 'f1_score', 'rand_index']
    deltas = [float(row[f'custom_{m}']) - float(row[f'sklearn_{m}']) for m in metrics]

    plt.figure(figsize=(10, 6))
    colors = ['#F6BD16' if d >= 0 else '#F08BB4' for d in deltas]
    plt.bar(metrics, deltas, color=colors)
    plt.axhline(0.0, color='black', linewidth=1)
    plt.ylabel('Delta (custom - sklearn)')
    plt.title('Delta metriche: custom vs scikit-learn')
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(comparison_csv_path), 'comparison_metrics_delta.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot salvato in {output_path}")
    return output_path


def plot_confusion_pair_deltas(comparison_csv_path: str, output_path: str | None = None) -> str:
    """
    Plotta le differenze (custom - sklearn) su TP, FP, TN, FN.

    Args:
        comparison_csv_path (str): Path al file comparison_with_sklearn.csv.
        output_path (str | None): Path completo del file immagine di output.

    Returns:
        str: Path del file salvato.
    """
    row = _load_comparison_row(comparison_csv_path)
    comps = ['tp', 'fp', 'tn', 'fn']
    deltas = [int(row[f'custom_{c}']) - int(row[f'sklearn_{c}']) for c in comps]

    plt.figure(figsize=(10, 6))
    colors = ['#5B8FF9' if d >= 0 else '#F08BB4' for d in deltas]
    plt.bar([c.upper() for c in comps], deltas, color=colors)
    plt.axhline(0, color='black', linewidth=1)
    plt.ylabel('Delta conteggi (custom - sklearn)')
    plt.title('Differenze TP/FP/TN/FN: custom vs scikit-learn')
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(comparison_csv_path), 'comparison_pairs_delta.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot salvato in {output_path}")
    return output_path


def plot_confusion_matrices_custom_vs_sklearn(comparison_csv_path: str, output_path: str | None = None) -> str:
    """
    Plotta due matrici 2x2 affiancate (pair-count confusion matrix):
    - Clustering Ibrido (custom)
    - Scikit-learn (baseline)

    La matrice e nel formato:
    [[TP, FP],
     [FN, TN]]
    """
    row = _load_comparison_row(comparison_csv_path)

    custom_matrix = np.array([
        [int(row['custom_tp']), int(row['custom_fp'])],
        [int(row['custom_fn']), int(row['custom_tn'])],
    ])
    sklearn_matrix = np.array([
        [int(row['sklearn_tp']), int(row['sklearn_fp'])],
        [int(row['sklearn_fn']), int(row['sklearn_tn'])],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    matrices = [custom_matrix, sklearn_matrix]
    titles = [
        'Clustering Ibrido (Custom)',
        f"Scikit-learn ({row.get('sklearn_model', 'baseline')})",
    ]

    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks([0, 1], labels=['Pred Pos', 'Pred Neg'])
        ax.set_yticks([0, 1], labels=['True Pos', 'True Neg'])
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Confronto matrici di confusione (pair-count)')
    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(comparison_csv_path), 'comparison_confusion_matrices.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot salvato in {output_path}")
    return output_path