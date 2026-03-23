import os
import time
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from .data import DataHandler
from .evaluation import (
    evaluate_clustering,
    print_contingency_matrix,
    save_evaluation_results,
)
from .hierarchical_clustering import HierarchicalClustering
from .plot import (
    save_silhouette_plot,
    plot_dendrogram,
    plot_cluster_projection_pca,
    plot_contingency_heatmap,
)


def _fit_agglomerative_labels(
        X_data: np.ndarray,
        linkage_method: str,
        distance: str,
        optimal_k: int,
) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    kwargs = {
        'n_clusters': optimal_k,
        'linkage': linkage_method,
    }
    if linkage_method == 'ward':
        kwargs['metric'] = 'euclidean'
    else:
        kwargs['metric'] = distance

    try:
        model = AgglomerativeClustering(**kwargs)
    except TypeError:
        # Compatibilita con versioni sklearn che usano ancora "affinity".
        affinity = kwargs.pop('metric')
        kwargs['affinity'] = affinity
        model = AgglomerativeClustering(**kwargs)

    return model.fit_predict(X_data)


def _sklearn_baseline_labels(
        X: np.ndarray,
        linkage_method: str,
        distance: str,
        optimal_k: int,
        pre_clustering: bool,
        k_means_reduction: int,
) -> Tuple[np.ndarray, str]:
    from sklearn.cluster import KMeans

    if pre_clustering:
        reduced_X, kmeans_labels = kmeans_pre_clustering(X, max_clusters=k_means_reduction)

        if linkage_method in {'single', 'complete', 'average', 'ward'} and (linkage_method != 'ward' or distance == 'euclidean'):
            reduced_labels = _fit_agglomerative_labels(
                X_data=reduced_X,
                linkage_method=linkage_method,
                distance=distance,
                optimal_k=optimal_k,
            )
            mapped_labels = reduced_labels[kmeans_labels]
            return mapped_labels, 'AgglomerativeClustering_on_kmeans_centroids'

        # Fallback strutturale per linkage non disponibile in sklearn (es. centroid).
        reduced_labels = KMeans(n_clusters=optimal_k, random_state=42, init='k-means++').fit_predict(reduced_X)
        mapped_labels = reduced_labels[kmeans_labels]
        return mapped_labels, 'KMeans_fallback_on_kmeans_centroids'

    if linkage_method in {'single', 'complete', 'average', 'ward'} and (linkage_method != 'ward' or distance == 'euclidean'):
        labels = _fit_agglomerative_labels(
            X_data=X,
            linkage_method=linkage_method,
            distance=distance,
            optimal_k=optimal_k,
        )
        return labels, 'AgglomerativeClustering'

    labels = KMeans(n_clusters=optimal_k, random_state=42, init='k-means++').fit_predict(X)
    return labels, 'KMeans_fallback'


def setup_directories(dataset_name: str = 'Frogs_MFCCs') -> Tuple[str, str, str]:
    """
    Configura e crea le directory necessarie per il progetto.

    Returns:
        Tuple[str, str, str]: Percorsi per dataset, risultati e plot.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, 'assets', 'Dataset')

    output_dir = os.path.join(project_root, 'assets', dataset_name)
    output_dir = os.path.join(output_dir, 'Results')

    plot_dir = os.path.join(project_root, 'assets', dataset_name)
    plot_dir = os.path.join(plot_dir, 'Plot')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return dataset_dir, output_dir, plot_dir


def load_and_preprocess_data(dataset_path: str, categorical:List[str]=None, soglia:float=1.01) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carica e pre-processa il dataset.

    Args:
        dataset_path (str): Percorso del file del dataset.
        categorical
        soglia

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Features e labels del dataset.
    """

    data_handler = DataHandler(dataset_path)
    data_handler.preprocess_data(soglia, categorical=categorical)
    X = data_handler.get_features()
    y = data_handler.get_labels().iloc[:, -1].values
    return X, y


def kmeans_pre_clustering(X: pd.DataFrame, max_clusters: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Esegue un pre-clustering utilizzando K-Means.

    Args:
        X (pd.DataFrame): Dati di input.
        max_clusters (int): Numero di cluster.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Centroidi e etichette dei cluster.
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=max_clusters, random_state=42, init='k-means++')
    labels = kmeans.fit_predict(X)
    return kmeans.cluster_centers_, labels


# def kmeans_pre_clustering(X: pd.DataFrame, max_clusters: int = 30) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Esegue un pre-clustering utilizzando K-Means, determinando automaticamente il numero ottimale di cluster.
#
#     Args:
#         X (pd.DataFrame): Dati di input.
#         max_clusters (int): Numero massimo di cluster da provare.
#
#     Returns:
#         Tuple[np.ndarray, np.ndarray]: Centroidi e etichette dei cluster ottimali.
#     """
#     silhouette_scores = []
#     kmeans_models = []
#
#     for n_clusters in range(2, max_clusters + 1):
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = kmeans.fit_predict(X)
#         score = silhouette_score(X, labels)
#         silhouette_scores.append(score)
#         kmeans_models.append(kmeans)
#
#     optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
#     optimal_kmeans = kmeans_models[optimal_n_clusters - 2]
#
#     return optimal_kmeans.cluster_centers_, optimal_kmeans.labels_


def create_linkage_matrix(hc: HierarchicalClustering) -> np.ndarray:
    """
    Crea la matrice di linkage per il dendrogramma.

    Args:
        hc (HierarchicalClustering): Oggetto di clustering gerarchico.

    Returns:
        np.ndarray: Matrice di linkage.
    """
    linkage_matrix = []
    name_to_idx = {}
    current_idx = 0

    for a, b, dist in hc.get_cluster_history():
        if a not in name_to_idx:
            name_to_idx[a.name] = float(a.name)
            current_idx += 1
        if b not in name_to_idx:
            name_to_idx[b.name] = float(b.name)
            current_idx += 1

        linkage_matrix.append([
            name_to_idx[a.name],
            name_to_idx[b.name],
            dist,
            len(a.indices) + len(b.indices)
        ])

    return np.array(linkage_matrix)


def run_clustering(
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        linkage_method: str,
        distance: str,
        output_dir: str,
        plot_dir: str,
        max_clusters: int = 8,
        k_means_reduction: int = 10,
        optimal_k: int = 4,
        pre_clustering: bool = True,
        compare_with_sklearn: bool = True,
        dendrogram_display_branches: int = 10,
        plot_cluster_views: bool = False) -> None:
    """
    Esegue il clustering gerarchico e salva i risultati.

    Args:
        X (pd.DataFrame): Dati di input.
        y (np.ndarray): Etichette vere.
        linkage_method (str): Metodo di linkage.
        distance (str): Metrica di distanza.
        output_dir (str): Directory per i risultati.
        plot_dir (str): Directory per i plot.
        max_clusters (int)
        k_means_reduction (int)
        optimal_k (int)
    """
    # Configurazione delle sottocartelle per i risultati
    sub_dir = f'k_means_reduction={k_means_reduction}'
    sub_output_dir = os.path.join(output_dir, sub_dir)
    sub_plot_dir = os.path.join(plot_dir, sub_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    os.makedirs(sub_plot_dir, exist_ok=True)
    sub_dir = os.path.join(sub_dir, f"{linkage_method}_{distance}")
    sub_output_dir = os.path.join(output_dir, sub_dir)
    sub_plot_dir = os.path.join(plot_dir, sub_dir)
    os.makedirs(sub_output_dir, exist_ok=True)
    os.makedirs(sub_plot_dir, exist_ok=True)

    # Inizializzazione e fit del modello
    if not pre_clustering:
        hc = HierarchicalClustering(linkage=linkage_method, X=X, distance_metric=distance,
                                    pre_clustering_func=None,
                                    max_clusters=k_means_reduction)
    else:
        hc = HierarchicalClustering(linkage=linkage_method, X=X, distance_metric=distance,
                                    pre_clustering_func=kmeans_pre_clustering,
                                    max_clusters=k_means_reduction)

    print(f'Inizio fit per {linkage_method} linkage e distanza {distance}')
    hc.fit()
    # hc.save_cluster_history_to_json(filename="history_for_dendrogram.json")
    print('Fine fit')
    # Creazione del dendrogramma
    linkage_matrix = create_linkage_matrix(hc)

    if int(optimal_k) < 2:
        raise ValueError("optimal_k deve essere >= 2. Per Frogs usare 4.")

    dendrogram_k = max(2, min(int(optimal_k), len(linkage_matrix) + 1))

    cut_distance: float = plot_dendrogram(
        linkage_matrix,
        sub_plot_dir,
        dendrogram_k,
        max_display_branches=dendrogram_display_branches,
    )
    print(f'Dendrogramma tagliato a {dendrogram_k} cluster (soglia={cut_distance:.4f})')

    if dendrogram_k != int(optimal_k):
        print(f"optimal_k={optimal_k} ridotto a {dendrogram_k} per taglio valido del dendrogramma")
    optimal_k = dendrogram_k
    print(f"Numero di cluster finale impostato: {optimal_k}")
    # Creazione del grafico del gomito
    # save_elbow_plot(X, max_clusters, hc.predict, sub_plot_dir)
    # Previsione e valutazione
    labels = hc.predict(optimal_k)

    # Calcola e salva le metriche di valutazione
    # Valuta il clustering
    # evaluation_results = evaluate_clustering(y_true=y, y_pred=labels, X=X, model_name="Hierarchical Clustering")
    evaluation_results = evaluate_clustering(y_true=y, y_pred=labels)
    print_contingency_matrix(y_true=y, y_pred=labels)
    evaluation_results['clusters'] = optimal_k
    evaluation_results['k_means_reduction'] = k_means_reduction
    save_evaluation_results(evaluation_results, "evaluation_results.csv", sub_output_dir)

    if plot_cluster_views:
        plot_cluster_projection_pca(
            X=X,
            labels=labels,
            plot_dir=sub_plot_dir,
            title=f'PCA cluster projection - {linkage_method} ({distance})',
        )
        plot_contingency_heatmap(
            y_true=y,
            y_pred=labels,
            plot_dir=sub_plot_dir,
            title=f'Contingency heatmap - {linkage_method} ({distance})',
        )

    if compare_with_sklearn:
        sklearn_start = time.time()
        sklearn_labels, sklearn_model_name = _sklearn_baseline_labels(
            X=X,
            linkage_method=linkage_method,
            distance=distance,
            optimal_k=optimal_k,
            pre_clustering=pre_clustering,
            k_means_reduction=k_means_reduction,
        )
        sklearn_elapsed = time.time() - sklearn_start

        sklearn_eval = evaluate_clustering(y_true=y, y_pred=sklearn_labels)
        comparison_results = {
            'dataset_size': len(y),
            'true_classes_count': int(len(np.unique(y))),
            'true_classes': '|'.join([str(c) for c in sorted(np.unique(y))]),
            'linkage_method': linkage_method,
            'distance': distance,
            'clusters': optimal_k,
            'k_means_reduction': k_means_reduction,
            'pre_clustering': pre_clustering,
            'sklearn_model': sklearn_model_name,
            'sklearn_elapsed_seconds': sklearn_elapsed,
        }
        comparison_results.update({f'custom_{k}': v for k, v in evaluation_results.items()})
        comparison_results.update({f'sklearn_{k}': v for k, v in sklearn_eval.items()})
        save_evaluation_results(comparison_results, "comparison_with_sklearn.csv", sub_output_dir)
    # Salva solo la silhouette finale della configurazione scelta.
    if len(np.unique(labels)) > 1:
        save_silhouette_plot(X, labels, optimal_k, sub_plot_dir)

    print(f"Risultati per {linkage_method} linkage e distanza {distance} salvati.")
