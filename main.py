import argparse
import os
import numpy as np

from src.funzioni import load_and_preprocess_data, run_clustering, setup_directories


def resolve_kmeans_reduction(k_means_reduction: str, n_samples: int) -> int:
    value = str(k_means_reduction).strip().lower()
    if value == 'auto':
        # Regola suggerita: sqrt(N/2)
        return max(2, int(round(np.sqrt(n_samples / 2))))

    parsed = int(value)
    if parsed < 2:
        raise ValueError("k_means_reduction deve essere >= 2")
    return parsed


def single_run(
        linkage_method: str,
        distance_metric: str,
        max_clusters: int = 8,
    k_means_reduction: str = '15',
        optimal_k=-1,
        categorical=None,
        soglia: float = 1.01,
        dataset_name='Frogs_MFCCs',
        pre_clustering: bool = True,
        compare_with_sklearn: bool = True,
        dendrogram_display_branches: int = 10,
):
    # Setup iniziale
    dataset_dir, output_dir, plot_dir = setup_directories(dataset_name)
    dataset_path = os.path.join(dataset_dir, f'{dataset_name}.csv')
    if dataset_name == 'Frogs_MFCCs':
        categorical = ['Species', 'Genus', 'RecordID']

    # Caricamento e pre-processing dei dati
    X, y = load_and_preprocess_data(dataset_path, categorical=categorical, soglia=soglia)
    resolved_k_reduction = resolve_kmeans_reduction(k_means_reduction, len(X))
    print(f'Dataset {dataset_name} caricato e pre-processato')
    print(f'k_means_reduction risolto a: {resolved_k_reduction}')
    run_clustering(X,
                   y,
                   linkage_method,
                   distance_metric,
                   output_dir,
                   plot_dir,
                   max_clusters=max_clusters,
                   k_means_reduction=resolved_k_reduction,
                   optimal_k=optimal_k,
                   pre_clustering=pre_clustering,
                   compare_with_sklearn=compare_with_sklearn,
                   dendrogram_display_branches=dendrogram_display_branches)

    print("Progetto completato e tutti i risultati salvati.")


def multi_run(
        max_clusters: int = 8,
        k_min: int = 5,
        k_max: int = 45,
        optimal_k=-1,
        categorical=None,
        soglia: float = 1.01,
        dataset_name='iris_dataset',
        pre_clustering: bool = True,
        compare_with_sklearn: bool = True,
        dendrogram_display_branches: int = 10,
    ):
    # Setup iniziale
    dataset_dir, output_dir, plot_dir = setup_directories(dataset_name)
    if dataset_name == 'Frogs_MFCCs':
        categorical = ['Species', 'Genus', 'RecordID']
    dataset_path = os.path.join(dataset_dir, f'{dataset_name}.csv')

    # Caricamento e pre-processing dei dati
    X, y = load_and_preprocess_data(dataset_path, categorical=categorical, soglia=soglia)
    print(f'Dataset {dataset_name} caricato e pre-processato')

    # Definizione dei metodi di linkage e delle metriche di distanza da utilizzare
    linkage_methods = ['single', 'complete', 'average', 'centroid', 'ward']
    distance_metrics = ['euclidean']

    if k_min < 2:
        raise ValueError("k_min deve essere >= 2")
    if k_max <= k_min:
        raise ValueError("k_max deve essere maggiore di k_min")

    # Esecuzione del clustering per ogni combinazione di linkage e distanza
    for k in range(k_min, k_max):
        for linkage_method in linkage_methods:
            for distance in distance_metrics:
                run_clustering(X,
                               y,
                               linkage_method,
                               distance,
                               output_dir,
                               plot_dir,
                               max_clusters=max_clusters,
                               k_means_reduction=k,
                               optimal_k=optimal_k,
                               pre_clustering=pre_clustering,
                               compare_with_sklearn=compare_with_sklearn,
                               dendrogram_display_branches=dendrogram_display_branches,
                )

    print("Progetto completato e tutti i risultati salvati.")

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline di clustering ibrido: pre-clustering K-Means + clustering gerarchico agglomerativo."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Esegue una singola configurazione o una ricerca su piu configurazioni.",
    )
    parser.add_argument(
        "--dataset",
        choices=["Frogs_MFCCs", "winequality-red", "winequality-white", "iris_dataset", "hepatitis"],
        default="Frogs_MFCCs",
        help="Dataset da utilizzare.",
    )
    parser.add_argument(
        "--linkage",
        choices=["single", "complete", "average", "centroid", "ward"],
        default="ward",
        help="Metodo di linkage per la modalita single.",
    )
    parser.add_argument(
        "--distance",
        default="euclidean",
        help="Metrica di distanza da usare (es. euclidean).",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Numero massimo di cluster candidati quando optimal_k non e fissato.",
    )
    parser.add_argument(
        "--kmeans-reduction",
        type=str,
        default="15",
        help="Numero di centroidi usati nel pre-clustering K-Means oppure 'auto' (sqrt(N/2)).",
    )
    parser.add_argument(
        "--optimal-k",
        type=int,
        default=-1,
        help="Numero di cluster finale fisso. Usa -1 per selezione automatica.",
    )
    parser.add_argument(
        "--pre-clustering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abilita/disabilita il pre-clustering con K-Means.",
    )
    parser.add_argument(
        "--soglia",
        type=float,
        default=1.01,
        help="Soglia per eliminazione feature correlate (<=1 attiva la riduzione).",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=5,
        help="Valore iniziale (incluso) per la scansione k in modalita multi.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=45,
        help="Valore finale (escluso) per la scansione k in modalita multi.",
    )
    parser.add_argument(
        "--compare-sklearn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abilita/disabilita il confronto con baseline scikit-learn.",
    )
    parser.add_argument(
        "--dendrogram-branches",
        type=int,
        default=10,
        help="Numero massimo di rami mostrati nel dendrogramma (truncate_mode='lastp').",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "single":
        single_run(
            linkage_method=args.linkage,
            distance_metric=args.distance,
            max_clusters=args.max_clusters,
            k_means_reduction=args.kmeans_reduction,
            optimal_k=args.optimal_k,
            soglia=args.soglia,
            dataset_name=args.dataset,
            pre_clustering=args.pre_clustering,
            compare_with_sklearn=args.compare_sklearn,
            dendrogram_display_branches=args.dendrogram_branches,
        )
    else:
        multi_run(
            max_clusters=args.max_clusters,
            k_min=args.k_min,
            k_max=args.k_max,
            optimal_k=args.optimal_k,
            soglia=args.soglia,
            dataset_name=args.dataset,
            pre_clustering=args.pre_clustering,
            compare_with_sklearn=args.compare_sklearn,
            dendrogram_display_branches=args.dendrogram_branches,
        )


if __name__ == "__main__":
    main()

