import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggrega i risultati di clustering e mostra le migliori combinazioni di iperparametri.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Filtra un dataset specifico (es. iris_dataset, Frogs_MFCCs). Se omesso, analizza tutti i dataset.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Numero di risultati migliori da mostrare e salvare.",
    )
    parser.add_argument(
        "--metric",
        default="f1_score",
        help="Metrica di ordinamento (es. f1_score, rand_index, recall, precision).",
    )
    return parser.parse_args()


def collect_results(project_root: Path, dataset_filter: str | None = None) -> pd.DataFrame:
    assets_dir = project_root / "assets"
    results_rows: list[pd.DataFrame] = []

    for dataset_dir in assets_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue

        results_dir = dataset_dir / "Results"
        if not results_dir.exists():
            continue

        for csv_path in results_dir.rglob("evaluation_results.csv"):
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"Errore nella lettura di {csv_path}: {exc}")
                continue

            relative_parent = csv_path.parent.relative_to(results_dir)
            parts = relative_parent.parts
            k_reduction = parts[0] if len(parts) > 0 else "unknown"
            method_distance = parts[1] if len(parts) > 1 else "unknown"

            df["dataset"] = dataset_dir.name
            df["k_folder"] = k_reduction
            df["method_distance"] = method_distance
            df["source_file"] = str(csv_path)
            results_rows.append(df)

    if not results_rows:
        return pd.DataFrame()

    return pd.concat(results_rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    all_results = collect_results(project_root, args.dataset)
    if all_results.empty:
        print("Nessun risultato trovato")
        return

    if args.metric not in all_results.columns:
        available = ", ".join(sorted(all_results.columns))
        raise ValueError(f"Metrica '{args.metric}' non trovata. Colonne disponibili: {available}")

    top_results = all_results.sort_values(args.metric, ascending=False).head(args.top)

    cols = [
        "dataset",
        "method_distance",
        "k_means_reduction",
        "clusters",
        "tp",
        "fp",
        "tn",
        "fn",
        "rand_index",
        "precision",
        "recall",
        "f1_score",
    ]
    cols_to_show = [c for c in cols if c in top_results.columns]

    output_name = "best_results.json" if not args.dataset else f"best_results_{args.dataset}.json"
    output_path = project_root / "assets" / output_name
    top_results.to_json(output_path, orient="records", indent=4)

    print(f"\nTop {args.top} combinazioni ordinate per {args.metric}:")
    print(top_results[cols_to_show])
    print(f"\nRisultati salvati in: {output_path}")


if __name__ == "__main__":
    main()