import argparse
from pathlib import Path

from .plot import (
    plot_comparison_deltas,
    plot_comparison_metrics_bar,
    plot_confusion_pair_deltas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera plot di confronto custom vs scikit-learn a partire da comparison_with_sklearn.csv"
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path al file comparison_with_sklearn.csv",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Genera tutti e tre i plot (bar, delta metriche, delta TP/FP/TN/FN).",
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Genera solo il plot a barre metriche custom vs sklearn.",
    )
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Genera solo il plot dei delta metriche.",
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="Genera solo il plot dei delta TP/FP/TN/FN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"File non trovato: {csv_path}")

    run_all = args.all or not (args.bar or args.delta or args.pairs)

    generated = []
    if run_all or args.bar:
        generated.append(plot_comparison_metrics_bar(str(csv_path)))
    if run_all or args.delta:
        generated.append(plot_comparison_deltas(str(csv_path)))
    if run_all or args.pairs:
        generated.append(plot_confusion_pair_deltas(str(csv_path)))

    print("\nFile generati:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
