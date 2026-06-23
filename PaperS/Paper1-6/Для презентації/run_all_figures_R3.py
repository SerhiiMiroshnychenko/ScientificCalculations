from pathlib import Path

from amount_success_chart import create_amount_success_chart
from changes_success_chart import create_changes_success_chart
from dayofweek_success_chart import create_dayofweek_success_chart
from discount_success_chart import create_discount_success_chart
from messages_success_chart import create_messages_success_chart


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data_collector_extended.csv"
OUTPUT_DIR = BASE_DIR / "R3_outputs"


FIGURES = [
    ("Figure 9", create_messages_success_chart, "messages_success_chart_R3"),
    ("Figure 10", create_changes_success_chart, "changes_success_chart_R3"),
    ("Figure 11", create_amount_success_chart, "amount_success_chart_R3"),
    ("Figure 12", create_discount_success_chart, "discount_success_chart_R3"),
    ("Figure 13", create_dayofweek_success_chart, "dayofweek_success_chart_R3"),
]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    failed = []
    for figure_label, create_func, output_name in FIGURES:
        output_base = OUTPUT_DIR / output_name
        print(f"\n=== Regenerating {figure_label}: {output_name} ===")
        ok = create_func(str(CSV_PATH), str(output_base))
        if not ok:
            failed.append(figure_label)

    if failed:
        raise RuntimeError(f"Failed to regenerate: {', '.join(failed)}")

    print("\nAll R3 figures and statistical reports were regenerated successfully.")


if __name__ == "__main__":
    main()
