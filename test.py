from __future__ import annotations

import os
from pathlib import Path
import config
import pandas as pd


IMBALANCE_RATIO_THRESHOLD = 10.0  # max/min class ratio to flag as imbalanced


def get_target_column(df: pd.DataFrame) -> str:
    if "class" in df.columns:
        return "class"
    return df.columns[-1]


def summarize_dataset(csv_path: Path) -> dict | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read {csv_path}: {exc}")
        return None

    if df.empty:
        print(f"[WARN] Empty dataset: {csv_path}")
        return None

    target_col = get_target_column(df)
    counts = df[target_col].value_counts(dropna=False)
    total = int(counts.sum())
    proportions = (counts / total).sort_values(ascending=False)

    min_prop = proportions.iloc[-1]
    max_prop = proportions.iloc[0]
    imbalance_ratio = float("inf") if min_prop == 0 else float(max_prop / min_prop)
    num_features = len(df.columns) - 1  # 排除目標欄位
    file = csv_path.split('\\')[-1].split('.csv')[0]
    return {
        "path": file,
        "total": total,
        "num_features": num_features,
        "num_classes": int(len(counts)),
        "max_prop": float(max_prop),
        "min_prop": float(min_prop),
        "imbalance_ratio": imbalance_ratio,
        "counts": counts,
        "proportions": proportions,
        "target_col": target_col,
    }


def format_counts(counts: pd.Series, proportions: pd.Series) -> str:
    parts = []
    for label, count in counts.items():
        pct = proportions[label] * 100
        parts.append(f"{label}: {int(count)} ({pct:.2f}%)")
    return ", ".join(parts)


def main() -> None:
    dataset_root = Path("datasets") / "離散化資料集" / "多類別"
    dataset_list = config.DATASET_LIST

    summaries = []
    for dataset in dataset_list:
        print(f"Processing dataset: {dataset}")
        csv_path = os.path.join(dataset_root, dataset + '.csv')
        summary = summarize_dataset(csv_path)
        if summary:
            summaries.append(summary)

    # summaries.sort(key=lambda x: x["imbalance_ratio"], reverse=True)  # 註解掉以保持 DATASET_LIST 的順序

    output_rows = []

    print("=== Class Balance Summary ===")
    for s in summaries:
        rel_path = s["path"]
        is_imbalanced = s["imbalance_ratio"] >= IMBALANCE_RATIO_THRESHOLD
        flag = "IMBALANCED" if is_imbalanced else "ok"
        max_pct = s["max_prop"] * 100
        min_pct = s["min_prop"] * 100
        print(
            f"{rel_path} | target={s['target_col']} | n={s['total']} | "
            f"features={s['num_features']} | classes={s['num_classes']} | max={max_pct:.2f}% | "
            f"min={min_pct:.2f}% | ratio={s['imbalance_ratio']:.2f} | {flag}"
        )
        print(f"  counts: {format_counts(s['counts'], s['proportions'])}")
        output_rows.append(
            {
                "dataset": rel_path,
                "target_column": s["target_col"],
                "total_samples": s["total"],
                "num_features": s["num_features"],
                "num_classes": s["num_classes"],
                "max_class_pct": round(max_pct, 6),
                "min_class_pct": round(min_pct, 6),
                "imbalance_ratio": round(s["imbalance_ratio"], 6),
                "status": flag,
                "class_counts": format_counts(s["counts"], s["proportions"]),
            }
        )

    output_path = Path("class_balance_summary.csv")
    pd.DataFrame(output_rows).to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
