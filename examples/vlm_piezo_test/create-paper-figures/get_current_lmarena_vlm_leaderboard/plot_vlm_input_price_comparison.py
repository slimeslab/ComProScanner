#!/usr/bin/env python3
"""Plot VLM ELO score vs input price ($/1M tokens) from vlm_model_score_price.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = "final_vlm_model_score_price.csv"
OUTPUT_PNG = "../plots-raw/vlm_input_price_comparison_base.png"
SHOW_MODEL_NAMES = False


COMPANY_COLORS = {
    "Google": "#efb118",
    "OpenAI": "#4169E1",
    "Anthropic": "#40E0D0",
    "Alibaba": "#7043a5",
    "Baidu": "#3b82f6",
    "Bytedance": "#f97316",
    "Tencent": "#22c55e",
    "xAI": "#ef4444",
    "StepFun": "#a855f7",
    "Moonshot": "#14b8a6",
    "Mistral": "#f43f5e",
    "Other": "#6b7280",
}


def load_and_filter_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"model", "company", "elo_score", "input_price_per_1m"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in CSV: {sorted(missing)}")

    # Keep only rows with usable input-price and score values.
    df["elo_score"] = pd.to_numeric(df["elo_score"], errors="coerce")
    df["input_price_per_1m"] = pd.to_numeric(df["input_price_per_1m"], errors="coerce")
    df = df.dropna(subset=["elo_score", "input_price_per_1m"])

    # Log x-scale needs positive values.
    df = df[df["input_price_per_1m"] >= 0]
    return df


def annotate_without_overlap(ax: plt.Axes, df: pd.DataFrame) -> None:
    x_span = 3.5
    y_span = max(1.0, 1350 - float(df["elo_score"].min()))
    x_step = x_span * 0.015
    y_step = y_span * 0.03

    placed: list[tuple[float, float]] = []
    offsets = [
        (x_step, y_step),
        (x_step, -y_step),
        (2 * x_step, y_step),
        (2 * x_step, -y_step),
        (-2 * x_step, y_step),
        (-2 * x_step, -y_step),
    ]

    # Higher scores labeled first.
    sorted_df = df.sort_values("elo_score", ascending=False)
    for _, row in sorted_df.iterrows():
        x = float(row["input_price_per_1m"])
        y = float(row["elo_score"])
        label = str(row["model"])

        target_x, target_y = x + x_step, y + y_step
        for dx, dy in offsets:
            cand_x = min(3.45, max(0.02, x + dx))
            cand_y = min(1348, max(float(df["elo_score"].min()) + 1, y + dy))
            if all(
                abs(cand_x - px) > x_step * 1.2 or abs(cand_y - py) > y_step * 0.8
                for px, py in placed
            ):
                target_x, target_y = cand_x, cand_y
                break

        placed.append((target_x, target_y))
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(target_x, target_y),
            textcoords="data",
            fontsize=9,
            alpha=0.9,
            arrowprops={"arrowstyle": "-", "lw": 0.5, "alpha": 0.45},
        )


def plot_input_price_vs_elo(
    df: pd.DataFrame, out_path: Path, show_model_names: bool = SHOW_MODEL_NAMES
) -> None:
    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    # Shaded region: high performance and relatively low input cost.
    x_shade = [0, 1.5, 1.5, 0]
    y_shade = [1250, 1250, 1475, 1475]
    ax.fill(
        x_shade,
        y_shade,
        color="#f5427e",
        alpha=0.15,
        zorder=0,
        label="High Performance,\nLow Input Cost",
    )

    # Stable plotting order for legend readability.
    company_order = [
        "Google",
        "OpenAI",
        "Anthropic",
        "Alibaba",
        "Baidu",
        "Bytedance",
        "Tencent",
        "xAI",
        "StepFun",
        "Moonshot",
        "Mistral",
        "Other",
    ]

    for company in company_order:
        group = df[df["company"] == company]
        if group.empty:
            continue

        color = COMPANY_COLORS.get(company, COMPANY_COLORS["Other"])
        plt.scatter(
            group["input_price_per_1m"],
            group["elo_score"],
            c=color,
            s=140,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.5,
            label=company,
            zorder=2,
        )

    # Any company not listed above is grouped under "Other".
    known = set(company_order)
    other_df = df[~df["company"].isin(known)]
    if not other_df.empty:
        plt.scatter(
            other_df["input_price_per_1m"],
            other_df["elo_score"],
            c=COMPANY_COLORS["Other"],
            s=140,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.5,
            label="Other",
            zorder=2,
        )

    plt.xlabel("Input Cost ($/1M Tokens)", fontsize=16, labelpad=15)
    plt.ylabel("Arena ELO Score", fontsize=16, labelpad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Requested fixed axes for this plot.
    plt.xlim(0, 3.5)
    plt.ylim(float(df["elo_score"].min()) - 5, 1325)
    plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, zorder=1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if show_model_names:
        annotate_without_overlap(ax, df)

    plt.legend(
        loc="upper right", frameon=True, facecolor="white", framealpha=1.0, fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")


def main() -> None:
    here = Path(__file__).resolve().parent
    csv_path = here / INPUT_CSV
    out_path = here / OUTPUT_PNG

    data = load_and_filter_data(csv_path)
    if data.empty:
        print("No rows with valid positive input cost were found. Plot not created.")
        return

    plot_input_price_vs_elo(data, out_path, show_model_names=SHOW_MODEL_NAMES)
    print(f"Saved plot: {out_path}")
    print(f"Rows plotted: {len(data)}")


if __name__ == "__main__":
    main()
