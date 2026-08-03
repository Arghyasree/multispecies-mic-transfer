#!/usr/bin/env python3
"""Generate protocol-specific multisource adaptation curves for the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = REPO_ROOT / "results" / "evaluation"
OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "manuscript"
OUTPUT_STEM = OUTPUT_DIR / "multisource_adaptation_by_protocol"

SUPPORT_BUDGETS = np.array([0, 1, 5, 10], dtype=int)

TARGETS = {
    "ec": {
        "label": "Ec",
        "source_regime": "kp_plus_se_to_ec",
    },
    "kp": {
        "label": "Kp",
        "source_regime": "se_plus_ec_to_kp",
    },
    "se": {
        "label": "Se",
        "source_regime": "kp_plus_ec_to_se",
    },
}

PROTOCOLS = {
    "random_pair": {
        "title": "(a) Random-pair",
        "path": EVALUATION_DIR
        / "random_pair_pretrained_limited_label.tsv",
        "mean_column": "macro_rmse_mean_across_folds",
        "sd_column": "macro_rmse_sd_across_folds",
        "zero_mean_column": "zero_shot_macro_rmse_mean_across_folds",
        "zero_sd_column": "zero_shot_macro_rmse_sd_across_folds",
    },
    "genome_disjoint": {
        "title": "(b) Genome-disjoint",
        "path": EVALUATION_DIR
        / "genome_disjoint_pretrained_limited_label.tsv",
        "mean_column": "macro_rmse_mean_across_folds",
        "sd_column": "macro_rmse_sd_across_folds",
        "zero_mean_column": "zero_shot_macro_rmse_mean_across_folds",
        "zero_sd_column": "zero_shot_macro_rmse_sd_across_folds",
    },
    "antibiotic_held_out": {
        "title": "(c) Leave-one-antibiotic-out",
        "path": EVALUATION_DIR
        / "antibiotic_held_out_pretrained_limited_label.tsv",
        "mean_column": "macro_rmse_mean_across_held_out_drugs",
        "sd_column": "macro_rmse_sd_across_held_out_drugs",
        "zero_mean_column":
        "zero_shot_macro_rmse_mean_across_held_out_drugs",
        "zero_sd_column":
        "zero_shot_macro_rmse_sd_across_held_out_drugs",
    },
}


def configure_matplotlib() -> None:
    """Set compact defaults suitable for an ACM two-column paper."""
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_protocol_results(
    protocol: dict[str, object],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Read multisource means for one evaluation protocol."""
    path = Path(protocol["path"])

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
    )

    frame["support_budget_percent"] = pd.to_numeric(
        frame["support_budget_percent"],
        errors="raise",
    ).astype(int)

    curves: dict[str, np.ndarray] = {}
    output_rows: list[dict[str, object]] = []

    for target_code, target_spec in TARGETS.items():
        subset = frame.loc[
            (frame["outer_target_code"] == target_code)
            & (
                frame["source_regime_id"]
                == target_spec["source_regime"]
            )
            & (
                frame["model_kind"]
                == "source_pretrained_few_shot"
            )
            & frame["support_budget_percent"].isin([1, 5, 10])
        ].copy()

        subset = subset.sort_values("support_budget_percent")

        if subset["support_budget_percent"].tolist() != [1, 5, 10]:
            raise RuntimeError(
                f"Missing support budgets for {path.name}, "
                f"target {target_code}: "
                f"{subset['support_budget_percent'].tolist()}"
            )

        zero_means = pd.to_numeric(
            subset[protocol["zero_mean_column"]],
            errors="raise",
        ).to_numpy(dtype=float)

        zero_sds = pd.to_numeric(
            subset[protocol["zero_sd_column"]],
            errors="raise",
        ).to_numpy(dtype=float)

        if not np.allclose(zero_means, zero_means[0]):
            raise RuntimeError(
                f"Inconsistent zero-shot means for "
                f"{path.name}, target {target_code}"
            )

        if not np.allclose(zero_sds, zero_sds[0]):
            raise RuntimeError(
                f"Inconsistent zero-shot SDs for "
                f"{path.name}, target {target_code}"
            )

        adapted_means = pd.to_numeric(
            subset[protocol["mean_column"]],
            errors="raise",
        ).to_numpy(dtype=float)

        adapted_sds = pd.to_numeric(
            subset[protocol["sd_column"]],
            errors="raise",
        ).to_numpy(dtype=float)

        means = np.concatenate(
            ([zero_means[0]], adapted_means)
        )
        sds = np.concatenate(
            ([zero_sds[0]], adapted_sds)
        )

        target_label = str(target_spec["label"])
        curves[target_label] = means

        for support, mean, sd in zip(
            SUPPORT_BUDGETS,
            means,
            sds,
            strict=True,
        ):
            output_rows.append(
                {
                    "protocol": str(protocol["title"])[4:],
                    "target": target_label,
                    "target_support_percent": int(support),
                    "macro_rmse_mean": float(mean),
                    "macro_rmse_sd": float(sd),
                }
            )

    return curves, output_rows


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.55),
        sharex=True,
        sharey=True,
    )

    all_output_rows: list[dict[str, object]] = []

    for axis, protocol in zip(
        axes,
        PROTOCOLS.values(),
        strict=True,
    ):
        curves, output_rows = read_protocol_results(protocol)
        all_output_rows.extend(output_rows)

        for target_label, values in curves.items():
            axis.plot(
                SUPPORT_BUDGETS,
                values,
                marker="o",
                linewidth=1.5,
                markersize=4,
                label=target_label,
            )

        axis.set_title(str(protocol["title"]))
        axis.set_xlabel("Target support (%)")
        axis.set_xticks(SUPPORT_BUDGETS)
        axis.set_xlim(-0.35, 10.35)
        axis.grid(
            axis="y",
            linewidth=0.5,
            alpha=0.35,
        )

    for axis in axes:
        axis.set_ylabel("Macro-RMSE")
        axis.tick_params(axis="y", labelleft=True)

    axes[0].set_ylim(1.05, 2.45)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Target",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )

    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        bottom=0.20,
        top=0.79,
        wspace=0.32,
    )

    source_table = pd.DataFrame(all_output_rows)
    source_table.to_csv(
        OUTPUT_DIR / "multisource_adaptation_by_protocol.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    pdf_path = OUTPUT_STEM.with_suffix(".pdf")
    png_path = OUTPUT_STEM.with_suffix(".png")

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    figure.savefig(
        png_path,
        bbox_inches="tight",
        dpi=600,
    )
    plt.close(figure)

    print(f"Wrote {pdf_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {png_path.relative_to(REPO_ROOT)}")
    print(
        "Wrote "
        "results/figures/manuscript/"
        "multisource_adaptation_by_protocol.tsv"
    )


if __name__ == "__main__":
    main()
