"""Render a presentation figure from phase-segmentation validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


INK = "#17202A"
MUTED = "#66727F"
GRID = "#D9E0E6"
FIXED = "#8A98A6"
AUTO = "#176B87"
IMPROVED = "#2A7F62"
EQUAL = "#AAB4BD"
WORSENED = "#C9793A"


def configure_font() -> None:
    candidates = ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (font for font in candidates if font in installed), "DejaVu Sans"
    )
    plt.rcParams["axes.unicode_minus"] = False


def rate(value: float) -> float:
    return 100.0 * float(value)


def render(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cohort = summary["cohorts"]["movement_table_560_with_local_npy"]
    checkpoints = cohort["realign_checkpoints"]

    relaxed = [
        rate(cohort["fixed"]["relaxed_count_success_rate"]),
        rate(cohort["auto"]["relaxed_count_success_rate"]),
    ]
    strict = [
        rate(cohort["fixed"]["strict_count_success_rate"]),
        rate(cohort["auto"]["strict_count_success_rate"]),
    ]
    errors = [
        float(checkpoints["fixed_mae_frames"]),
        float(checkpoints["auto_mae_frames"]),
    ]
    change_counts = [
        int(checkpoints["auto_improved_n"]),
        int(checkpoints["auto_equal_n"]),
        int(checkpoints["auto_worsened_n"]),
    ]

    configure_font()
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.95,
        top=0.76,
        bottom=0.15,
        width_ratios=[1.05, 0.95],
        height_ratios=[1, 1],
        hspace=0.52,
        wspace=0.32,
    )

    fig.text(
        0.07,
        0.91,
        "반복 수를 맞추는 비율은 높아졌지만, 경계 검수는 남아 있다",
        fontsize=25,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.855,
        "Stanford 공개 pose trajectory 456명 · 고정 평활값과 반복 수 기반 자동 선택 비교",
        fontsize=13,
        color=MUTED,
    )
    fig.add_artist(
        plt.Line2D([0.07, 0.95], [0.815, 0.815], color=INK, linewidth=1.2)
    )

    ax_rate = fig.add_subplot(grid[:, 0])
    x = np.arange(2)
    width = 0.32
    fixed_bars = ax_rate.bar(
        x - width / 2,
        [relaxed[0], strict[0]],
        width,
        color=FIXED,
        label="고정값",
    )
    auto_bars = ax_rate.bar(
        x + width / 2,
        [relaxed[1], strict[1]],
        width,
        color=AUTO,
        label="자동 선택",
    )
    ax_rate.set_title("반복 수 QC 통과율", loc="left", fontsize=17, fontweight="bold")
    ax_rate.set_xticks(x, ["완화 기준\n기립 5 · 착석 5~6", "엄격 기준\n기립 5 · 착석 6"])
    ax_rate.set_ylim(0, 100)
    ax_rate.set_ylabel("통과율 (%)", fontsize=11, color=MUTED)
    ax_rate.grid(axis="y", color=GRID, linewidth=0.8)
    ax_rate.set_axisbelow(True)
    ax_rate.spines[["top", "right", "left"]].set_visible(False)
    ax_rate.spines["bottom"].set_color(GRID)
    ax_rate.tick_params(axis="y", colors=MUTED)
    ax_rate.tick_params(axis="x", length=0, pad=10)
    ax_rate.legend(frameon=False, loc="upper right", ncols=2, fontsize=11)
    for bars in (fixed_bars, auto_bars):
        for bar in bars:
            ax_rate.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color=INK,
            )

    ax_error = fig.add_subplot(grid[0, 1])
    y = np.arange(2)
    ax_error.hlines(y, 0, errors, color=GRID, linewidth=5)
    ax_error.scatter(errors, y, s=150, color=[FIXED, AUTO], zorder=3)
    ax_error.set_yticks(y, ["고정값", "자동 선택"])
    ax_error.invert_yaxis()
    ax_error.set_xlim(0, 25)
    ax_error.set_xlabel("평균 절대오차 (frame)", fontsize=10, color=MUTED)
    ax_error.set_title(
        "부분 착석 경계 57개", loc="left", fontsize=17, fontweight="bold"
    )
    ax_error.grid(axis="x", color=GRID, linewidth=0.8)
    ax_error.set_axisbelow(True)
    ax_error.spines[["top", "right", "left"]].set_visible(False)
    ax_error.spines["bottom"].set_color(GRID)
    ax_error.tick_params(axis="y", length=0, pad=8)
    ax_error.tick_params(axis="x", colors=MUTED)
    for yi, value in zip(y, errors):
        ax_error.text(
            value + 0.7,
            yi,
            f"{value:.2f}",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

    ax_change = fig.add_subplot(grid[1, 1])
    left = 0
    labels = ["개선 29", "동일 12", "악화 16"]
    colors = [IMPROVED, EQUAL, WORSENED]
    for count, label, color in zip(change_counts, labels, colors):
        ax_change.barh([0], [count], left=left, height=0.42, color=color, label=label)
        ax_change.text(
            left + count / 2,
            0,
            str(count),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white" if color != EQUAL else INK,
        )
        left += count
    ax_change.set_xlim(0, sum(change_counts))
    ax_change.set_yticks([])
    ax_change.set_xticks([])
    ax_change.set_title(
        "자동 선택 후 경계 변화", loc="left", fontsize=17, fontweight="bold"
    )
    ax_change.legend(frameon=False, loc="lower left", ncols=3, fontsize=10)
    ax_change.spines[:].set_visible(False)

    fig.text(
        0.07,
        0.065,
        "해석 범위  반복 수 QC의 부분 자동화 · realign은 일부 수정 경계이며 전체 사람 검수 정답이 아님 · 반복 수 일치는 경계 정확도를 보장하지 않음",
        fontsize=10.5,
        color=MUTED,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
