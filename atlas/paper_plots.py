from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Helvetica, Arial, sans-serif" font-weight="{weight}" fill="#111827">{escape(text)}</text>'
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#9ca3af", width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr} />'


def _svg_circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "none", stroke_width: float = 1.0) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def _svg_rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str = "none", stroke_width: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def _write_svg(path: Path, *, width: int, height: int, elements: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _svg_rect(0, 0, width, height, fill="#ffffff"),
        *elements,
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def render_horizon_scatter_svg(points_csv: Path, output_svg: Path) -> None:
    rows = _read_csv_rows(points_csv)
    if not rows:
        raise ValueError("points CSV must not be empty")

    width = 960
    height = 540
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_y1 = margin_top + plot_height

    palette = {
        1: "#0f766e",
        3: "#b45309",
        10: "#7c3aed",
    }

    horizons = sorted({int(row["horizon"]) for row in rows})
    grouped: dict[int, list[dict[str, str]]] = {h: [] for h in horizons}
    for row in rows:
        grouped[int(row["horizon"])].append(row)

    max_value = max(float(row["final_return_mean"]) for row in rows)
    y_max = max(1.0, max_value * 1.15)

    def x_for_horizon(horizon: int) -> float:
        index = horizons.index(horizon)
        return plot_x0 + plot_width * ((index + 0.5) / len(horizons))

    def y_for_value(value: float) -> float:
        return plot_y1 - (value / y_max) * plot_height

    elements: list[str] = []
    elements.append(_svg_text(width / 2, 36, "Figure 2. Final Fine-Tune Return by Backup Horizon", size=22, anchor="middle", weight="bold"))
    elements.append(_svg_text(width / 2, 58, "Secondary robustness observation under the fixed severe shift", size=13, anchor="middle"))

    # Axes and grid.
    for tick_index in range(6):
        value = y_max * tick_index / 5.0
        y = y_for_value(value)
        elements.append(_svg_line(plot_x0, y, plot_x0 + plot_width, y, stroke="#e5e7eb", width=1.0))
        elements.append(_svg_text(plot_x0 - 10, y + 5, f"{value:.2f}", size=12, anchor="end"))

    elements.append(_svg_line(plot_x0, plot_y0, plot_x0, plot_y1, stroke="#111827", width=1.5))
    elements.append(_svg_line(plot_x0, plot_y1, plot_x0 + plot_width, plot_y1, stroke="#111827", width=1.5))
    elements.append(_svg_text(26, plot_y0 + plot_height / 2, "Final return", size=14))

    for horizon in horizons:
        cx = x_for_horizon(horizon)
        elements.append(_svg_text(cx, plot_y1 + 28, f"n = {horizon}", size=14, anchor="middle", weight="bold"))
        horizon_rows = grouped[horizon]
        count = len(horizon_rows)
        for index, row in enumerate(sorted(horizon_rows, key=lambda item: int(item["seed"]))):
            jitter = 0.0 if count == 1 else ((index / (count - 1)) - 0.5) * 40.0
            value = float(row["final_return_mean"])
            cy = y_for_value(value)
            elements.append(
                _svg_circle(
                    cx + jitter,
                    cy,
                    6.0,
                    fill=palette.get(horizon, "#2563eb"),
                    stroke="#ffffff",
                    stroke_width=1.5,
                )
            )
        mean_value = sum(float(row["final_return_mean"]) for row in horizon_rows) / len(horizon_rows)
        mean_y = y_for_value(mean_value)
        elements.append(_svg_line(cx - 30, mean_y, cx + 30, mean_y, stroke="#111827", width=3.0))
        elements.append(_svg_text(cx, mean_y - 10, f"{mean_value:.3f}", size=12, anchor="middle"))

    _write_svg(output_svg, width=width, height=height, elements=elements)


def render_warning_summary_svg(summary_csv: Path, output_svg: Path) -> None:
    rows = _read_csv_rows(summary_csv)
    if not rows:
        raise ValueError("warning summary CSV must not be empty")

    width = 960
    height = 540
    margin_left = 90
    margin_right = 90
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_y1 = margin_top + plot_height

    max_count = max(int(row["run_count"]) for row in rows)

    def y_count(value: float) -> float:
        return plot_y1 - (value / max_count) * plot_height if max_count > 0 else plot_y1

    def y_auc(value: float) -> float:
        return plot_y1 - value * plot_height

    elements: list[str] = []
    elements.append(_svg_text(width / 2, 36, "Figure 1. Warning-Signal Pilot Outcome", size=22, anchor="middle", weight="bold"))
    elements.append(_svg_text(width / 2, 58, "Warning counts, collapse counts, and pilot ROC-AUC", size=13, anchor="middle"))

    for tick in range(max_count + 1):
        y = y_count(float(tick))
        elements.append(_svg_line(plot_x0, y, plot_x0 + plot_width, y, stroke="#e5e7eb", width=1.0))
        elements.append(_svg_text(plot_x0 - 10, y + 5, str(tick), size=12, anchor="end"))

    # Right-side AUC ticks.
    for tick in (0.0, 0.5, 1.0):
        y = y_auc(tick)
        elements.append(_svg_text(plot_x0 + plot_width + 10, y + 5, f"{tick:.1f}", size=12))
    elements.append(_svg_line(plot_x0, y_auc(0.5), plot_x0 + plot_width, y_auc(0.5), stroke="#9ca3af", width=1.0, dash="6 4"))

    elements.append(_svg_line(plot_x0, plot_y0, plot_x0, plot_y1, stroke="#111827", width=1.5))
    elements.append(_svg_line(plot_x0, plot_y1, plot_x0 + plot_width, plot_y1, stroke="#111827", width=1.5))
    elements.append(_svg_line(plot_x0 + plot_width, plot_y0, plot_x0 + plot_width, plot_y1, stroke="#111827", width=1.0))
    elements.append(_svg_text(28, plot_y0 + plot_height / 2, "Run count", size=14))
    elements.append(_svg_text(plot_x0 + plot_width + 56, plot_y0 + plot_height / 2, "ROC-AUC", size=14, anchor="middle"))

    group_width = plot_width / len(rows)
    bar_width = 34.0
    for idx, row in enumerate(rows):
        center_x = plot_x0 + group_width * (idx + 0.5)
        warning_count = int(row["runs_with_warning"])
        collapse_count = int(row["runs_with_collapse"])
        warning_y = y_count(float(warning_count))
        collapse_y = y_count(float(collapse_count))
        elements.append(_svg_rect(center_x - 42, warning_y, bar_width, plot_y1 - warning_y, fill="#2563eb"))
        elements.append(_svg_rect(center_x + 8, collapse_y, bar_width, plot_y1 - collapse_y, fill="#dc2626"))
        elements.append(_svg_text(center_x - 25, warning_y - 8, str(warning_count), size=12, anchor="middle"))
        elements.append(_svg_text(center_x + 25, collapse_y - 8, str(collapse_count), size=12, anchor="middle"))
        pilot_label = row["pilot_id"].replace("pilot_gate_", "").replace("_", " ")
        elements.append(_svg_text(center_x, plot_y1 + 28, pilot_label, size=13, anchor="middle", weight="bold"))
        elements.append(_svg_text(center_x, plot_y1 + 48, "warnings / collapses", size=11, anchor="middle"))

        roc_auc_raw = row["global_roc_auc"]
        if roc_auc_raw not in ("", None):
            auc = float(roc_auc_raw)
            auc_y = y_auc(auc)
            elements.append(_svg_circle(center_x, auc_y, 6.0, fill="#111827"))
            elements.append(_svg_text(center_x, auc_y - 10, f"{auc:.3f}", size=12, anchor="middle"))
        else:
            elements.append(_svg_text(center_x, plot_y0 + 18, "AUC undefined", size=11, anchor="middle"))

    # Legend
    legend_y = height - 26
    elements.append(_svg_rect(140, legend_y - 12, 14, 14, fill="#2563eb"))
    elements.append(_svg_text(162, legend_y, "runs with warning", size=12))
    elements.append(_svg_rect(320, legend_y - 12, 14, 14, fill="#dc2626"))
    elements.append(_svg_text(342, legend_y, "runs with collapse", size=12))
    elements.append(_svg_circle(515, legend_y - 5, 6, fill="#111827"))
    elements.append(_svg_text(530, legend_y, "global ROC-AUC", size=12))

    _write_svg(output_svg, width=width, height=height, elements=elements)


def render_all_paper_figures(artifacts_dir: Path, output_dir: Path) -> dict[str, str]:
    figure1 = output_dir / "figure1_warning_pilot_summary.svg"
    figure2 = output_dir / "figure2_horizon_final_return.svg"
    render_warning_summary_svg(artifacts_dir / "warning_pilot_summary.csv", figure1)
    render_horizon_scatter_svg(artifacts_dir / "horizon_final_return_points.csv", figure2)
    return {
        "figure1_svg": str(figure1),
        "figure2_svg": str(figure2),
    }
