from __future__ import annotations

import json
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evaluation" / "figures"
COLORS = ("#176b62", "#b65b3d", "#405b78")


def text(x: float, y: float, value: str, *, size: int = 16, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="#17262d">{escape(value)}</text>'
    )


def grouped_bar_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
) -> None:
    width, height = 1100, 650
    left, right, top, bottom = 95, 40, 145, 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(categories)
    usable_group = group_width * 0.72
    bar_width = usable_group / len(series)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        text(left, 48, title, size=28, weight=700),
        text(left, 78, subtitle, size=15),
    ]

    legend_x = left
    for index, (label, _) in enumerate(series):
        color = COLORS[index]
        parts.append(f'<rect x="{legend_x}" y="101" width="15" height="15" rx="2" fill="{color}"/>')
        parts.append(text(legend_x + 23, 114, label, size=14))
        legend_x += 145

    for tick in range(5):
        value = tick / 4
        y = top + plot_height - value * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#dedbd2"/>')
        parts.append(text(left - 14, y + 5, f"{value:.2f}", size=13, anchor="end"))

    for category_index, category in enumerate(categories):
        group_left = left + category_index * group_width + (group_width - usable_group) / 2
        for series_index, (_, values) in enumerate(series):
            value = values[category_index]
            bar_height = value * plot_height
            x = group_left + series_index * bar_width
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{bar_width - 4:.1f}" '
                f'height="{bar_height:.1f}" rx="3" fill="{COLORS[series_index]}"/>'
            )
            parts.append(text(x + bar_width / 2, y - 8, f"{value:.3f}", size=12, anchor="middle", weight=600))
        parts.append(
            text(
                left + (category_index + 0.5) * group_width,
                top + plot_height + 32,
                category,
                size=14,
                anchor="middle",
            )
        )

    parts.extend(
        [
            (
                f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
                f'y2="{top + plot_height}" stroke="#17262d"/>'
            ),
            text(left, height - 24, "Values read from committed frozen evaluation artifacts.", size=12),
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_retrieval() -> None:
    source = ROOT / "evaluation" / "results" / "tat_dqa_locked_test_selected.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    metric_keys = ("hit_at_1", "hit_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
    series = [
        (mode.title(), [result["metrics"][mode][metric] for metric in metric_keys])
        for mode in ("keyword", "semantic", "hybrid")
    ]
    grouped_bar_chart(
        OUTPUT_DIR / "retrieval_locked_test.svg",
        title="Locked retrieval performance",
        subtitle="20 TAT-DQA questions; higher is better for every metric.",
        categories=["Hit@1", "Hit@5", "Recall@10", "MRR@10", "nDCG@10"],
        series=series,
    )


def render_generation() -> None:
    result_path = ROOT / "evaluation" / "results" / "generation_locked_test_qwen.json"
    manifest_path = ROOT / "evaluation" / "generation" / "locked_test_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    answer_types = {question["uid"]: question.get("answer_type") for question in manifest["questions"]}
    coverage_by_type: dict[str, list[float]] = defaultdict(list)
    for question in result["questions"]:
        answer_type = answer_types.get(question["uid"])
        if question["expected_status"] == "answered" and answer_type:
            coverage_by_type[answer_type].append(question["reference_coverage"])

    categories = ("span", "multi-span", "arithmetic")
    values = [sum(coverage_by_type[name]) / len(coverage_by_type[name]) for name in categories]
    values.append(result["metrics"]["refusal_accuracy"])
    grouped_bar_chart(
        OUTPUT_DIR / "generation_locked_test.svg",
        title="Selected Qwen locked-test outcomes",
        subtitle="Reference coverage by answer type; unanswerable questions use refusal accuracy.",
        categories=["Span", "Multi-span", "Arithmetic", "Unanswerable"],
        series=[("Success rate", values)],
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_retrieval()
    render_generation()
    print(f"Rendered report figures in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
