#!/usr/bin/env python3
"""
Create a paper-style hard-case table for LFW 1:1 verification.

The script selects only pairs that Direct judges correctly:
  1. the hardest genuine pair: lowest correct Direct similarity score;
  2. the most confusing impostor pair: highest correct Direct similarity score.

It then plots the selected image pairs and each method's similarity score.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


FONT_SIZE_PT = 25
GENUINE_COLOR = "#2b8a3e"
IMPOSTOR_COLOR = "#d62728"
CORRECT_MARK = "√"
WRONG_MARK = "×"


@dataclass(frozen=True)
class PairRecord:
    pair_index: int
    query_id: int
    compare_id: int
    label: int
    direct_score: float


METHOD_SCORE_FILES = [
    ("Direct", Path("results/baseline/lfw/score.list")),
    ("IronMask", Path("results/ironmask/lfw/score.list")),
    ("FHE", Path("results/sfm/lfw/score.list")),
    ("Subspace", Path("results/ase/lfw/score.list")),
    ("SecureVector-512", Path("results/securevector/lfw/score.list")),
    (
        "Ours-F",
        Path("results") / "sv_dj_cluster" / "1对1匹配_s=1" / "lfw" / "score.list",
    ),
    (
        "Ours-S",
        Path("results") / "sv_dj_cluster" / "1对1匹配_s=2" / "lfw" / "score.list",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize hardest LFW 1:1 genuine/impostor examples."
    )
    parser.add_argument("--pair-list", default="data/lfw/pair.list")
    parser.add_argument("--direct-score-list", default="results/baseline/lfw/score.list")
    parser.add_argument(
        "--image-path-list", default="data/lfw/lfw_numbered_image_paths.list"
    )
    parser.add_argument(
        "--original-path-list", default="data/lfw/lfw_original_image_paths.list"
    )
    parser.add_argument("--image-dir", default="image_data/lfw_numbered")
    parser.add_argument(
        "--out-png",
        default="results/visualizations/lfw_1v1_hard_cases_table.png",
    )
    parser.add_argument(
        "--out-csv",
        default="results/visualizations/lfw_1v1_hard_cases_table.csv",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=FONT_SIZE_PT)
    return parser.parse_args()


def configure_fonts(font_size):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "SimSun",
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": font_size,
            "axes.unicode_minus": False,
        }
    )


def read_path_list(path_list):
    paths = {}
    path_list = Path(path_list)
    if not path_list.is_file():
        return paths

    with path_list.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(f"{path_list}:{line_no}: expected id<TAB>path")
            paths[int(parts[0])] = Path(parts[1])
    return paths


def build_image_paths(path_list, image_dir):
    paths = read_path_list(path_list)
    if paths:
        return paths

    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Missing image path list {path_list} and fallback directory {image_dir}"
        )
    return {int(path.stem): path for path in image_dir.glob("*.jpg") if path.stem.isdigit()}


def compact_name(path):
    if not path:
        return ""
    path = Path(path)
    if path.parent.name:
        return path.parent.name.replace("_", " ")
    return path.stem.replace("_", " ")


def load_scored_pairs(pair_list, direct_score_list):
    records = []
    pair_list = Path(pair_list)
    direct_score_list = Path(direct_score_list)

    with pair_list.open("r", encoding="utf-8") as f_pair, direct_score_list.open(
        "r", encoding="utf-8"
    ) as f_score:
        for idx, (pair_line, score_line) in enumerate(zip(f_pair, f_score)):
            pair_parts = pair_line.strip().split()
            score_parts = score_line.strip().split()
            if len(pair_parts) != 3 or len(score_parts) != 3:
                continue

            query_id, compare_id, label = map(int, pair_parts)
            score_query, score_compare = map(int, score_parts[:2])
            score = float(score_parts[2])
            if (query_id, compare_id) != (score_query, score_compare):
                raise ValueError(
                    "pair list and Direct score list are not aligned at "
                    f"row {idx}: pair=({query_id}, {compare_id}), "
                    f"score=({score_query}, {score_compare})"
                )

            records.append(PairRecord(idx, query_id, compare_id, label, score))
    return records


def direct_is_correct(record, direct_threshold):
    pred_same = record.direct_score >= direct_threshold
    return pred_same == (record.label == 1)


def select_hard_cases(records, direct_threshold):
    direct_correct = [
        record for record in records if direct_is_correct(record, direct_threshold)
    ]
    genuine = [record for record in direct_correct if record.label == 1]
    impostor = [record for record in direct_correct if record.label == 0]
    if not genuine or not impostor:
        raise RuntimeError(
            "Need both Direct-correct genuine and impostor pairs to select hard cases."
        )

    hard_genuine = min(genuine, key=lambda record: (record.direct_score, record.pair_index))
    hard_impostor = max(
        impostor, key=lambda record: (record.direct_score, -record.pair_index)
    )
    return [("难正样本", hard_genuine), ("难负样本", hard_impostor)]


def read_scores_for_pairs(score_file, pairs):
    wanted = {(record.query_id, record.compare_id) for _, record in pairs}
    scores = {}
    score_file = Path(score_file)

    with score_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                query_id, compare_id = int(parts[0]), int(parts[1])
                score = float(parts[2])
            except ValueError as exc:
                raise ValueError(f"{score_file}:{line_no}: malformed score line") from exc

            key = (query_id, compare_id)
            reverse_key = (compare_id, query_id)
            if key in wanted:
                scores[key] = score
            elif reverse_key in wanted:
                scores[reverse_key] = score

            if len(scores) == len(wanted):
                break

    missing = wanted - set(scores)
    if missing:
        raise RuntimeError(f"Missing scores in {score_file}: {sorted(missing)}")
    return scores


def read_labeled_scores(pair_list, score_file):
    labeled_scores = []
    pair_list = Path(pair_list)
    score_file = Path(score_file)
    with pair_list.open("r", encoding="utf-8") as f_pair, score_file.open(
        "r", encoding="utf-8"
    ) as f_score:
        for idx, (pair_line, score_line) in enumerate(zip(f_pair, f_score)):
            pair_parts = pair_line.strip().split()
            score_parts = score_line.strip().split()
            if len(pair_parts) != 3 or len(score_parts) != 3:
                continue
            query_id, compare_id, label = map(int, pair_parts)
            score_query, score_compare = map(int, score_parts[:2])
            score = float(score_parts[2])
            if (query_id, compare_id) != (score_query, score_compare):
                raise ValueError(
                    "pair list and score list are not aligned at "
                    f"row {idx}: pair=({query_id}, {compare_id}), "
                    f"score=({score_query}, {score_compare})"
                )
            labeled_scores.append((score, label))
    return labeled_scores


def best_threshold(labeled_scores):
    if not labeled_scores:
        raise RuntimeError("No labeled scores available for threshold selection.")

    unique_scores = sorted({score for score, _label in labeled_scores})
    if len(unique_scores) == 1:
        return unique_scores[0]

    candidates = [unique_scores[0] - 1e-6]
    candidates.extend(
        (left + right) / 2.0 for left, right in zip(unique_scores, unique_scores[1:])
    )
    candidates.append(unique_scores[-1] + 1e-6)

    def accuracy(threshold):
        correct = 0
        for score, label in labeled_scores:
            pred_same = score >= threshold
            correct += pred_same == (label == 1)
        return correct

    return max(candidates, key=lambda threshold: (accuracy(threshold), threshold))


def compute_method_thresholds(pair_list):
    thresholds = {}
    for method_name, score_file in METHOD_SCORE_FILES:
        thresholds[method_name] = best_threshold(read_labeled_scores(pair_list, score_file))
    return thresholds


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def label_color(label):
    if label == 1:
        return GENUINE_COLOR
    if label == 0:
        return IMPOSTOR_COLOR
    return "#495057"


def draw_header(ax, text, font_size):
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=font_size,
        color="#000000",
        linespacing=1.2,
    )


def draw_row_label(ax, text, font_size):
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=font_size,
        color="#000000",
    )


def draw_pair_cell(ax, record, image_paths, original_paths, font_size):
    ax.set_axis_off()
    image_specs = [
        (
            0.25,
            "Query ID",
            record.query_id,
            image_paths.get(record.query_id),
            compact_name(original_paths.get(record.query_id)),
        ),
        (
            0.75,
            "Comparison ID",
            record.compare_id,
            image_paths.get(record.compare_id),
            compact_name(original_paths.get(record.compare_id)),
        ),
    ]

    for center_x, title, image_id, image_path, person_name in image_specs:
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"Missing image for ID {image_id}: {image_path}")

        ax.text(
            center_x,
            0.94,
            f"{title} {image_id}",
            ha="center",
            va="top",
            fontsize=font_size,
            fontweight="bold",
            color="#000000",
        )

        image_ax = ax.inset_axes([center_x - 0.18, 0.28, 0.36, 0.54])
        image_ax.imshow(load_image(image_path))
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.9)

        ax.text(
            center_x,
            0.16,
            person_name,
            ha="center",
            va="center",
            fontsize=font_size * 0.86,
            fontweight="bold",
            color="#000000",
        )


def draw_result_cell(ax, correct, font_size):
    ax.set_axis_off()
    mark = CORRECT_MARK if correct else WRONG_MARK
    ax.text(
        0.5,
        0.5,
        mark,
        ha="center",
        va="center",
        fontsize=font_size * 2.1,
        color="#000000",
        fontweight="bold",
    )


def add_legend(fig, font_size):
    fig.text(
        0.5,
        0.022,
        f"{CORRECT_MARK}: correct judgment     {WRONG_MARK}: incorrect judgment",
        ha="center",
        va="center",
        fontsize=font_size * 0.9,
        color="#000000",
    )


def add_column_separators(fig, width_ratios, margins):
    left, right, bottom, top = margins
    total = sum(width_ratios)
    cumulative = 0.0
    for ratio in width_ratios[:-1]:
        cumulative += ratio
        x = left + (right - left) * cumulative / total
        fig.add_artist(
            Line2D(
                [x, x],
                [bottom, top],
                transform=fig.transFigure,
                color="#000000",
                linewidth=1.0,
                linestyle=(0, (4, 3)),
            )
        )


def plot_table(
    pairs,
    method_scores,
    method_thresholds,
    image_paths,
    original_paths,
    out_png,
    dpi,
    font_size,
):
    method_names = [name for name, _ in METHOD_SCORE_FILES]
    headers = ["", "", *method_names]
    width_ratios = [0.82, 3.35, 1.0, 1.0, 1.0, 1.22, 1.82, 1.0, 1.0]
    height_ratios = [0.48, 2.65, 2.65]
    margins = (0.015, 0.995, 0.09, 0.95)

    fig = plt.figure(figsize=(24.0, 8.8), facecolor="white")
    grid = fig.add_gridspec(
        3,
        len(headers),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.0,
        hspace=0.0,
    )
    fig.subplots_adjust(
        left=margins[0],
        right=margins[1],
        bottom=margins[2],
        top=margins[3],
        wspace=0.0,
        hspace=0.0,
    )

    for col, header in enumerate(headers):
        draw_header(fig.add_subplot(grid[0, col]), header, font_size)

    for row_idx, (row_label, record) in enumerate(pairs, start=1):
        draw_row_label(fig.add_subplot(grid[row_idx, 0]), row_label, font_size)
        draw_pair_cell(
            fig.add_subplot(grid[row_idx, 1]),
            record,
            image_paths,
            original_paths,
            font_size,
        )
        for col_idx, method_name in enumerate(method_names, start=2):
            score = method_scores[method_name][(record.query_id, record.compare_id)]
            pred_same = score >= method_thresholds[method_name]
            correct = pred_same == (record.label == 1)
            draw_result_cell(
                fig.add_subplot(grid[row_idx, col_idx]),
                correct,
                font_size,
            )

    add_column_separators(fig, width_ratios, margins)
    add_legend(fig, font_size)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def write_csv(pairs, method_scores, image_paths, original_paths, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    method_names = [name for name, _ in METHOD_SCORE_FILES]
    fieldnames = [
        "case_type",
        "pair_index",
        "query_id",
        "compare_id",
        "label",
        "query_image",
        "compare_image",
        "query_original",
        "compare_original",
        *method_names,
    ]

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_type, record in pairs:
            row = {
                "case_type": case_type,
                "pair_index": record.pair_index,
                "query_id": record.query_id,
                "compare_id": record.compare_id,
                "label": "Genuine" if record.label == 1 else "Impostor",
                "query_image": image_paths.get(record.query_id, ""),
                "compare_image": image_paths.get(record.compare_id, ""),
                "query_original": original_paths.get(record.query_id, ""),
                "compare_original": original_paths.get(record.compare_id, ""),
            }
            for method_name in method_names:
                row[method_name] = f"{method_scores[method_name][(record.query_id, record.compare_id)]:.8f}"
            writer.writerow(row)


def main():
    args = parse_args()
    configure_fonts(args.font_size)

    records = load_scored_pairs(args.pair_list, args.direct_score_list)
    direct_threshold = best_threshold(
        read_labeled_scores(args.pair_list, args.direct_score_list)
    )
    selected_pairs = select_hard_cases(records, direct_threshold)
    image_paths = build_image_paths(args.image_path_list, args.image_dir)
    original_paths = read_path_list(args.original_path_list)

    method_scores = {
        method_name: read_scores_for_pairs(score_file, selected_pairs)
        for method_name, score_file in METHOD_SCORE_FILES
    }
    method_thresholds = compute_method_thresholds(args.pair_list)

    plot_table(
        selected_pairs,
        method_scores,
        method_thresholds,
        image_paths,
        original_paths,
        args.out_png,
        args.dpi,
        args.font_size,
    )
    write_csv(selected_pairs, method_scores, image_paths, original_paths, args.out_csv)

    print("[Hard cases] Selected pairs by Direct score:")
    print(f"[Hard cases] Direct threshold used for selection: {direct_threshold:.8f}")
    for case_type, record in selected_pairs:
        print(
            f"  {case_type}: pair_index={record.pair_index}, "
            f"IDs={record.query_id}-{record.compare_id}, "
            f"Direct={record.direct_score:.8f}"
        )
    print(f"[Hard cases] Saved PNG: {args.out_png}")
    print(f"[Hard cases] Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
