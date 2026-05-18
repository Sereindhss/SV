#!/usr/bin/env python3
"""Create an IJB-B 1:N retrieval qualitative figure.

The figure is a compact table:
probe column + Direct + SecureVector-512 + Ours
with a fixed number of retrieved thumbnails per method.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


METHODS = [
    ("Direct", Path("results/baseline/b/score.list")),
    ("SecureVector-512", Path("results/securevector/b/score.list")),
    ("Ours", Path("results/sv_dj_cluster/score.list")),
]

PAIR_LIST = Path("data/ijb/ijbb.pair.list")
FACE_TID_MID = Path("data/ijb/meta/ijbb_face_tid_mid.txt")
IMAGE_DIR = Path("image_data/ijb/IJBB/loose_crop")
DEFAULT_PROBE_OVERRIDES = {0: 38, 3: 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IJB-B 1:N retrieval visualization")
    parser.add_argument("--probe-count", type=int, default=5)
    parser.add_argument("--display-k", type=int, default=3)
    parser.add_argument(
        "--probe-ids",
        default="",
        help="Optional comma-separated probe ids. Default: first N unique identities in the pair list.",
    )
    parser.add_argument(
        "--output",
        default="results/visualizations/ijbb_1vn_retrieval.png",
    )
    parser.add_argument("--thumb-width", type=int, default=96)
    parser.add_argument("--thumb-height", type=int, default=120)
    parser.add_argument("--gap", type=int, default=10)
    parser.add_argument("--margin", type=int, default=18)
    parser.add_argument("--font-size", type=int, default=16, help="Font size. Default is Chinese Sanhao.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
    ]:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def read_template_face_map(face_tid_mid: Path) -> Dict[int, List[int]]:
    template_to_faces: Dict[int, List[int]] = {}
    with face_tid_mid.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            face_name = parts[0]
            face_id = int(Path(face_name).stem) - 1
            template_id = int(parts[1])
            template_to_faces.setdefault(template_id, []).append(face_id)

    template_ids = sorted(template_to_faces)
    template_index_to_faces: Dict[int, List[int]] = {}
    for template_index, template_id in enumerate(template_ids):
        template_index_to_faces[template_index] = template_to_faces[template_id]
    return template_index_to_faces


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, item: int) -> int:
        if item not in self.parent:
            self.parent[item] = item
            return item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def read_template_identity_map(pair_list: Path) -> Dict[int, int]:
    uf = UnionFind()
    with pair_list.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            left = int(parts[0])
            right = int(parts[1])
            label = int(parts[2])
            uf.find(left)
            uf.find(right)
            if label == 1:
                uf.union(left, right)
    return {template_id: uf.find(template_id) for template_id in uf.parent}


def read_unique_probe_ids(
    pair_list: Path, template_to_identity: Dict[int, int], limit: int
) -> List[int]:
    probe_ids: List[int] = []
    seen_identities = set()
    with pair_list.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            probe_id = int(parts[0])
            identity_id = template_to_identity.get(probe_id, probe_id)
            if identity_id in seen_identities:
                continue
            seen_identities.add(identity_id)
            probe_ids.append(probe_id)
            if len(probe_ids) >= limit:
                break
    return probe_ids


def replace_probe_rows(
    pair_list: Path, probe_ids: Sequence[int], row_to_probe: Dict[int, int]
) -> List[int]:
    updated = list(probe_ids)
    rows_to_replace = [row for row in row_to_probe if row < len(updated)]
    if not rows_to_replace:
        return updated

    selected = set(updated)
    for row in rows_to_replace:
        replacement = row_to_probe[row]
        if replacement in selected and replacement != updated[row]:
            raise RuntimeError(f"Replacement probe {replacement} already selected.")
        selected.discard(updated[row])
        updated[row] = replacement
        selected.add(replacement)
    return updated


def face_image_path(face_id: int) -> Path:
    return IMAGE_DIR / f"{face_id + 1}.jpg"


SHARPNESS_CACHE: Dict[int, float] = {}
HASH_CACHE: Dict[int, Tuple[int, ...]] = {}


def image_sharpness(face_id: int) -> float:
    if face_id in SHARPNESS_CACHE:
        return SHARPNESS_CACHE[face_id]
    path = face_image_path(face_id)
    if not path.exists():
        SHARPNESS_CACHE[face_id] = -1.0
        return SHARPNESS_CACHE[face_id]
    try:
        img = Image.open(path).convert("L").resize((64, 64), Image.Resampling.LANCZOS)
        px = list(img.getdata())
        width, height = img.size
        total = 0.0
        count = 0
        for y in range(height - 1):
            row = y * width
            next_row = (y + 1) * width
            for x in range(width - 1):
                center = px[row + x]
                total += abs(center - px[row + x + 1]) + abs(center - px[next_row + x])
                count += 2
        SHARPNESS_CACHE[face_id] = total / max(1, count)
    except Exception:
        SHARPNESS_CACHE[face_id] = -1.0
    return SHARPNESS_CACHE[face_id]


def image_hash(face_id: int) -> Tuple[int, ...]:
    if face_id in HASH_CACHE:
        return HASH_CACHE[face_id]
    path = face_image_path(face_id)
    if not path.exists():
        HASH_CACHE[face_id] = tuple()
        return HASH_CACHE[face_id]
    try:
        img = Image.open(path).convert("L").resize((16, 16), Image.Resampling.LANCZOS)
        px = list(img.getdata())
        avg = sum(px) / len(px)
        HASH_CACHE[face_id] = tuple(1 if p >= avg else 0 for p in px)
    except Exception:
        HASH_CACHE[face_id] = tuple()
    return HASH_CACHE[face_id]


def hash_distance(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    if not left or not right:
        return 0
    return sum(a != b for a, b in zip(left, right))


def choose_face(
    face_ids: Sequence[int],
    used_hashes: Sequence[Tuple[int, ...]] = (),
) -> int | None:
    if not face_ids:
        return None

    def score(face_id: int) -> Tuple[float, float]:
        sharpness = image_sharpness(face_id)
        face_hash = image_hash(face_id)
        if used_hashes:
            diversity = min(hash_distance(face_hash, used_hash) for used_hash in used_hashes)
        else:
            diversity = 256.0
        return sharpness, diversity

    return max(face_ids, key=score)


def make_placeholder(size: Tuple[int, int], text: str = "N/A") -> Image.Image:
    canvas = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(180, 180, 180), width=2)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - (bbox[2] - bbox[0])) // 2
    y = (size[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill=(90, 90, 90), font=font)
    return canvas


def load_thumbnail(path: Path, size: Tuple[int, int]) -> Image.Image:
    if not path.exists():
        return make_placeholder(size)
    try:
        img = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(img, size, method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        offset = ((size[0] - thumb.size[0]) // 2, (size[1] - thumb.size[1]) // 2)
        canvas.paste(thumb, offset)
        return canvas
    except Exception:
        return make_placeholder(size, "ERR")


def load_topk_for_probes(
    pair_list: Path,
    score_list: Path,
    probe_ids: Sequence[int],
    template_to_identity: Dict[int, int],
    top_k: int,
) -> Dict[int, List[Tuple[float, int, int]]]:
    selected = set(probe_ids)
    best_by_identity: Dict[int, Dict[int, Tuple[float, int, int]]] = {
        pid: {} for pid in probe_ids
    }

    with pair_list.open("r", encoding="utf-8") as fp, score_list.open(
        "r", encoding="utf-8"
    ) as fs:
        for pair_line, score_line in zip(fp, fs):
            pair_parts = pair_line.strip().split()
            if len(pair_parts) < 3:
                continue
            probe_id = int(pair_parts[0])
            if probe_id not in selected:
                continue

            gallery_id = int(pair_parts[1])
            label = int(pair_parts[2])
            try:
                score = float(score_line.strip().split()[-1])
            except (IndexError, ValueError):
                continue

            if score < 0:
                continue

            identity_id = template_to_identity.get(gallery_id, gallery_id)
            item = (score, gallery_id, label)
            current = best_by_identity[probe_id].get(identity_id)
            if current is None or score > current[0]:
                best_by_identity[probe_id][identity_id] = item

    ranked: Dict[int, List[Tuple[float, int, int]]] = {}
    for probe_id, identity_results in best_by_identity.items():
        ranked[probe_id] = sorted(
            identity_results.values(), key=lambda x: x[0], reverse=True
        )[:top_k]
    return ranked


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_dashed_vertical_line(
    draw: ImageDraw.ImageDraw,
    x: int,
    y0: int,
    y1: int,
    fill: Tuple[int, int, int] = (150, 150, 150),
    dash: int = 8,
    gap: int = 6,
    width: int = 4,
) -> None:
    y = y0
    while y < y1:
        draw.line((x, y, x, min(y + dash, y1)), fill=fill, width=width)
        y += dash + gap


def build_figure(
    probe_ids: Sequence[int],
    template_index_to_faces: Dict[int, List[int]],
    method_results: Dict[str, Dict[int, List[Tuple[float, int, int]]]],
    display_k: int,
    output: Path,
    thumb_size: Tuple[int, int],
    gap: int,
    margin: int,
    font_size: int,
) -> None:
    font = load_font(font_size)

    probe_w, probe_h = thumb_size
    method_cell_w = display_k * probe_w
    header_h = int(font_size * 2.4)

    total_w = margin * 2 + probe_w + gap + len(METHODS) * (method_cell_w + gap)
    total_h = margin * 2 + header_h + len(probe_ids) * probe_h + max(0, len(probe_ids) - 1) * gap

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # Headers
    x = margin
    header_y = margin // 2
    draw_centered_text(
        draw, (x, header_y, x + probe_w, header_y + header_h), "Probe", font
    )
    x += probe_w + gap
    for method_name, _ in METHODS:
        draw_centered_text(
            draw, (x, header_y, x + method_cell_w, header_y + header_h), method_name, font
        )
        x += method_cell_w + gap

    separator_y0 = margin
    separator_y1 = total_h - margin
    separator_x = margin + probe_w + gap // 2
    draw_dashed_vertical_line(draw, separator_x, separator_y0, separator_y1)
    x = margin + probe_w + gap
    for _method_name, _ in METHODS[:-1]:
        separator_x = x + method_cell_w + gap // 2
        draw_dashed_vertical_line(draw, separator_x, separator_y0, separator_y1)
        x += method_cell_w + gap

    # Rows
    y = margin + header_h
    for row_idx, probe_id in enumerate(probe_ids):
        probe_box = (margin, y, margin + probe_w, y + probe_h)
        probe_face_id = choose_face(template_index_to_faces.get(probe_id, []))
        probe_img = (
            load_thumbnail(face_image_path(probe_face_id), thumb_size)
            if probe_face_id is not None
            else make_placeholder(thumb_size)
        )
        canvas.paste(probe_img, (probe_box[0], probe_box[1]))
        draw.rectangle(probe_box, outline=(60, 60, 60), width=2)

        x = margin + probe_w + gap
        for method_name, _ in METHODS:
            results = method_results[method_name].get(probe_id, [])
            used_hashes: List[Tuple[int, ...]] = []
            for rank in range(display_k):
                cell_x = x + rank * probe_w
                if rank < len(results):
                    score, gallery_id, label = results[rank]
                    gallery_face_id = choose_face(
                        template_index_to_faces.get(gallery_id, []),
                        used_hashes,
                    )
                    if gallery_face_id is not None:
                        used_hashes.append(image_hash(gallery_face_id))
                    gallery_img = (
                        load_thumbnail(face_image_path(gallery_face_id), thumb_size)
                        if gallery_face_id is not None
                        else make_placeholder(thumb_size)
                    )
                else:
                    gallery_img = make_placeholder(thumb_size)

                canvas.paste(gallery_img, (cell_x, y))
            x += method_cell_w + gap

        y += probe_h + gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300))
    print(f"Saved to {output}")


def main() -> None:
    args = parse_args()
    template_to_identity = read_template_identity_map(PAIR_LIST)

    if args.probe_ids.strip():
        probe_ids = [int(x) for x in args.probe_ids.split(",") if x.strip()]
    else:
        probe_ids = read_unique_probe_ids(PAIR_LIST, template_to_identity, args.probe_count)
        probe_ids = replace_probe_rows(PAIR_LIST, probe_ids, DEFAULT_PROBE_OVERRIDES)

    if len(probe_ids) < args.probe_count:
        raise RuntimeError(f"Only found {len(probe_ids)} probes in {PAIR_LIST}")

    template_index_to_faces = read_template_face_map(FACE_TID_MID)
    print("Selected probes:", probe_ids)
    print("Selected identities:", [template_to_identity.get(pid, pid) for pid in probe_ids])
    print("Selected face ids:", [choose_face(template_index_to_faces.get(pid, [])) for pid in probe_ids])
    method_results: Dict[str, Dict[int, List[Tuple[float, int, int]]]] = {}
    for method_name, score_list in METHODS:
        if not score_list.exists():
            raise FileNotFoundError(score_list)
        method_results[method_name] = load_topk_for_probes(
            PAIR_LIST, score_list, probe_ids, template_to_identity, args.display_k
        )
        for probe_id in probe_ids:
            print(
                f"{method_name:>16} | probe {probe_id}: "
                f"{len(method_results[method_name][probe_id])} results"
            )

    build_figure(
        probe_ids=probe_ids,
        template_index_to_faces=template_index_to_faces,
        method_results=method_results,
        display_k=args.display_k,
        output=Path(args.output),
        thumb_size=(args.thumb_width, args.thumb_height),
        gap=args.gap,
        margin=args.margin,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
