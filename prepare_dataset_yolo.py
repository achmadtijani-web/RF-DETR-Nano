"""Audit a COCO detection export and prepare a leakage-reduced YOLO dataset.

The raw input is always read-only. Pillow is the only third-party dependency.
Run ``python prepare_dataset.py --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


SPLITS = ("train", "valid", "test")
YOLO_SPLITS = {"train": "train", "valid": "val", "test": "test"}
PRIORITY = {"train": 0, "valid": 1, "test": 2}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RF_NAME = re.compile(r"^(.*?)\.rf\.[0-9a-f]+\.[^.]+$", re.IGNORECASE)
BBOX_TOLERANCE_PX = 1.0
COLOR_PALETTE = (
    "#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#b388eb", "#ff924c",
    "#52b788", "#e76f51", "#4cc9f0", "#f72585", "#a7c957", "#f4a261",
    "#56cfe1", "#c77dff", "#e9c46a", "#80ed99", "#f28482", "#a2d2ff",
)


class DatasetValidationError(ValueError):
    """Invalid dataset input; no training dataset should be produced."""


@dataclass
class Record:
    split: str
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    byte_sha256: str = ""
    pixel_sha256: str = ""
    label_sha256: str = ""
    rf_original_key: str = ""
    group: str = ""
    exclusion: str = ""
    output_name: str = ""

    @property
    def key(self) -> str:
        return f"{self.split}/{self.file_name}"


class DSU:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first == second:
            return
        if self.rank[first] < self.rank[second]:
            first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value: Any, context: str, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or (positive and value <= 0):
        raise DatasetValidationError(f"{context}: expected {'positive ' if positive else ''}integer")
    return value


def validate_bbox(raw: Any, width: int, height: int) -> tuple[tuple[float, ...], bool]:
    """Reject invalid/large overflows; clip <=1 px rounding overflow only."""
    if not isinstance(raw, list) or len(raw) != 4:
        raise DatasetValidationError("bbox must contain exactly four numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in raw):
        raise DatasetValidationError("bbox values must be finite numbers")
    x, y, w, h = map(float, raw)
    if w <= 0 or h <= 0:
        raise DatasetValidationError("bbox width and height must be positive")
    overflow = max(0.0, -x, -y, x + w - width, y + h - height)
    if overflow > BBOX_TOLERANCE_PX + 1e-9:
        raise DatasetValidationError(f"bbox outside image by {overflow:.6f} px (>1 px tolerance)")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        raise DatasetValidationError("bbox has no positive area inside the image")
    return (x1, y1, x2 - x1, y2 - y1), overflow > 0


def safe_image_path(directory: Path, file_name: Any) -> Path:
    if not isinstance(file_name, str) or not file_name.strip():
        raise DatasetValidationError("file_name must be a nonempty string")
    normalized = file_name.replace("\\", "/")
    if Path(normalized).is_absolute() or ":" in normalized or ".." in Path(normalized).parts:
        raise DatasetValidationError(f"unsafe file_name: {file_name!r}")
    result = (directory / normalized).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise DatasetValidationError(f"file_name escapes split directory: {file_name!r}")
    if result.suffix.lower() not in IMAGE_EXTENSIONS:
        raise DatasetValidationError(f"unsupported image extension: {file_name!r}")
    return result


def inspect_image(record: Record) -> tuple[str, str, str]:
    """Verify the complete file, then hash unrotated, decoded RGB pixels."""
    try:
        byte_digest = sha256_file(record.path)
        with Image.open(record.path) as image:
            image.verify()
        with Image.open(record.path) as image:
            if image.size != (record.width, record.height):
                return "", "", f"{record.key}: COCO dimensions {record.width}x{record.height} differ from pixels {image.width}x{image.height}"
            # COCO coordinates refer to the stored raster. Do not apply EXIF rotation.
            if image.getexif().get(274, 1) not in (None, 1):
                return "", "", f"{record.key}: nontrivial EXIF orientation needs explicit coordinate review"
            image.load()
            pixels = image.convert("RGB")
            digest = hashlib.sha256(f"RGB:{image.width}x{image.height}:".encode("ascii"))
            digest.update(pixels.tobytes())
            return byte_digest, digest.hexdigest(), ""
    except Exception as exc:
        return "", "", f"{record.key}: cannot verify/decode image: {exc}"


def label_signature(record: Record) -> str:
    # Round only at numerical precision well beyond this dataset's COCO coordinates.
    normalized = sorted((cat, *(round(value, 9) for value in box)) for cat, *box in record.boxes)
    return hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_and_audit(source: Path, workers: int, audit: dict) -> tuple[list[Record], list[dict]]:
    errors: list[str] = []
    records: list[Record] = []
    categories: dict[int, dict] | None = None
    clipping: list[dict] = []
    seen_active: set[int] = set()
    orphans: dict[str, list[str]] = {}
    ignored_files: dict[str, list[str]] = {}
    annotation_hashes = {}
    duplicate_annotations = []
    for split in SPLITS:
        split_dir = source / split
        annotation_file = split_dir / "_annotations.coco.json"
        if not annotation_file.is_file():
            errors.append(f"missing annotation file: {annotation_file}")
            continue
        annotation_hashes[split] = sha256_file(annotation_file)
        try:
            data = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or any(not isinstance(data.get(key), list) for key in ("images", "annotations", "categories")):
                raise DatasetValidationError("COCO images, annotations, categories must all be lists")
            split_categories = {}
            for category in data["categories"]:
                category_id = integer(category["id"], "category id")
                if category_id in split_categories:
                    raise DatasetValidationError(f"duplicate category id {category_id}")
                if not isinstance(category.get("name"), str) or not category["name"].strip():
                    raise DatasetValidationError(f"category {category_id}: missing name")
                split_categories[category_id] = {
                    "id": category_id, "name": category["name"],
                    "supercategory": category.get("supercategory", ""),
                }
            if categories is None:
                categories = split_categories
            elif categories != split_categories:
                raise DatasetValidationError("category IDs/names/supercategories differ across splits")
            by_id: dict[int, Record] = {}
            seen_paths: set[Path] = set()
            for image in data["images"]:
                try:
                    image_id = integer(image["id"], "image id")
                    if image_id in by_id:
                        raise DatasetValidationError(f"duplicate image id {image_id}")
                    width = integer(image["width"], "image width", positive=True)
                    height = integer(image["height"], "image height", positive=True)
                    path = safe_image_path(split_dir, image.get("file_name"))
                    if path in seen_paths:
                        raise DatasetValidationError(f"multiple image entries reference {path.name}")
                    seen_paths.add(path)
                    if not path.is_file():
                        raise DatasetValidationError(f"missing referenced image: {image['file_name']}")
                    record = Record(split, image_id, image["file_name"], path, width, height)
                    original = RF_NAME.match(Path(record.file_name).name)
                    record.rf_original_key = original.group(1).casefold() if original else ""
                    by_id[image_id] = record
                    records.append(record)
                except (DatasetValidationError, KeyError, TypeError) as exc:
                    errors.append(f"{split}: invalid image entry {image.get('id', '?') if isinstance(image, dict) else '?'}: {exc}")
            annotation_ids: set[int] = set()
            for annotation in data["annotations"]:
                try:
                    annotation_id = integer(annotation["id"], "annotation id")
                    if annotation_id in annotation_ids:
                        raise DatasetValidationError(f"duplicate annotation id {annotation_id}")
                    annotation_ids.add(annotation_id)
                    image_id = integer(annotation["image_id"], "annotation image_id")
                    category_id = integer(annotation["category_id"], "annotation category_id")
                    if image_id not in by_id:
                        raise DatasetValidationError(f"dangling image_id {image_id}")
                    if category_id not in split_categories:
                        raise DatasetValidationError(f"dangling category_id {category_id}")
                    if annotation.get("iscrowd", 0) not in (0, False):
                        raise DatasetValidationError("iscrowd cannot be represented as an ordinary YOLO object")
                    record = by_id[image_id]
                    bbox, clipped = validate_bbox(annotation.get("bbox"), record.width, record.height)
                    if clipped:
                        clipping.append({"split": split, "annotation_id": annotation_id, "image_id": image_id,
                                         "before": annotation["bbox"], "after": list(bbox)})
                    if "area" in annotation and (isinstance(annotation["area"], bool) or not isinstance(annotation["area"], (float, int)) or not math.isfinite(annotation["area"]) or annotation["area"] < 0):
                        raise DatasetValidationError("annotation area must be a finite nonnegative number")
                    record.boxes.append((category_id, *bbox))
                    seen_active.add(category_id)
                except (DatasetValidationError, KeyError, TypeError) as exc:
                    errors.append(f"{split}: invalid annotation {annotation.get('id', '?') if isinstance(annotation, dict) else '?'}: {exc}")
            actual_files = {p.resolve() for p in split_dir.rglob("*") if p.is_file()}
            image_files = {p for p in actual_files if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("._")}
            orphans[split] = sorted(str(p.relative_to(split_dir)) for p in image_files - seen_paths)
            ignored_files[split] = sorted(str(p.relative_to(split_dir)) for p in actual_files - image_files - {annotation_file.resolve()})
            for record in by_id.values():
                count = len(record.boxes) - len(set(record.boxes))
                if count:
                    duplicate_annotations.append({"image": record.key, "duplicate_box_rows": count})
        except (OSError, json.JSONDecodeError, DatasetValidationError, KeyError, TypeError) as exc:
            errors.append(f"{split}: cannot read COCO structure: {exc}")

    print(f"Validating and decoding {len(records):,} images using {workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (record, result) in enumerate(zip(records, executor.map(inspect_image, records)), 1):
            record.byte_sha256, record.pixel_sha256, error = result
            record.label_sha256 = label_signature(record)
            if error:
                errors.append(error)
            if index % 1000 == 0 or index == len(records):
                print(f"  verified {index:,}/{len(records):,}", flush=True)
    audit["validation"] = {
        "errors": errors, "error_count": len(errors),
        "orphan_image_files": orphans, "orphan_image_count": sum(map(len, orphans.values())),
        "ignored_nonimage_files": ignored_files,
        "clipped_box_count": len(clipping), "clipped_boxes": clipping,
        "clipping_tolerance_pixels": BBOX_TOLERANCE_PX,
        "duplicate_annotation_rows": duplicate_annotations,
        "duplicate_annotation_policy": "Preserved and reported; labels were not silently removed.",
        "image_checks": ["file exists", "Pillow verify", "full decode", "dimensions match COCO", "EXIF orientation is identity"],
        "annotation_source_sha256": annotation_hashes,
        "segmentation_policy": "COCO bbox is used for detection; segmentation polygons are not used as training masks.",
    }
    all_categories = list((categories or {}).values())
    audit["inactive_categories"] = [c for c in all_categories if c["id"] not in seen_active]
    if errors:
        raise DatasetValidationError(f"Dataset failed validation with {len(errors)} error(s); see dataset_audit.json")
    if not records or not seen_active:
        raise DatasetValidationError("Dataset has no usable images or active object classes")
    active = sorted((c for c in all_categories if c["id"] in seen_active), key=lambda c: c["id"])
    return records, active


def duplicate_report(records: list[Record], attribute: str) -> dict:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        value = getattr(record, attribute)
        if value:
            groups[value].append(record)
    duplicates = [items for items in groups.values() if len(items) > 1]
    cross_split = [items for items in duplicates if len({r.split for r in items}) > 1]
    return {
        "group_count": len(duplicates),
        "image_count": sum(map(len, duplicates)),
        "extra_copies": sum(len(items) - 1 for items in duplicates),
        "cross_split_group_count": len(cross_split),
        "cross_split_image_count": sum(map(len, cross_split)),
        "groups": [{"key": getattr(items[0], attribute), "images": [r.key for r in items],
                    "splits": sorted({r.split for r in items})} for items in duplicates],
    }


def resolve_leakage(records: list[Record]) -> dict:
    """Combine all evidence transitively, then keep the highest-priority split."""
    dsu = DSU(len(records))
    for attribute in ("byte_sha256", "pixel_sha256", "rf_original_key"):
        first_seen = {}
        for index, record in enumerate(records):
            value = getattr(record, attribute)
            if value:
                if value in first_seen:
                    dsu.union(index, first_seen[value])
                else:
                    first_seen[value] = index
    groups: dict[int, list[Record]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[dsu.find(index)].append(record)
    linked = sorted(groups.values(), key=lambda items: min(r.key for r in items))
    cross_groups = []
    for number, group in enumerate(linked, 1):
        group_id = f"G{number:06d}"
        winner = max((r.split for r in group), key=PRIORITY.__getitem__)
        splits = {r.split for r in group}
        if len(splits) > 1:
            cross_groups.append({"group": group_id, "retained_split": winner,
                                 "images": [r.key for r in group]})
        for record in group:
            record.group = group_id
            if record.split != winner:
                record.exclusion = f"cross_split_group_retained_in_{YOLO_SPLITS[winner]}"
    # Identical decoded RGB pixels and identical labels are redundant within a split.
    # Conflicting labels remain visible and are not combined or guessed.
    seen = {}
    for record in sorted(records, key=lambda r: (r.split, r.file_name, r.image_id)):
        if record.exclusion:
            continue
        signature = (record.split, record.pixel_sha256, record.label_sha256)
        if signature in seen:
            record.exclusion = f"same_split_identical_pixels_and_labels:{seen[signature].key}"
        else:
            seen[signature] = record
    pixel_groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        pixel_groups[record.pixel_sha256].append(record)
    conflicts = [{"pixel_sha256": key, "images": [r.key for r in items]}
                 for key, items in pixel_groups.items() if len({r.label_sha256 for r in items}) > 1]
    retained = [r for r in records if not r.exclusion]
    remaining_cross_split = []
    for attribute in ("byte_sha256", "pixel_sha256", "rf_original_key"):
        report = duplicate_report(retained, attribute)
        if report["cross_split_group_count"]:
            remaining_cross_split.append(attribute)
    if remaining_cross_split:
        raise RuntimeError(f"Internal error: cross-split evidence remains: {remaining_cross_split}")
    return {
        "policy": "Linked components of byte SHA256, decoded RGB pixel SHA256, and original .rf. basename; retain test > valid > train. Lower-priority members are excluded, never reassigned.",
        "rf_heuristic": "Only names matching PREFIX.rf.HEX.EXT are grouped. PREFIX is case-folded. This is conservative source-family evidence, not proof of identical scenes; filename collisions can over-group unrelated images.",
        "exact_duplicate_policy": "Within the retained split, only identical RGB pixels AND identical box/class labels are deduplicated. Augmented family members and label conflicts are kept and reported.",
        "linked_group_count": len(linked), "cross_split_groups": cross_groups,
        "cross_split_group_count": len(cross_groups),
        "excluded_by_reason": dict(Counter(r.exclusion.split(":", 1)[0] for r in records if r.exclusion)),
        "identical_pixels_with_conflicting_labels": conflicts,
        "remaining_cross_split_byte_pixel_or_rf_groups": 0,
        "residual_risk": "Perceptually similar images, alternate filenames, recompressed/resized copies without shared .rf. prefixes, adjacent video frames, and common farms/capture sessions are not proven independent. No perceptual or spatial/temporal near-duplicate detection is claimed. Verify source-group and field independence before operational accuracy claims.",
    }


def box_distribution(records: list[Record]) -> dict:
    result = Counter({"total": 0, "area_lt_16_squared": 0, "area_lt_32_squared": 0,
                      "area_32_squared_to_96_squared": 0, "area_ge_96_squared": 0,
                      "min_side_lt_4_px": 0, "min_side_lt_8_px": 0, "min_side_lt_16_px": 0})
    areas, widths, heights = [], [], []
    for record in records:
        for _, _, _, width, height in record.boxes:
            area = width * height
            areas.append(area)
            widths.append(width)
            heights.append(height)
            result["total"] += 1
            result["area_lt_16_squared"] += area < 16**2
            if area < 32**2:
                result["area_lt_32_squared"] += 1
            elif area < 96**2:
                result["area_32_squared_to_96_squared"] += 1
            else:
                result["area_ge_96_squared"] += 1
            for threshold in (4, 8, 16):
                result[f"min_side_lt_{threshold}_px"] += min(width, height) < threshold
    def quantiles(values):
        ordered = sorted(values)
        if not ordered:
            return {}
        def percentile(q):
            position = (len(ordered) - 1) * q
            lower, upper = math.floor(position), math.ceil(position)
            return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)
        return {name: percentile(q) for name, q in (("min", 0), ("p01", .01), ("p05", .05), ("p50", .5), ("p95", .95), ("max", 1))}
    return {**dict(result), "area_px2_quantiles": quantiles(areas),
            "width_px_quantiles": quantiles(widths), "height_px_quantiles": quantiles(heights),
            "definition": "Absolute pixels at the stored source resolution; thresholds overlap for the tiny/min-side diagnostics."}


def dataset_statistics(records: list[Record], categories: list[dict]) -> dict:
    result = {}
    for split in (*SPLITS, "all"):
        items = records if split == "all" else [r for r in records if r.split == split]
        annotations = Counter(cat for r in items for cat, *_ in r.boxes)
        images = Counter(cat for r in items for cat in {box[0] for box in r.boxes})
        result[YOLO_SPLITS.get(split, split)] = {
            "images": len(items), "annotations": sum(annotations.values()),
            "empty_label_images": sum(not r.boxes for r in items),
            "dimensions": dict(Counter(f"{r.width}x{r.height}" for r in items)),
            "classes": [{"coco_id": c["id"], "name": c["name"], "annotations": annotations[c["id"]], "images": images[c["id"]]} for c in categories],
            "box_sizes": box_distribution(items),
        }
    return result


def yolo_label_text(record: Record, mapping: dict[int, int]) -> str:
    lines = []
    for category, x, y, width, height in record.boxes:
        values = ((x + width / 2) / record.width, (y + height / 2) / record.height,
                  width / record.width, height / record.height)
        if any(not 0 <= value <= 1 for value in values) or min(values[2:]) <= 0:
            raise RuntimeError(f"Invalid normalized output for {record.key}")
        # Twelve decimal places preserve even sub-pixel labels in large images.
        lines.append(f"{mapping[category]} " + " ".join(f"{value:.12f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def materialize(records: list[Record], categories: list[dict], output: Path) -> dict:
    mapping = {category["id"]: i for i, category in enumerate(categories)}
    methods = Counter()
    for split in YOLO_SPLITS.values():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.exclusion:
            continue
        split = YOLO_SPLITS[record.split]
        # Short names avoid Windows path limits while the manifest preserves source names.
        record.output_name = f"{record.image_id:08d}_{record.byte_sha256[:12]}{record.path.suffix.lower()}"
        image_path = output / "images" / split / record.output_name
        label_path = output / "labels" / split / (Path(record.output_name).stem + ".txt")
        try:
            os.link(record.path, image_path)
            methods["hardlink"] += 1
        except OSError:
            shutil.copy2(record.path, image_path)
            methods["copy"] += 1
        label_path.write_text(yolo_label_text(record, mapping), encoding="utf-8", newline="\n")
    yaml_lines = ["# Generated from validated COCO boxes; see reports/dataset_audit.json.",
                  "path: " + json.dumps(output.as_posix(), ensure_ascii=False),
                  "train: images/train", "val: images/val", "test: images/test", f"nc: {len(categories)}", "names:"]
    yaml_lines.extend(f"  {index}: {json.dumps(category['name'], ensure_ascii=False)}" for index, category in enumerate(categories))
    (output / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return {"methods": dict(methods), "note": "Hardlinks share storage with raw files. Treat image files in both locations as immutable; label files are independent."}


def write_manifest(records: list[Record], output: Path, reports: Path) -> None:
    fields = ["source_split", "image_id", "source_file", "source_path", "width", "height", "annotations",
              "byte_sha256", "pixel_sha256", "label_sha256", "rf_original_key", "linked_group",
              "included", "prepared_split", "exclusion_reason", "output_image", "output_label"]
    with (reports / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            split = YOLO_SPLITS[record.split]
            writer.writerow({"source_split": record.split, "image_id": record.image_id, "source_file": record.file_name,
                             "source_path": str(record.path), "width": record.width, "height": record.height,
                             "annotations": len(record.boxes), "byte_sha256": record.byte_sha256,
                             "pixel_sha256": record.pixel_sha256, "label_sha256": record.label_sha256,
                             "rf_original_key": record.rf_original_key, "linked_group": record.group,
                             "included": int(not record.exclusion), "prepared_split": split if not record.exclusion else "",
                             "exclusion_reason": record.exclusion,
                             "output_image": str(output / "images" / split / record.output_name) if not record.exclusion else "",
                             "output_label": str(output / "labels" / split / (Path(record.output_name).stem + ".txt")) if not record.exclusion else ""})


def get_font(size: int):
    for candidate in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_visuals(records: list[Record], categories: list[dict], reports: Path) -> None:
    prepared = [r for r in records if not r.exclusion]
    mapping = {c["id"]: i for i, c in enumerate(categories)}
    # One representative per active class makes coverage inspectable and deterministic.
    samples = []
    for category in categories:
        candidates = [r for r in prepared if any(box[0] == category["id"] for box in r.boxes)]
        candidates.sort(key=lambda r: (r.split != "test", len(r.boxes), r.key))
        if candidates:
            samples.append((category, candidates[0]))
    columns, cell_w, cell_h = 3, 410, 350
    rows = math.ceil(len(samples) / columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h + 64), "#101923")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 15), "Rice object labels: one prepared example per active class", font=get_font(23), fill="white")
    draw.text((16, 43), "Audit preview only. Bounding boxes are COCO boxes, including small polygon-fragment boxes.", font=get_font(13), fill="#b8c7d9")
    for index, (category, record) in enumerate(samples):
        column, row = index % columns, index // columns
        left, top = column * cell_w, row * cell_h + 64
        with Image.open(record.path) as raw:
            image = raw.convert("RGB")
            image.thumbnail((390, 286))
        x0, y0 = left + (cell_w - image.width) // 2, top + 50
        sheet.paste(image, (x0, y0))
        draw.text((left + 10, top + 4), f"{mapping[category['id']]:02d}  {category['name']}", font=get_font(17), fill="white")
        draw.text((left + 10, top + 28), f"{YOLO_SPLITS[record.split]} | image {record.image_id} | {len(record.boxes)} boxes", font=get_font(13), fill="#b8c7d9")
        sx, sy = image.width / record.width, image.height / record.height
        for cat, x, y, width, height in record.boxes:
            color = COLOR_PALETTE[mapping[cat] % len(COLOR_PALETTE)]
            draw.rectangle((x0 + x*sx, y0 + y*sy, x0 + (x+width)*sx, y0 + (y+height)*sy), outline=color, width=2)
            if cat == category["id"]:
                draw.text((x0 + x*sx + 2, y0 + y*sy + 2), str(mapping[cat]), font=get_font(12), fill=color, stroke_width=1, stroke_fill="#000000")
    sheet.save(reports / "label_samples.jpg", quality=93)
    width, row_h, top = 1300, 39, 94
    chart = Image.new("RGB", (width, top + row_h * len(categories) + 75), "#101923")
    draw = ImageDraw.Draw(chart)
    draw.text((24, 18), "Prepared dataset: bounding boxes by class and split", font=get_font(27), fill="white")
    draw.text((24, 55), "Counts after source-family leakage exclusion and exact pixel+label deduplication", font=get_font(16), fill="#b8c7d9")
    colors = {"train": "#4cc9f0", "valid": "#ffca3a", "test": "#ff595e"}
    counts = {split: Counter(cat for r in prepared if r.split == split for cat, *_ in r.boxes) for split in SPLITS}
    maximum = max((sum(counts[s][c["id"]] for s in SPLITS) for c in categories), default=1) or 1
    for index, category in enumerate(categories):
        y = top + index * row_h
        draw.text((24, y + 3), f"{index:02d}  {category['name']}", font=get_font(17), fill="white")
        x = 340.0
        for split in SPLITS:
            bar_w = 770 * counts[split][category["id"]] / maximum
            if bar_w:
                draw.rectangle((x, y + 3, x + bar_w, y + 26), fill=colors[split])
            x += bar_w
        draw.text((1130, y + 3), str(sum(counts[s][category["id"]] for s in SPLITS)), font=get_font(17), fill="white")
    y = top + row_h * len(categories) + 15
    for index, split in enumerate(SPLITS):
        x = 340 + index * 180
        draw.rectangle((x, y, x+20, y+20), fill=colors[split])
        draw.text((x+29, y), YOLO_SPLITS[split], font=get_font(17), fill="white")
    chart.save(reports / "class_distribution.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Read-only directory containing train/valid/test COCO folders")
    parser.add_argument("--output", required=True, type=Path, help="New or empty destination for images, labels, and data.yaml")
    parser.add_argument("--reports", required=True, type=Path, help="Destination for audit JSON, mapping, manifest, and previews")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Image verification threads (default <=8)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, output, reports = args.source.resolve(), args.output.resolve(), args.reports.resolve()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise SystemExit("Source and output must be separate directories; output must not contain raw input")
    if reports == source or reports.is_relative_to(source):
        raise SystemExit("Reports must not be written inside the raw source directory")
    if reports == output or reports.is_relative_to(output) or output.is_relative_to(reports):
        raise SystemExit("Output and reports must be separate sibling directories")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output must be empty to prevent stale labels; choose a new destination: {output}")
    report_names = ("dataset_audit.json", "class_mapping.json", "manifest.csv", "label_samples.jpg", "class_distribution.png")
    conflicts = [name for name in report_names if (reports / name).exists()]
    if conflicts:
        raise SystemExit(f"Refusing to overwrite existing audit artifacts in {reports}: {', '.join(conflicts)}")
    reports.mkdir(parents=True, exist_ok=True)
    audit = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
             "source": str(source), "output": str(output), "status": "auditing",
             "class_policy": "Preserve each active category name and ID relation. Similar names were not merged without domain evidence.",
             "scope": "Object detection from still images; dataset labels do not establish field-level agronomic diagnosis or drone suitability."}
    try:
        records, categories = load_and_audit(source, args.workers, audit)
        audit["original"] = dataset_statistics(records, categories)
        audit["duplicates"] = {
            "byte_exact": duplicate_report(records, "byte_sha256"),
            "decoded_pixel_exact": duplicate_report(records, "pixel_sha256"),
            "rf_original_basename": duplicate_report(records, "rf_original_key"),
        }
        audit["leakage_control"] = resolve_leakage(records)
        retained = [r for r in records if not r.exclusion]
        audit["prepared"] = dataset_statistics(retained, categories)
        if any(audit["prepared"][s]["images"] == 0 for s in ("train", "val", "test")):
            raise DatasetValidationError("Leakage grouping left an empty split; inspect source grouping before training")
        mapping = {"schema_version": 1, "active_class_count": len(categories),
                   "classes": [{"yolo_id": i, "coco_id": c["id"], "name": c["name"], "supercategory": c["supercategory"]} for i, c in enumerate(categories)],
                   "excluded_inactive_categories": audit["inactive_categories"],
                   "policy": audit["class_policy"]}
        audit["materialization"] = materialize(records, categories, output)
        write_manifest(records, output, reports)
        write_json(reports / "class_mapping.json", mapping)
        write_json(output / "class_mapping.json", mapping)
        make_visuals(records, categories, reports)
        audit["status"] = "complete"
        audit["artifacts"] = {name: str(reports / name) for name in ("manifest.csv", "class_mapping.json", "label_samples.jpg", "class_distribution.png")}
        write_json(reports / "dataset_audit.json", audit)
        print(f"Prepared {len(retained):,}/{len(records):,} images, {len(categories)} classes.", flush=True)
        for split in ("train", "val", "test"):
            stats = audit["prepared"][split]
            print(f"  {split}: {stats['images']:,} images; {stats['annotations']:,} boxes", flush=True)
        print(f"Data YAML: {output / 'data.yaml'}\nAudit: {reports / 'dataset_audit.json'}", flush=True)
        return 0
    except Exception as exc:
        audit["status"] = "failed"
        audit["failure"] = str(exc)
        write_json(reports / "dataset_audit.json", audit)
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
