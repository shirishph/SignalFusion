#!/usr/bin/env python3

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

RANDOM_SEED = 42

# Number of negative pairs to generate.
NEGATIVE_PAIR_COUNT = 100


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum for a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_image_files(raw_dir: Path) -> list[Path]:
    """Return supported image files from raw directory."""
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ensure_directories(base_dir: Path) -> None:
    """Create required project directories."""

    directories = [
        base_dir / "data" / "originals",
        base_dir / "data" / "variants",
        base_dir / "data" / "metadata",
        base_dir / "data" / "ground_truth",
        base_dir / "evaluation",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to CSV."""

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Original images
# ---------------------------------------------------------------------------

def process_originals(
    raw_files: list[Path],
    project_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Copy originals and create image metadata.

    Returns:
        image_records
        transformation_records
    """

    originals_dir = project_dir / "data" / "originals"

    image_records = []
    transformation_records = []

    for index, source_path in enumerate(raw_files, start=1):

        image_id = f"img_{index:03d}"

        destination = originals_dir / f"{image_id}{source_path.suffix.lower()}"

        shutil.copy2(source_path, destination)

        with Image.open(source_path) as image:
            width, height = image.size
            image_format = image.format

        image_records.append(
            {
                "image_id": image_id,
                "filename": destination.name,
                "source_filename": source_path.name,
                "type": "original",
                "parent_id": "",
                "transformation": "",
                "parameters": "",
                "width": width,
                "height": height,
                "format": image_format,
                "sha256": calculate_sha256(destination),
                "category": "TODO",
                "human_review": "TODO",
            }
        )

    return image_records, transformation_records


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def create_resize_variant(image: Image.Image) -> Image.Image:
    """Resize image to 50%."""
    width, height = image.size

    return image.resize(
        (
            max(1, width // 2),
            max(1, height // 2),
        )
    )


def create_crop_variant(image: Image.Image) -> Image.Image:
    """Crop approximately 15% from each side."""

    width, height = image.size

    crop_x = int(width * 0.15)
    crop_y = int(height * 0.15)

    left = crop_x
    top = crop_y
    right = width - crop_x
    bottom = height - crop_y

    return image.crop((left, top, right, bottom))


def create_jpeg_variant(image: Image.Image) -> Image.Image:
    """
    JPEG compression is handled when saving.
    The image itself does not need modification.
    """
    return image.copy()


def create_brightness_variant(image: Image.Image) -> Image.Image:
    """Increase brightness by approximately 20%."""
    enhancer = ImageEnhance.Brightness(image)

    return enhancer.enhance(1.2)


def create_rotation_variant(image: Image.Image) -> Image.Image:
    """Rotate image by 5 degrees."""
    return image.rotate(
        5,
        expand=True,
        fillcolor="white",
    )


def save_variant(
    image: Image.Image,
    output_path: Path,
    transformation: str,
    jpeg_quality: int | None = None,
) -> None:
    """Save a generated image variant."""

    if output_path.suffix.lower() in {".jpg", ".jpeg"}:

        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        save_kwargs = {}

        if jpeg_quality is not None:
            save_kwargs["quality"] = jpeg_quality

        image.save(
            output_path,
            format="JPEG",
            **save_kwargs,
        )

    else:
        image.save(output_path)


def create_variants(
    image_records: list[dict],
    project_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Create controlled variants.

    Returns:
        variant image records
        positive ground-truth relationships
    """

    variants_dir = project_dir / "data" / "variants"

    variant_records = []
    positive_pairs = []

    for record in image_records:

        parent_id = record["image_id"]

        original_path = (
            project_dir
            / "data"
            / "originals"
            / record["filename"]
        )

        with Image.open(original_path) as source_image:

            transformations = [
                (
                    "resize_50",
                    "50%",
                    create_resize_variant(source_image),
                ),
                (
                    "crop_15",
                    "15% border crop",
                    create_crop_variant(source_image),
                ),
                (
                    "brightness_20",
                    "+20% brightness",
                    create_brightness_variant(source_image),
                ),
                (
                    "rotate_5",
                    "+5 degrees",
                    create_rotation_variant(source_image),
                ),
                (
                    "jpeg_quality_50",
                    "JPEG quality=50",
                    create_jpeg_variant(source_image),
                ),
            ]

            for transformation, parameters, variant_image in transformations:

                variant_id = f"{parent_id}_{transformation}"

                # Save all variants as JPEG.
                output_path = variants_dir / f"{variant_id}.jpg"

                jpeg_quality = (
                    50
                    if transformation == "jpeg_quality_50"
                    else 95
                )

                save_variant(
                    variant_image,
                    output_path,
                    transformation,
                    jpeg_quality,
                )

                with Image.open(output_path) as saved_image:
                    width, height = saved_image.size
                    image_format = saved_image.format

                variant_records.append(
                    {
                        "image_id": variant_id,
                        "filename": output_path.name,
                        "source_filename": "",
                        "type": "variant",
                        "parent_id": parent_id,
                        "transformation": transformation,
                        "parameters": parameters,
                        "width": width,
                        "height": height,
                        "format": image_format,
                        "sha256": calculate_sha256(output_path),
                        "category": "",
                        "human_review": "",
                    }
                )

                positive_pairs.append(
                    {
                        "query_id": parent_id,
                        "candidate_id": variant_id,
                        "label": 1,
                        "relationship": transformation,
                        "review_status": "AUTO",
                    }
                )

    return variant_records, positive_pairs


# ---------------------------------------------------------------------------
# Negative pairs
# ---------------------------------------------------------------------------

def create_negative_pairs(
    image_records: list[dict],
    variant_records: list[dict],
    positive_pairs: list[dict],
) -> list[dict]:
    """
    Create unrelated image pairs.

    These are proposed negatives and require human review.
    """

    random.seed(RANDOM_SEED)

    all_images = (
        [record["image_id"] for record in image_records]
        + [record["image_id"] for record in variant_records]
    )

    positive_set = {
        (pair["query_id"], pair["candidate_id"])
        for pair in positive_pairs
    }

    candidates = []

    for query_id in all_images:

        for candidate_id in all_images:

            if query_id == candidate_id:
                continue

            if (query_id, candidate_id) in positive_set:
                continue

            candidates.append(
                (query_id, candidate_id)
            )

    random.shuffle(candidates)

    selected = candidates[:NEGATIVE_PAIR_COUNT]

    negative_pairs = []

    for query_id, candidate_id in selected:

        negative_pairs.append(
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "label": 0,
                "relationship": "TODO_VERIFY_UNRELATED",
                "review_status": "TODO",
            }
        )

    return negative_pairs


# ---------------------------------------------------------------------------
# Evaluation definition
# ---------------------------------------------------------------------------

def create_metrics_file(project_dir: Path) -> None:

    path = project_dir / "evaluation" / "metrics.md"

    content = """# Evaluation Metrics

These metrics will be used in later experiments.

## Primary metric

### Recall@K

Of all known true matches, how many appear in the top K results?

Measure:

- Recall@1
- Recall@5
- Recall@10

## Secondary metrics

### Precision@K

Of the top K returned candidates, how many are true matches?

Measure:

- Precision@5
- Precision@10

### Mean Reciprocal Rank (MRR)

Measures how highly the first correct match appears.

### False Positive Rate

Measures incorrect matches against known negative relationships.

## Experimental comparison

Every matching experiment should record:

- Method
- Dataset
- Transformation
- Recall@1
- Recall@5
- Recall@10
- Precision@5
- Precision@10
- MRR
- False positives

The main hypothesis is concerned with whether combining multiple
independent signals improves recall compared with a single matching
technique.
"""

    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# TODO file
# ---------------------------------------------------------------------------

def create_todo_file(
    project_dir: Path,
    image_count: int,
    variant_count: int,
    negative_count: int,
) -> None:

    path = project_dir / "TODO.md"

    content = f"""# Day 1 Dataset TODO

Dataset generated automatically.

## Dataset

- [ ] Review {image_count} original images
- [ ] Confirm originals are legally usable for the POC
- [ ] Confirm image diversity is adequate
- [ ] Identify images containing faces
- [ ] Identify images containing text
- [ ] Identify visually similar images

## Ground truth

- [ ] Review {negative_count} proposed negative pairs
- [ ] Confirm proposed negatives are genuinely unrelated
- [ ] Identify useful hard-negative pairs
- [ ] Add additional manually selected negative pairs if required

## Transformations

- [ ] Visually inspect representative variants
- [ ] Confirm transformations are realistic enough for the experiment

## Evaluation

- [ ] Review evaluation/metrics.md
- [ ] Confirm Recall@K will be the primary measure

## Day 2

- [ ] Build first single-signal baseline matcher
"""
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def create_readme(project_dir: Path) -> None:

    path = project_dir / "README.md"

    content = """# Image Matching POC

One-week POC investigating:

> Can a free/open-source multi-signal image matching pipeline
> substantially increase recall compared with a single matching technique?

## Dataset generation

Place raw images in:

    raw_files/

Then run:

    python3 src/dataset_builder.py raw_files/

The script creates:

    data/originals/
    data/variants/
    data/metadata/
    data/ground_truth/
    evaluation/

No image matching is performed by the dataset builder.

Ground truth generated from controlled transformations is automatic.

Human judgement is represented using TODO fields.
"""

    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Build Day-1 dataset for image matching POC."
    )

    parser.add_argument(
        "raw_directory",
        type=Path,
        help="Directory containing raw image files.",
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Root directory of the POC project.",
    )

    args = parser.parse_args()

    raw_dir = args.raw_directory.resolve()
    project_dir = args.project_dir.resolve()

    if not raw_dir.exists():
        raise SystemExit(
            f"Raw directory does not exist: {raw_dir}"
        )

    raw_files = get_image_files(raw_dir)

    if not raw_files:
        raise SystemExit(
            "No supported image files found."
        )

    print()
    print("IMAGE MATCHING POC - DAY 1")
    print("=" * 40)
    print(f"Raw directory: {raw_dir}")
    print(f"Images found:  {len(raw_files)}")
    print()

    if len(raw_files) < 20:
        print(
            "WARNING: Fewer than 20 images found."
        )

    if len(raw_files) > 30:
        print(
            "WARNING: More than 30 images found."
        )

    ensure_directories(project_dir)

    # Originals
    print("Processing originals...")

    image_records, _ = process_originals(
        raw_files,
        project_dir,
    )

    # Variants
    print("Generating controlled variants...")

    variant_records, positive_pairs = create_variants(
        image_records,
        project_dir,
    )

    # Negatives
    print("Generating proposed negative pairs...")

    negative_pairs = create_negative_pairs(
        image_records,
        variant_records,
        positive_pairs,
    )

    # Metadata
    all_records = image_records + variant_records

    save_csv(
        project_dir
        / "data"
        / "metadata"
        / "images.csv",
        [
            "image_id",
            "filename",
            "source_filename",
            "type",
            "parent_id",
            "transformation",
            "parameters",
            "width",
            "height",
            "format",
            "sha256",
            "category",
            "human_review",
        ],
        all_records,
    )

    # Transformation metadata
    transformation_rows = [
        {
            "image_id": record["image_id"],
            "parent_id": record["parent_id"],
            "transformation": record["transformation"],
            "parameters": record["parameters"],
        }
        for record in variant_records
    ]

    save_csv(
        project_dir
        / "data"
        / "metadata"
        / "transformations.csv",
        [
            "image_id",
            "parent_id",
            "transformation",
            "parameters",
        ],
        transformation_rows,
    )

    # Ground truth
    save_csv(
        project_dir
        / "data"
        / "ground_truth"
        / "positive_pairs.csv",
        [
            "query_id",
            "candidate_id",
            "label",
            "relationship",
            "review_status",
        ],
        positive_pairs,
    )

    save_csv(
        project_dir
        / "data"
        / "ground_truth"
        / "negative_pairs.csv",
        [
            "query_id",
            "candidate_id",
            "label",
            "relationship",
            "review_status",
        ],
        negative_pairs,
    )

    # Documentation
    create_metrics_file(project_dir)

    create_todo_file(
        project_dir,
        len(image_records),
        len(variant_records),
        len(negative_pairs),
    )

    create_readme(project_dir)

    print()
    print("=" * 40)
    print("DATASET BUILD COMPLETE")
    print("=" * 40)
    print(f"Original images:  {len(image_records)}")
    print(f"Variants:         {len(variant_records)}")
    print(f"Positive pairs:   {len(positive_pairs)}")
    print(f"Negative pairs:   {len(negative_pairs)}")
    print()
    print("Review:")
    print("  TODO.md")
    print("  data/ground_truth/negative_pairs.csv")
    print()
    print("Day 1 dataset is ready for human review.")


if __name__ == "__main__":
    main()
