"""
Cell Extraction for Alexandrium minutum Classification
======================================================

This script uses a trained YOLO detector to locate cells in microscopy
images and extract individual cell crops for the subsequent ResNet
classification stage.

Output classes:
    - A.minutum-cells
    - other-cells

Original academic project:
    "Classification of Alexandrium minutum Occurrences and Blooms
    in Maritime Context"

Authors:
    Eya Chtourou
    Syrine Ayedi

Academic year:
    2024-2025
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

DEFAULT_CONFIDENCE = 0.25

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Output directories
# ============================================================

def create_output_directories(output_dir):
    """
    Create the directories used to store extracted cells.

    Parameters
    ----------
    output_dir : str or Path
        Root output directory.

    Returns
    -------
    tuple
        Paths to the Alexandrium minutum and other-cell directories.
    """

    output_dir = Path(output_dir)

    minutum_dir = output_dir / "A.minutum-cells"
    other_dir = output_dir / "other-cells"

    minutum_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    other_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return minutum_dir, other_dir


# ============================================================
# Image discovery
# ============================================================

def get_image_files(image_dir):
    """
    Find supported image files recursively.
    """

    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Input image directory not found: {image_dir}"
        )

    image_files = [
        path
        for path in image_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise RuntimeError(
            f"No supported images found in: {image_dir}"
        )

    return sorted(image_files)


# ============================================================
# Bounding-box handling
# ============================================================

def clip_bounding_box(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height,
):
    """
    Ensure a bounding box remains inside image boundaries.
    """

    x1 = max(0, min(int(x1), image_width))
    y1 = max(0, min(int(y1), image_height))

    x2 = max(0, min(int(x2), image_width))
    y2 = max(0, min(int(y2), image_height))

    return x1, y1, x2, y2


# ============================================================
# Cell extraction
# ============================================================

def extract_cells(
    model_path,
    image_dir,
    output_dir,
    minutum_class_id=1,
    confidence=DEFAULT_CONFIDENCE,
):
    """
    Detect and crop individual cells using a trained YOLO model.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained YOLO checkpoint.

    image_dir : str or Path
        Directory containing source microscopy images.

    output_dir : str or Path
        Directory where extracted cells will be stored.

    minutum_class_id : int
        YOLO class ID corresponding to Alexandrium minutum.

    confidence : float
        Minimum detection confidence.

    Returns
    -------
    dict
        Number of saved A. minutum and other-cell crops.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model checkpoint not found: {model_path}"
        )

    image_files = get_image_files(image_dir)

    minutum_dir, other_dir = create_output_directories(
        output_dir
    )

    print("\nLoading detector...")
    print(f"Model: {model_path}")

    model = YOLO(str(model_path))

    print(f"Images found: {len(image_files)}")
    print(f"Confidence threshold: {confidence}")
    print(f"A. minutum class ID: {minutum_class_id}")

    minutum_count = 0
    other_count = 0

    # --------------------------------------------------------
    # Process each source image
    # --------------------------------------------------------

    for image_index, image_path in enumerate(
        image_files,
        start=1,
    ):

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Warning: unable to read {image_path}"
            )
            continue

        image_height, image_width = image.shape[:2]

        # Run YOLO inference.
        results = model.predict(
            source=str(image_path),
            conf=confidence,
            verbose=False,
        )

        # ----------------------------------------------------
        # Process detections
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            for detection_index, box in enumerate(
                result.boxes,
                start=1,
            ):

                coordinates = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = coordinates

                x1, y1, x2, y2 = clip_bounding_box(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    image_width=image_width,
                    image_height=image_height,
                )

                # Ignore invalid boxes.
                if x2 <= x1 or y2 <= y1:
                    continue

                cell_crop = image[
                    y1:y2,
                    x1:x2,
                ]

                if cell_crop.size == 0:
                    continue

                class_id = int(
                    box.cls[0].item()
                )

                # ------------------------------------------------
                # Select output class
                # ------------------------------------------------

                if class_id == minutum_class_id:

                    destination_dir = minutum_dir
                    minutum_count += 1

                    crop_number = minutum_count
                    class_prefix = "a_minutum"

                else:

                    destination_dir = other_dir
                    other_count += 1

                    crop_number = other_count
                    class_prefix = "other"

                # ------------------------------------------------
                # Save crop
                # ------------------------------------------------

                output_name = (
                    f"{class_prefix}_"
                    f"{image_path.stem}_"
                    f"{image_index:05d}_"
                    f"{detection_index:03d}_"
                    f"{crop_number:06d}.jpg"
                )

                output_path = (
                    destination_dir
                    / output_name
                )

                success = cv2.imwrite(
                    str(output_path),
                    cell_crop,
                )

                if not success:
                    print(
                        f"Warning: could not save {output_path}"
                    )

        print(
            f"[{image_index}/{len(image_files)}] "
            f"Processed {image_path.name}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CELL EXTRACTION COMPLETED")
    print("=" * 60)

    print(
        f"A. minutum cells: {minutum_count}"
    )

    print(
        f"Other cells:      {other_count}"
    )

    print(
        f"Total crops:      "
        f"{minutum_count + other_count}"
    )

    print(
        f"\nSaved to: {Path(output_dir)}"
    )

    return {
        "A.minutum-cells": minutum_count,
        "other-cells": other_count,
    }


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Extract detected cells from microscopy images "
            "using a trained YOLO model."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to the trained YOLO checkpoint (.pt)."
        ),
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help=(
            "Directory containing source images."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/extracted_cells",
        help=(
            "Directory used to save extracted cells."
        ),
    )

    parser.add_argument(
        "--minutum-class-id",
        type=int,
        default=1,
        help=(
            "YOLO class ID corresponding to "
            "Alexandrium minutum."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=(
            f"Detection confidence threshold "
            f"(default: {DEFAULT_CONFIDENCE})."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    extract_cells(
        model_path=args.model,
        image_dir=args.images,
        output_dir=args.output,
        minutum_class_id=args.minutum_class_id,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()