"""
Alexandrium minutum Object Detection
====================================

Training, evaluation, and prediction utilities for the object-detection
models investigated in the project:

    "Classification of Alexandrium minutum Occurrences and Blooms
    in Maritime Context"

Detection models:
    - YOLOv8
    - YOLO11
    - RF-DETR

Authors:
    Eya Chtourou
    Syrine Ayedi

Academic year:
    2024-2025
"""

import argparse
from pathlib import Path


# ============================================================
# Default configuration
# ============================================================

DEFAULT_EPOCHS = 50
DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_CONFIDENCE = 0.25


# ============================================================
# YOLOv8
# ============================================================

def train_yolov8(
    data_yaml,
    epochs=DEFAULT_EPOCHS,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Train a YOLOv8s object-detection model.

    Parameters
    ----------
    data_yaml : str or Path
        Path to the YOLO dataset configuration file.

    epochs : int
        Number of training epochs.

    image_size : int
        Input image size used during training.

    batch_size : int
        Training batch size.

    Returns
    -------
    ultralytics.YOLO
        Trained YOLOv8 model.
    """

    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Training YOLOv8")
    print("=" * 60)

    print(f"Dataset:    {data_yaml}")
    print(f"Epochs:     {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")

    # Base model used in the original experiment
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        name="alexandrium_yolov8",
    )

    return model


# ============================================================
# YOLO11
# ============================================================

def train_yolo11(
    data_yaml,
    epochs=DEFAULT_EPOCHS,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Train a YOLO11n object-detection model.

    Parameters
    ----------
    data_yaml : str or Path
        Path to the YOLO dataset configuration file.

    epochs : int
        Number of training epochs.

    image_size : int
        Input image size used during training.

    batch_size : int
        Training batch size.

    Returns
    -------
    ultralytics.YOLO
        Trained YOLO11 model.
    """

    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("Training YOLO11")
    print("=" * 60)

    print(f"Dataset:    {data_yaml}")
    print(f"Epochs:     {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")

    # Base model used in the original experiment
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        name="alexandrium_yolov11",
    )

    return model


# ============================================================
# YOLO evaluation
# ============================================================

def evaluate_yolo(model):
    """
    Evaluate a trained YOLO model on its validation dataset.

    Parameters
    ----------
    model
        Trained Ultralytics YOLO model.

    Returns
    -------
    metrics
        Ultralytics validation metrics.
    """

    print("\n" + "=" * 60)
    print("Evaluating model")
    print("=" * 60)

    metrics = model.val()

    print("\nValidation metrics:")

    if hasattr(metrics, "results_dict"):

        for metric_name, value in metrics.results_dict.items():
            print(f"{metric_name}: {value}")

    return metrics


# ============================================================
# YOLO prediction
# ============================================================

def predict_yolo(
    model,
    source,
    confidence=DEFAULT_CONFIDENCE,
):
    """
    Run inference using a trained YOLO model.

    Parameters
    ----------
    model
        Trained Ultralytics YOLO model.

    source : str or Path
        Image, image directory, or another Ultralytics-supported source.

    confidence : float
        Minimum prediction confidence.

    Returns
    -------
    results
        Prediction results returned by Ultralytics.
    """

    print("\n" + "=" * 60)
    print("Running predictions")
    print("=" * 60)

    print(f"Source: {source}")
    print(f"Confidence threshold: {confidence}")

    results = model.predict(
        source=str(source),
        conf=confidence,
        save=True,
        project="runs/detect",
        name="alexandrium_predictions",
    )

    return results


# ============================================================
# RF-DETR
# ============================================================

def train_rfdetr(
    dataset_dir,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Train RF-DETR on the Alexandrium dataset.

    Notes
    -----
    RF-DETR uses a different training interface and dataset format from
    Ultralytics YOLO. Therefore, it is handled separately.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to the dataset prepared for RF-DETR.

    epochs : int
        Number of training epochs.

    batch_size : int
        Training batch size.

    Returns
    -------
    model
        Trained RF-DETR model.
    """

    try:
        from rfdetr import RFDETRBase

    except ImportError as exc:

        raise ImportError(
            "RF-DETR is not installed. "
            "Install the dependencies from requirements.txt."
        ) from exc

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"RF-DETR dataset directory not found: {dataset_dir}"
        )

    print("\n" + "=" * 60)
    print("Training RF-DETR")
    print("=" * 60)

    print(f"Dataset:    {dataset_dir}")
    print(f"Epochs:     {epochs}")
    print(f"Batch size: {batch_size}")

    model = RFDETRBase()

    model.train(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        output_dir="runs/rfdetr/alexandrium_rfdetr",
    )

    return model


# ============================================================
# Dataset validation
# ============================================================

def validate_yolo_dataset(data_yaml):
    """
    Verify that the YOLO data.yaml file exists.
    """

    data_yaml = Path(data_yaml)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO dataset configuration file not found: {data_yaml}"
        )

    return data_yaml


def validate_dataset_directory(dataset_dir):
    """
    Verify that a dataset directory exists.
    """

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    return dataset_dir


# ============================================================
# Command-line interface
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train Alexandrium minutum object-detection models "
            "using YOLOv8, YOLO11, or RF-DETR."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "yolov8",
            "yolo11",
            "rfdetr",
        ],
        help="Object-detection model to train.",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=(
            "For YOLO: path to data.yaml. "
            "For RF-DETR: path to the prepared dataset directory."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE}).",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=(
            f"YOLO input image size "
            f"(default: {DEFAULT_IMAGE_SIZE})."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=(
            f"YOLO prediction confidence threshold "
            f"(default: {DEFAULT_CONFIDENCE})."
        ),
    )

    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help=(
            "Optional image or directory on which to run "
            "YOLO prediction after training."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    """
    Main training entry point.
    """

    args = parse_arguments()

    # --------------------------------------------------------
    # YOLOv8
    # --------------------------------------------------------

    if args.model == "yolov8":

        data_yaml = validate_yolo_dataset(args.data)

        model = train_yolov8(
            data_yaml=data_yaml,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )

        evaluate_yolo(model)

        if args.predict:

            predict_yolo(
                model=model,
                source=args.predict,
                confidence=args.confidence,
            )

    # --------------------------------------------------------
    # YOLO11
    # --------------------------------------------------------

    elif args.model == "yolo11":

        data_yaml = validate_yolo_dataset(args.data)

        model = train_yolo11(
            data_yaml=data_yaml,
            epochs=args.epochs,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )

        evaluate_yolo(model)

        if args.predict:

            predict_yolo(
                model=model,
                source=args.predict,
                confidence=args.confidence,
            )

    # --------------------------------------------------------
    # RF-DETR
    # --------------------------------------------------------

    elif args.model == "rfdetr":

        dataset_dir = validate_dataset_directory(args.data)

        train_rfdetr(
            dataset_dir=dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()