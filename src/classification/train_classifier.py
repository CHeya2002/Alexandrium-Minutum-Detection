"""
Alexandrium minutum Cell Classification
=======================================

Training and evaluation of CNN-based classifiers for distinguishing
Alexandrium minutum cells from other detected cells.

Models:
    - ResNet18
    - ResNet50

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
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from torch.optim import Adam

from torch.utils.data import (
    DataLoader,
    Subset,
    random_split,
)

from torchvision import datasets, models, transforms


# ============================================================
# Configuration
# ============================================================

DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42

NUM_CLASSES = 2


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=DEFAULT_SEED):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Image transformations
# ============================================================

def get_transforms(image_size=DEFAULT_IMAGE_SIZE):
    """
    Create image preprocessing transformations.

    Images are resized to 224 x 224 by default and normalized
    using ImageNet statistics because the classifiers are
    initialized with ImageNet pretrained weights.
    """

    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(
    data_dir,
    image_size=DEFAULT_IMAGE_SIZE,
):
    """
    Load the classification dataset.

    Expected directory structure:

        data_dir/
        ├── A.minutum-cells/
        │   ├── image_001.jpg
        │   └── ...
        │
        └── other-cells/
            ├── image_001.jpg
            └── ...

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing the class folders.

    image_size : int
        Input image size.

    Returns
    -------
    torchvision.datasets.ImageFolder
        Loaded image dataset.
    """

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    transform = get_transforms(image_size)

    dataset = datasets.ImageFolder(
        root=str(data_dir),
        transform=transform,
    )

    print("\nDataset loaded successfully.")
    print(f"Total images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    print(f"Class mapping: {dataset.class_to_idx}")

    return dataset


# ============================================================
# Dataset splitting
# ============================================================

def split_dataset(
    dataset,
    seed=DEFAULT_SEED,
):
    """
    Split the dataset into training, validation, and test sets.

    Split:
        80% training
        10% validation
        10% testing
    """

    dataset_size = len(dataset)

    train_size = int(0.80 * dataset_size)
    validation_size = int(0.10 * dataset_size)

    test_size = (
        dataset_size
        - train_size
        - validation_size
    )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [
            train_size,
            validation_size,
            test_size,
        ],
        generator=generator,
    )

    print("\nDataset split:")
    print(f"Training:   {len(train_dataset)}")
    print(f"Validation: {len(validation_dataset)}")
    print(f"Testing:    {len(test_dataset)}")

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


# ============================================================
# Data loaders
# ============================================================

def create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Create PyTorch DataLoaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


# ============================================================
# Model creation
# ============================================================

def create_model(
    architecture,
    num_classes=NUM_CLASSES,
):
    """
    Create a pretrained ResNet classifier.

    Supported architectures:
        - resnet18
        - resnet50
    """

    architecture = architecture.lower()

    if architecture == "resnet18":

        print("\nLoading pretrained ResNet18...")

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

    elif architecture == "resnet50":

        print("\nLoading pretrained ResNet50...")

        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

    else:

        raise ValueError(
            "Architecture must be 'resnet18' or 'resnet50'."
        )

    # Replace the final classification layer.
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes,
    )

    return model


# ============================================================
# Validation
# ============================================================

def evaluate_loss(
    model,
    data_loader,
    criterion,
    device,
):
    """
    Calculate average loss and accuracy for a dataset.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# Training
# ============================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    patience=DEFAULT_PATIENCE,
):
    """
    Train a ResNet classifier with early stopping.
    """

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model = model.to(device)

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    best_validation_loss = float("inf")

    epochs_without_improvement = 0

    for epoch in range(epochs):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        training_loss = (
            running_loss / total
        )

        training_accuracy = (
            correct / total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_loss, validation_accuracy = evaluate_loss(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {training_loss:.4f} | "
            f"Train Acc: {training_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:

                print(
                    "\nEarly stopping activated."
                )

                break

    # Restore the best validation checkpoint.
    model.load_state_dict(
        best_model_weights
    )

    return model


# ============================================================
# Final evaluation
# ============================================================

def evaluate_classifier(
    model,
    test_loader,
    device,
    class_names,
):
    """
    Evaluate the trained classifier on the test dataset.

    Metrics:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion matrix
        - Classification report
    """

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            true_labels.extend(
                labels.cpu().numpy()
            )

            predicted_labels.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix,
    }


# ============================================================
# Save model
# ============================================================

def save_model(
    model,
    architecture,
    class_names,
    output_dir,
):
    """
    Save the trained model checkpoint.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{architecture}_alexandrium_classifier.pth"
    )

    checkpoint = {
        "architecture": architecture,
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
    }

    torch.save(
        checkpoint,
        output_path,
    )

    print(
        f"\nModel saved to: {output_path}"
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train a ResNet classifier for "
            "Alexandrium minutum cell classification."
        )
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=(
            "Path to the classification dataset."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "resnet18",
            "resnet50",
        ],
        help="Classifier architecture.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            f"Maximum number of training epochs "
            f"(default: {DEFAULT_EPOCHS})."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            f"Training batch size "
            f"(default: {DEFAULT_BATCH_SIZE})."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=(
            f"Learning rate "
            f"(default: {DEFAULT_LEARNING_RATE})."
        ),
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=(
            f"Early-stopping patience "
            f"(default: {DEFAULT_PATIENCE})."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/classification",
        help="Directory used to save the trained model.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    set_seed()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_dataset(
        data_dir=args.data,
    )

    class_names = dataset.classes

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = split_dataset(dataset)

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        architecture=args.model,
        num_classes=len(class_names),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    evaluate_classifier(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    save_model(
        model=model,
        architecture=args.model,
        class_names=class_names,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()