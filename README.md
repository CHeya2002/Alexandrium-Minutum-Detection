\# Classification of \*Alexandrium minutum\* Occurrences and Blooms in Maritime Context



Deep-learning-based pipeline for the detection and classification of \*Alexandrium minutum\* in maritime microscopy and cytometry images.



\## Project Overview



Harmful algal blooms can represent significant risks to marine ecosystems, fisheries, and human health. \*Alexandrium minutum\* is a toxic microalgal species associated with harmful algal bloom events, making its monitoring and early detection important.



This project investigates the use of deep learning to support the automatic detection and classification of \*Alexandrium minutum\*. The proposed approach combines object detection and image classification techniques to process microscopy and cytometry images.



The project was developed as a first-year Master's end-of-year project in Data Science at the Higher Institute of Computer Science and Multimedia of Sfax (ISIMS), University of Sfax.



\## Objectives



The main objectives of the project were to:



\- Develop an AI-based pipeline for the automatic detection of \*Alexandrium minutum\*.

\- Process and annotate microscopy and cytometry image data.

\- Compare multiple deep-learning object detection architectures.

\- Classify detected cells using convolutional neural networks.

\- Evaluate the proposed pipeline on two different image datasets.



\## Proposed Pipeline



The developed workflow follows the general sequence:



\*\*Data Collection → Preprocessing \& Annotation → Object Detection → Cell Classification → Evaluation\*\*



The detection stage identifies and localizes cells in the input images, while the classification stage distinguishes \*Alexandrium minutum\* cells from other detected cells.



\## Models



\### Object Detection



Three object detection architectures were investigated:



\- YOLOv8

\- YOLO11

\- RF-DETR



These models were trained and evaluated for the localization and detection of target cells.



\### Image Classification



Following object detection, detected cells were cropped and processed for binary image classification using:



\- ResNet18

\- ResNet50



The classification stage distinguishes between:



\- \*A. minutum\* cells

\- Other cells



\## Data



Two image datasets were considered during the project:



1\. A dataset constructed using images collected from web-based sources.

2\. A dataset containing cytometer-derived images.



The datasets underwent preprocessing, annotation, augmentation, and organization before model training and evaluation.



The datasets themselves are not redistributed through this repository.



Additional information is available in \[`data/README.md`](data/README.md).



\## Technologies



\- Python

\- PyTorch

\- Torchvision

\- Ultralytics YOLO

\- RF-DETR

\- OpenCV

\- Scikit-learn

\- Roboflow

\- Matplotlib

\- Pandas

\- NumPy



\## Repository Structure



```text

.

├── README.md

├── requirements.txt

├── .gitignore

│

├── src/

│   ├── detection/

│   ├── classification/

│   └── preprocessing/

│

├── data/

│   └── README.md

│

├── assets/

│   └── README.md

│

└── docs/

