\# Data



This directory documents the datasets used in the project \*\*"Classification of Alexandrium minutum Occurrences and Blooms in Maritime Context."\*\*



The project used two image datasets to develop and evaluate the proposed deep-learning detection and classification pipeline.



\## Dataset 1 — Web-Based Image Dataset



The first dataset was constructed from images collected from web-based sources.



The data were prepared for the development and evaluation of the initial object-detection pipeline. The preparation process included image preprocessing, annotation, and data augmentation.



\## Dataset 2 — Cytometer Image Dataset



The second dataset consists of cytometer-derived images.



This dataset was used to further develop, optimize, and evaluate the proposed detection and classification pipeline on data originating from a different acquisition environment.



The dataset underwent dedicated preprocessing and annotation before model training and evaluation.



\## Data Preparation



Depending on the dataset and experimental stage, the preparation workflow included:



\- Image collection

\- Image preprocessing

\- Image annotation

\- Data augmentation

\- Dataset organization

\- Preparation for model training and evaluation



The prepared images were subsequently used for object detection with YOLOv8, YOLO11, and RF-DETR.



Detected cells were also processed for classification using ResNet18 and ResNet50.



\## Data Availability



The original datasets are \*\*not redistributed through this repository\*\*.



This repository provides the source code, methodology, documentation, and selected experimental results associated with the project.



For detailed information about data collection, preprocessing, annotation, and augmentation, refer to:



`../docs/project\_report.pdf`



\## Local Data Organization



Users who have access to the corresponding datasets can organize them locally according to the requirements of the training scripts.



Dataset paths should be configured locally and should not be committed to the repository.



\## Note



Some of the original experimental code was developed in Google Colab and used environment-specific paths. These paths have been removed or generalized in the public repository where appropriate.

