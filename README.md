# Automated Eye Disease Detection System - Project Report

## Project Overview

The goal of this project is to build a fully functional MLOps pipeline for automated diagnosis of eye diseases using retinal images. The system is designed to integrate into small private eye clinics and streamline ophthalmologists' diagnostic workflows.

Our solution ingests raw retinal images, performs validation and transformation, fine-tunes a deep learning model (ResNet-50), and prepares artifacts for deployment. All components are modular, scalable, and built with reproducibility in mind.

---

## Key Components

### 1. Data Ingestion
- **Input:** Raw retinal images organized into folders (one folder per disease class).
- **Process:**
  - Copy the dataset manually into the `input_data/` directory (since large datasets cannot be stored on GitHub).
  - Copy the dataset from `input_data/` into a `feature_store/` directory.
  - Split the images into training and testing sets based on an 80-20 split.
  - Maintain class-wise folder structure.
- **Output:**
  - Training directory: `Artifacts/<timestamp>/data_ingestion/ingested/train/`
  - Testing directory: `Artifacts/<timestamp>/data_ingestion/ingested/test/`

### 2. Data Validation
- **Checks Performed:**
  - Ensure train and test directories are non-empty.
  - Verify that both sets contain the same disease classes.
  - Generate a YAML drift report comparing the number of images per class between train and test.
- **Output:**
  - Validation status (pass/fail)
  - Drift report saved at: `Artifacts/<timestamp>/data_validation/drift_report/report.yaml`

### 3. Data Transformation
- **Process:**
  - Apply transformations to images:
    - Resize to (224, 224)
    - Normalize using ImageNet statistics
    - Data augmentation for training: random horizontal flips, slight rotations.
  - Create PyTorch DataLoaders for efficient training and validation.
- **Output:**
  - TrainLoader and ValLoader objects
  - Artifacts pointing to transformed train/test directories

### 4. Model Training
- **Model Used:** ResNet-50 (pretrained on ImageNet)
- **Strategy:**
  - Replace the final fully connected layer to match the number of disease classes.
  - Train for 1 epoch (for initial local testing) using Adam optimizer and CrossEntropyLoss.
  - Implement EarlyStopping based on validation loss to prevent overfitting.
- **Metrics Tracked:**
  - F1 Score (Macro)
  - Precision
  - Recall
- **Output:**
  - Best model saved at: `Artifacts/<timestamp>/model_trainer/trained_model/model.pth`
  - Training and Validation metrics saved in artifacts.

### 5. Artifact Management
- All generated files (datasets, models, metrics, reports) are systematically organized under an `Artifacts/` directory with timestamped versions.
- Ensures traceability and reproducibility across the project lifecycle.

---

## Tools & Frameworks Used
- **PyTorch**: Model development, training, evaluation.
- **TorchVision**: Pretrained models, data augmentations.
- **Scikit-learn**: Evaluation metrics (F1, Precision, Recall).
- **YAML**: Drift report storage.
- **Custom Python Packages**: Exception handling, logging, config management.


## Current Status
- Full pipeline from ingestion to model saving is operational.
- Model training limited to 1 epoch for quick local testing.
- All stages log outputs systematically.
- Drift reports and artifacts are correctly generated for auditability.


## How to Run the Project

1. Clone the repository locally.
2. Manually place the dataset (eye disease images organized by class) into the `input_data/` folder.
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the complete pipeline:
   ```bash
   python main.py
   ```
5. Monitor outputs and artifacts in the `Artifacts/` folder.

*Note: Ensure enough storage and compute power, especially when running on large datasets or multiple epochs.*

---

## Next Steps (Optional Enhancements)
- Train for 10–20 epochs to improve F1 Score.
- Move training and serving to Chameleon Cloud.
- Implement MLflow/Dagshub experiment tracking.
- Serve model via FastAPI/Flask for real-time inference.
- Automate retraining with weekly data refresh.


---

# Conclusion

This project sets up a strong foundation for AI-assisted clinical diagnostics for eye diseases. It follows modular MLOps best practices, ensuring the system is scalable, maintainable, and production-ready.

The team can now confidently extend this work toward deployment and scaling in real-world clinical settings.

