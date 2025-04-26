import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from src.exception import NetworkSecurityException
from src.logging.logger import logging

if __name__ == "__main__":
    try:
        logging.info("Starting Main Pipeline Execution")
        
        # Setup
        training_pipeline_config = TrainingPipelineConfig()

        # 1. Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating Data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # 2. Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiating Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed")

        # 3. Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_ingestion_artifact, data_transformation_config)
        logging.info("Initiating Data Transformation")
        train_loader, val_loader, data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed")

        # 4. Model Training
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, num_epochs=1)  # <-- limit to 1 epoch
        logging.info("Initiating Model Training")
        model_trainer_artifact = model_trainer.initiate_model_trainer(train_loader, val_loader)
        logging.info("Model Training completed")
        
        print(f"Model saved at: {model_trainer_artifact.trained_model_file_path}")
        print(f"Training F1 Score: {model_trainer_artifact.train_metric_artifact.f1_score:.4f}")
        print(f"Validation F1 Score: {model_trainer_artifact.test_metric_artifact.f1_score:.4f}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
