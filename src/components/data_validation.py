import os
import sys
import yaml
from src.entity.artifact_enity import DataIngestionArtifacts, DataValidationArtifacts
from src.entity.config_entity import DataValidationConfig
from src.exception import NetworkSecurityException
from src.logging.logger import logging

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifacts, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_folder_structure(self) -> bool:
        try:
            train_dir = self.data_ingestion_artifact.trained_data_dir
            test_dir = self.data_ingestion_artifact.test_data_dir

            if not os.path.exists(train_dir) or not os.listdir(train_dir):
                logging.error(f"Training directory {train_dir} is missing or empty.")
                return False
            if not os.path.exists(test_dir) or not os.listdir(test_dir):
                logging.error(f"Testing directory {test_dir} is missing or empty.")
                return False

            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_class_consistency(self) -> bool:
        try:
            train_classes = sorted(os.listdir(self.data_ingestion_artifact.trained_data_dir))
            test_classes = sorted(os.listdir(self.data_ingestion_artifact.test_data_dir))

            if train_classes != test_classes:
                logging.error(f"Mismatch between train and test classes.\nTrain: {train_classes}\nTest: {test_classes}")
                return False

            return True
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def generate_drift_report(self) -> None:
        try:
            drift_report = {}
            train_dir = self.data_ingestion_artifact.trained_data_dir
            test_dir = self.data_ingestion_artifact.test_data_dir

            for class_name in os.listdir(train_dir):
                train_images = len(os.listdir(os.path.join(train_dir, class_name)))
                test_images = len(os.listdir(os.path.join(test_dir, class_name)))

                if train_images == 0:
                    drift = 1.0
                else:
                    drift = abs(train_images - test_images) / train_images

                drift_report[class_name] = {
                    "train_images": train_images,
                    "test_images": test_images,
                    "drift_ratio": round(drift, 3)
                }

            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)

            with open(drift_report_path, "w") as f:
                yaml.dump(drift_report, f)

            logging.info(f"Drift report saved at {drift_report_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifacts:
        try:
            logging.info("Starting Data Validation Process...")

            folder_structure_valid = self.validate_folder_structure()
            class_consistency_valid = self.validate_class_consistency()

            validation_status = folder_structure_valid and class_consistency_valid

            if validation_status:
                self.generate_drift_report()
            else:
                logging.error("Data Validation Failed.")

            data_validation_artifacts = DataValidationArtifacts(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_data_dir,
                valid_test_file_path=self.data_ingestion_artifact.test_data_dir,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)

