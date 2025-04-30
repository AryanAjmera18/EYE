import os
import shutil
import sys
import random
from torchvision import datasets
from sklearn.model_selection import train_test_split
from src.exception.NetworkSecurityException import NetworkSecurityException 
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_enity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

            # Override hardcoded paths using environment variables if provided
            self.input_data_dir = os.environ.get("INPUT_DATA_DIR", self.data_ingestion_config.input_data_dir)
            self.feature_store_dir = os.environ.get("FEATURE_STORE_DIR", self.data_ingestion_config.feature_store_dir)
            self.training_dir = os.environ.get("TRAIN_DIR", self.data_ingestion_config.training_dir)
            self.testing_dir = os.environ.get("TEST_DIR", self.data_ingestion_config.testing_dir)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def copy_input_data_to_feature_store(self):
        try:
            if os.path.exists(self.feature_store_dir):
                shutil.rmtree(self.feature_store_dir)
            shutil.copytree(self.input_data_dir, self.feature_store_dir)

            logging.info(f"Copied input data to feature store at {self.feature_store_dir}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self):
        try:
            dataset = datasets.ImageFolder(self.feature_store_dir)
            total_size = len(dataset)
            indices = list(range(total_size))
            random.shuffle(indices)

            split = int(self.data_ingestion_config.train_split_ratio * total_size)
            train_indices, test_indices = indices[:split], indices[split:]

            class_to_idx = dataset.class_to_idx
            idx_to_class = {v: k for k, v in class_to_idx.items()}

            # Create train and test folders
            for target_dir in [self.training_dir, self.testing_dir]:
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                for class_name in class_to_idx.keys():
                    os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

            # Move images to respective folders
            for idx in train_indices:
                path, _ = dataset.samples[idx]
                class_name = idx_to_class[dataset.targets[idx]]
                shutil.copy(path, os.path.join(self.training_dir, class_name, os.path.basename(path)))

            for idx in test_indices:
                path, _ = dataset.samples[idx]
                class_name = idx_to_class[dataset.targets[idx]]
                shutil.copy(path, os.path.join(self.testing_dir, class_name, os.path.basename(path)))

            logging.info("Train and Test split completed.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            self.copy_input_data_to_feature_store()
            self.split_data_as_train_test()

            data_ingestion_artifacts = DataIngestionArtifacts(
                trained_data_dir=self.training_dir,
                test_data_dir=self.testing_dir
            )

            logging.info("Data Ingestion process completed.")
            return data_ingestion_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)
