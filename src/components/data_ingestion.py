import os
import shutil
import sys
import random
from torchvision import datasets
from sklearn.model_selection import train_test_split
from src.exception import NetworkSecurityException 
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_enity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def copy_input_data_to_feature_store(self):
        try:
            input_data_dir = self.data_ingestion_config.input_data_dir
            feature_store_dir = self.data_ingestion_config.feature_store_dir

            if os.path.exists(feature_store_dir):
                shutil.rmtree(feature_store_dir)
            shutil.copytree(input_data_dir, feature_store_dir)

            logging.info(f"Copied input data to feature store at {feature_store_dir}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self):
        try:
            feature_store_dir = self.data_ingestion_config.feature_store_dir
            train_dir = self.data_ingestion_config.training_dir
            test_dir = self.data_ingestion_config.testing_dir

            dataset = datasets.ImageFolder(feature_store_dir)
            total_size = len(dataset)
            indices = list(range(total_size))
            random.shuffle(indices)

            split = int(self.data_ingestion_config.train_split_ratio * total_size)
            train_indices, test_indices = indices[:split], indices[split:]

            class_to_idx = dataset.class_to_idx
            idx_to_class = {v: k for k, v in class_to_idx.items()}

            # Create train and test folders
            for target_dir in [train_dir, test_dir]:
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                for class_name in class_to_idx.keys():
                    os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

            # Move images to respective folders
            for idx in train_indices:
                path, _ = dataset.samples[idx]
                class_name = idx_to_class[dataset.targets[idx]]
                shutil.copy(path, os.path.join(train_dir, class_name, os.path.basename(path)))

            for idx in test_indices:
                path, _ = dataset.samples[idx]
                class_name = idx_to_class[dataset.targets[idx]]
                shutil.copy(path, os.path.join(test_dir, class_name, os.path.basename(path)))

            logging.info("Train and Test split completed.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            self.copy_input_data_to_feature_store()
            self.split_data_as_train_test()

            data_ingestion_artifacts = DataIngestionArtifacts(
                trained_data_dir=self.data_ingestion_config.training_dir,
                test_data_dir=self.data_ingestion_config.testing_dir
            )

            logging.info("Data Ingestion process completed.")
            return data_ingestion_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)


