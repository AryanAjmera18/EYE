import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.exception.NetworkSecurityException import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_enity import DataTransformationArtifacts, DataIngestionArtifacts

class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifacts,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config

            # Allow path override via environment variables
            self.train_dir = os.environ.get("TRAIN_DIR", self.data_ingestion_artifact.trained_data_dir)
            self.test_dir = os.environ.get("TEST_DIR", self.data_ingestion_artifact.test_data_dir)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> tuple:
        logging.info("Entered initiate_data_transformation method")
        try:
            logging.info("Starting Data Transformation")

            # Define image transformations
            train_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            val_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

            # Load datasets
            train_dataset = ImageFolder(root=self.train_dir, transform=train_transforms)
            val_dataset = ImageFolder(root=self.test_dir, transform=val_transforms)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

            logging.info("Data Transformation completed successfully.")

            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_object_file_path="N/A for image transformations",
                transformed_train_file_path=self.train_dir,
                transformed_test_file_path=self.test_dir,
            )

            return train_loader, val_loader, data_transformation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)
