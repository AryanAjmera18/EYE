import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
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

            # Override paths via environment variables if provided
            self.train_dir = os.environ.get("TRAIN_DIR", self.data_ingestion_artifact.trained_data_dir)
            self.test_dir = os.environ.get("TEST_DIR", self.data_ingestion_artifact.test_data_dir)

            # Output transformed data
            self.output_train_dir = os.path.join("Artifacts", "transformed_data", "train")
            self.output_test_dir = os.path.join("Artifacts", "transformed_data", "test")

            os.makedirs(self.output_train_dir, exist_ok=True)
            os.makedirs(self.output_test_dir, exist_ok=True)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def transform_and_save_images(self, input_dir, output_dir, transform):
        dataset = ImageFolder(root=input_dir)
        for idx, (img_path, label) in enumerate(dataset.samples):
            img = Image.open(img_path).convert("RGB")
            transformed_img = transform(img)

            class_name = dataset.classes[label]
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            save_path = os.path.join(class_dir, f"{idx}.png")
            transforms.ToPILImage()(transformed_img).save(save_path)

    def initiate_data_transformation(self) -> tuple:
        logging.info("Entered initiate_data_transformation method")
        try:
            logging.info("🔁 Starting Data Transformation")

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

            # Save transformed images
            logging.info("🖼️ Transforming and saving training images...")
            self.transform_and_save_images(self.train_dir, self.output_train_dir, train_transforms)

            logging.info("🖼️ Transforming and saving validation images...")
            self.transform_and_save_images(self.test_dir, self.output_test_dir, val_transforms)

            # Load transformed datasets
            train_dataset = ImageFolder(root=self.output_train_dir, transform=transforms.ToTensor())
            val_dataset = ImageFolder(root=self.output_test_dir, transform=transforms.ToTensor())

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

            logging.info("✅ Data Transformation completed successfully.")

            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_object_file_path="N/A for image transformations",
                transformed_train_file_path=self.output_train_dir,
                transformed_test_file_path=self.output_test_dir,
            )

            return train_loader, val_loader, data_transformation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)
