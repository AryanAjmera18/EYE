import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import f1_score, precision_score, recall_score
import mlflow
from mlflow.tracking import MlflowClient

from src.logging.logger import logging
from src.exception.NetworkSecurityException import NetworkSecurityException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_enity import ModelTrainerArtifacts, ClassificationMetricArtifacts


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, num_epochs=20):
        self.model_trainer_config = model_trainer_config
        self.num_epochs = num_epochs

    def initiate_model_trainer(self, train_loader, val_loader):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = len(train_loader.dataset.dataset.classes)

            # Load DenseNet121 and freeze all but last two blocks
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            features = list(model.features.children())
            for idx, module in enumerate(features):
                if idx < len(features) - 2:
                    for param in module.parameters():
                        param.requires_grad = False

            model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(model.classifier.in_features, num_classes)
            )
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
            early_stopping = EarlyStopping(patience=5)

            # MLflow setup
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "EyeDiseaseDetection")
            artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION", os.path.join(os.getcwd(), "mlruns"))

            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            if not client.get_experiment_by_name(experiment_name):
                client.create_experiment(name=experiment_name, artifact_location=artifact_location)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="DenseNet121_Training"):
                mlflow.log_param("epochs", self.num_epochs)

                for epoch in range(self.num_epochs):
                    model.train()
                    running_train_loss, correct_train = 0.0, 0
                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_train_loss += loss.item()
                        correct_train += (outputs.argmax(1) == labels).sum().item()
                    train_loss = running_train_loss / len(train_loader)
                    train_acc = correct_train / len(train_loader.dataset)

                    # Validation loop
                    model.eval()
                    val_loss, correct_val = 0.0, 0
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            preds = outputs.argmax(1)
                            correct_val += (preds == labels).sum().item()
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    val_loss /= len(val_loader)
                    val_acc = correct_val / len(val_loader.dataset)
                    f1 = f1_score(all_labels, all_preds, average="macro")
                    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
                    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall
                    }, step=epoch)

                    print(f"Epoch {epoch+1}/{self.num_epochs} | Train Loss: {train_loss:.4f}, "
                          f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f} | F1: {f1:.4f}")

                    scheduler.step(val_loss)
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        print("🛑 Early stopping triggered!")
                        break

                # Save and log the model
                model_dir = os.environ.get("MODEL_SAVE_DIR", os.path.dirname(self.model_trainer_config.trained_model_file_path))
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "model.pth")
                torch.save(model, model_path)
                mlflow.pytorch.log_model(model, "model")
                logging.info(f"✅ Model saved at {model_path} and logged to MLflow")

                metrics = ClassificationMetricArtifacts(f1, precision, recall)
                return ModelTrainerArtifacts(model_path, metrics, metrics)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
