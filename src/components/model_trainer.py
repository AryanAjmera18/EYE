import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import sys
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score
from mlflow.tracking import MlflowClient

from src.logging.logger import logging
from src.exception.NetworkSecurityException import NetworkSecurityException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_enity import ModelTrainerArtifacts, ClassificationMetricArtifacts

class EarlyStopping:
    def __init__(self, patience=3, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, num_epochs=1):
        self.model_trainer_config = model_trainer_config
        self.num_epochs = num_epochs

    def initiate_model_trainer(self, train_loader, val_loader):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(train_loader.dataset.classes))
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(patience=5, verbose=True)

            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "EyeDiseaseDetection")
            artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION", os.path.join(os.getcwd(), "mlruns"))

            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            if not client.get_experiment_by_name(experiment_name):
                client.create_experiment(name=experiment_name, artifact_location=artifact_location)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="TrainingPipelineRun"):
                mlflow.log_param("epochs", self.num_epochs)

                for epoch in range(self.num_epochs):
                    model.train()
                    train_loss = sum(self._train_batch(model, optimizer, criterion, device, batch)
                                     for batch in train_loader) / len(train_loader.dataset)

                    val_loss, all_preds, all_labels = self._validate(model, criterion, device, val_loader)
                    f1 = f1_score(all_labels, all_preds, average='macro')
                    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "f1_score": f1,
                        "precision": precision,
                        "recall": recall
                    }, step=epoch)

                    logging.info(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, "
                                 f"Val Loss: {val_loss:.4f}, F1: {f1:.4f}")

                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        break

                model_dir = os.environ.get("MODEL_SAVE_DIR", os.path.dirname(self.model_trainer_config.trained_model_file_path))
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, "model.pth")
                torch.save(model, model_path)

                mlflow.pytorch.log_model(model, "model")
                logging.info(f"Model saved locally at {model_path} and logged to MLflow.")

                metrics = ClassificationMetricArtifacts(f1, precision, recall)
                return ModelTrainerArtifacts(model_path, metrics, metrics)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _train_batch(self, model, optimizer, criterion, device, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item() * inputs.size(0)

    def _validate(self, model, criterion, device, val_loader):
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return val_loss / len(val_loader.dataset), all_preds, all_labels
