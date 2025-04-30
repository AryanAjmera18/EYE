import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import sys
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score

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
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(train_loader.dataset.classes))
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(patience=5, verbose=True)

            # Use env variable or default local MLflow
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "EyeDiseaseDetection")

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="TrainingPipelineRun"):
                mlflow.log_param("epochs", self.num_epochs)

                for epoch in range(self.num_epochs):
                    model.train()
                    running_loss = 0.0

                    for inputs, labels in train_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * inputs.size(0)

                    train_loss = running_loss / len(train_loader.dataset)

                    model.eval()
                    val_loss = 0.0
                    all_preds = []
                    all_labels = []

                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item() * inputs.size(0)
                            preds = torch.argmax(outputs, dim=1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())

                    val_loss /= len(val_loader.dataset)

                    f1 = f1_score(all_labels, all_preds, average='macro')
                    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("f1_score", f1, step=epoch)
                    mlflow.log_metric("precision", precision, step=epoch)
                    mlflow.log_metric("recall", recall, step=epoch)

                    logging.info(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}")

                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        logging.info("Early stopping triggered!")
                        break

                # Save model
                base_model_dir = os.environ.get("MODEL_SAVE_DIR", os.path.dirname(self.model_trainer_config.trained_model_file_path))
                os.makedirs(base_model_dir, exist_ok=True)
                model_save_path = os.path.join(base_model_dir, "model.pth")

                torch.save(model, model_save_path)
                mlflow.pytorch.log_model(model, "model")

                logging.info(f"Model saved locally at {model_save_path} and logged to MLflow.")

                train_metrics = ClassificationMetricArtifacts(f1_score=f1, precision_score=precision, recall_score=recall)
                test_metrics = ClassificationMetricArtifacts(f1_score=f1, precision_score=precision, recall_score=recall)

                return ModelTrainerArtifacts(
                    trained_model_file_path=model_save_path,
                    train_metric_artifact=train_metrics,
                    test_metric_artifact=test_metrics
                )

        except Exception as e:
            raise NetworkSecurityException(e, sys)
