from src.exception.NetworkSecurityException import NetworkSecurityException
from src.logging.logger import logging
from src.entity.artifact_enity import ClassificationMetricArtifacts
from sklearn.metrics import f1_score,precision_score,recall_score
import os , sys

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifacts:
    try:
        model_f1_score = f1_score(y_true,y_pred)
        model_precision_score = precision_score(y_true,y_pred)
        model_recall_score = recall_score(y_true,y_pred)
        
        classification_metric = ClassificationMetricArtifacts(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    