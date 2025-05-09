from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from src.components.data_ingestion import DataIngestion

# Step 1: Create TrainingPipelineConfig
training_pipeline_config = TrainingPipelineConfig()

# Step 2: Create DataIngestionConfig with TrainingPipelineConfig
data_ingestion_config = DataIngestionConfig(training_pipeline_config)

# Step 3: Initialize DataIngestion
data_ingestion = DataIngestion(data_ingestion_config)

# Step 4: Run Ingestion
data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

# Step 5 (Optional): Print paths
print(f"Train Directory: {data_ingestion_artifact.trained_data_dir}")
print(f"Test Directory: {data_ingestion_artifact.test_data_dir}")
