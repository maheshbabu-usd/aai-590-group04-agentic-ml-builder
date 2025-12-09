"""MLflow configuration and logging utilities"""
import mlflow
import mlflow.pytorch
from pathlib import Path

class MLflowTracker:
    """MLflow experiment tracker"""
    
    def __init__(self, experiment_name: str = "scene_classifier"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: str = None):
        """Start MLflow run"""
        mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: dict):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifact(self, local_path: str):
        """Log artifact file"""
        mlflow.log_artifact(local_path)
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()

# Usage example:
# tracker = MLflowTracker()
# tracker.start_run("training_run_1")
# tracker.log_params({"lr": 0.001, "batch_size": 32})
# tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
# tracker.log_model(model)
# tracker.end_run()
