# ============================================================================
# utility_files.py
# ============================================================================
"""
Collection of small utility helpers used by Agentic ML Builder.

This module groups common file I/O helpers (JSON/YAML/text), a basic
logging setup helper, and lightweight wrappers for Azure and OneDrive
operations used elsewhere in the project.

Responsibilities
- Read and write JSON/YAML/text files using `pathlib.Path`.
- Provide `setup_logging` to configure file + console logging.
- Provide thin wrappers for Azure ML, Azure Blob Storage and OneDrive
    operations (initialization, upload/download, model registration).

Environment variables used by the cloud helpers
- `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_AI_PROJECT_NAME`
- `AZURE_STORAGE_CONNECTION_STRING`
- `ONEDRIVE_CLIENT_ID`, `ONEDRIVE_CLIENT_SECRET`, `ONEDRIVE_TENANT_ID`

Usage example
```
from utils.utility_files import read_yaml, AzureClient, OneDriveClient

cfg = read_yaml(Path('config/experiment.yaml'))
ml = AzureClient()
onedrive = OneDriveClient()
```
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict
import yaml

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "ml_builder.log"),
            logging.StreamHandler()
        ]
    )

def read_json(filepath: Path) -> Dict:
    """Read JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(filepath: Path, data: Dict):
    """
    utility_files - Consolidated utility helpers for Agentic ML Builder.

    This file provides small, well-documented helpers for file I/O, logging
    setup, and thin wrappers around cloud operations used by the project.

    Note: cloud client classes are lightweight wrappers that expect the
    appropriate SDK packages to be available at runtime (Azure SDK, MSAL).
    """

    import json
    import logging
    import os
    from pathlib import Path
    from typing import Dict

    import yaml
    import requests

    logger = logging.getLogger(__name__)


    def setup_logging(level: str = "INFO") -> None:
        """Configure basic file + console logging.

        Creates `logs/ml_builder.log` and adds a console handler.
        """
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "ml_builder.log"),
                logging.StreamHandler(),
            ],
        )


    def read_json(filepath: Path) -> Dict:
        """Read and return JSON content from `filepath`."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


    def write_json(filepath: Path, data: Dict) -> None:
        """Write `data` as pretty JSON to `filepath`, creating parents."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


    def read_yaml(filepath: Path) -> Dict:
        """Read and return YAML content from `filepath`."""
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


    def write_yaml(filepath: Path, data: Dict) -> None:
        """Write `data` as YAML to `filepath`, creating parents."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)


    def write_file(filepath: Path, content: str) -> None:
        """Write text `content` to `filepath`, creating parents."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


    def read_file(filepath: Path) -> str:
        """Return text content of `filepath`."""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


    class AzureClient:
        """Client wrapper for Azure Machine Learning operations.

        Responsibilities:
        - Initialize `azure.ai.ml.MLClient` using `DefaultAzureCredential` and
          environment variables (`AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`,
          `AZURE_AI_PROJECT_NAME`).
        - Submit training jobs via `submit_training_job`.
        - Register models via `register_model`.

        If required environment variables are missing the underlying ML client
        will not be initialized and calls that require it will raise
        `RuntimeError`.
        """

        def __init__(self):
            try:
                from azure.identity import DefaultAzureCredential
                from azure.ai.ml import MLClient
            except Exception:
                DefaultAzureCredential = None  # type: ignore
                MLClient = None  # type: ignore

            self.credential = DefaultAzureCredential() if DefaultAzureCredential else None
            self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
            self.workspace_name = os.getenv("AZURE_AI_PROJECT_NAME")

            if all([self.subscription_id, self.resource_group, self.workspace_name, self.credential, MLClient]):
                self.ml_client = MLClient(
                    credential=self.credential,
                    subscription_id=self.subscription_id,
                    resource_group_name=self.resource_group,
                    workspace_name=self.workspace_name,
                )
                logger.info("Azure ML client initialized")
            else:
                self.ml_client = None
                logger.warning("Azure ML client not initialized - missing configuration or SDK")

        def submit_training_job(self, job_config: Dict):
            """Submit a training job to Azure ML using a minimal job spec.

            `job_config` is expected to contain keys like `code_path`, `command`,
            `environment`, `compute_target`, `display_name` and `experiment_name`.
            """
            if not self.ml_client:
                raise RuntimeError("Azure ML client not initialized")

            from azure.ai.ml import command

            job = command(
                code=job_config.get("code_path", "./"),
                command=job_config.get("command"),
                environment=job_config.get("environment"),
                compute=job_config.get("compute_target", "gpu-cluster"),
                display_name=job_config.get("display_name"),
                experiment_name=job_config.get("experiment_name"),
            )

            submitted_job = self.ml_client.jobs.create_or_update(job)
            logger.info(f"Job submitted: {submitted_job.name}")
            return submitted_job

        def register_model(self, model_path: Path, model_name: str):
            """Register a model artifact in Azure ML.

            Returns the registered model resource.
            """
            if not self.ml_client:
                raise RuntimeError("Azure ML client not initialized")

            from azure.ai.ml.entities import Model
            from azure.ai.ml.constants import AssetTypes

            model = Model(
                path=str(model_path),
                type=AssetTypes.CUSTOM_MODEL,
                name=model_name,
                description="Model generated by Agentic ML Builder",
            )

            registered_model = self.ml_client.models.create_or_update(model)
            logger.info(f"Model registered: {registered_model.name}")
            return registered_model


    class AzureStorageClient:
        """Thin wrapper around Azure Blob Storage operations.

        Uses `AZURE_STORAGE_CONNECTION_STRING` to initialize a client when
        available. Upload/download helpers are provided.
        """

        def __init__(self):
            try:
                from azure.storage.blob import BlobServiceClient
            except Exception:
                BlobServiceClient = None  # type: ignore

            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if connection_string and BlobServiceClient:
                self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                logger.info("Azure Storage client initialized")
            else:
                self.blob_service_client = None
                logger.warning("Azure Storage client not initialized - missing connection string or SDK")

        def upload_file(self, file_path: Path, container_name: str, blob_name: str) -> None:
            """Upload `file_path` to `container_name/blob_name`."""
            if not self.blob_service_client:
                raise RuntimeError("Azure Storage client not initialized")

            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            logger.info(f"Uploaded {file_path} to {container_name}/{blob_name}")

        def download_file(self, container_name: str, blob_name: str, download_path: Path) -> None:
            """Download blob to local `download_path`."""
            if not self.blob_service_client:
                raise RuntimeError("Azure Storage client not initialized")

            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            download_path.parent.mkdir(parents=True, exist_ok=True)
            with open(download_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())

            logger.info(f"Downloaded {container_name}/{blob_name} to {download_path}")


    class OneDriveClient:
        """Client for OneDrive file operations using Microsoft Graph.

        Responsibilities:
        - Acquire an app-only token using MSAL.
        - Read JSON files, upload files/folders, and download files from OneDrive.

        Note: methods are `async` for compatibility but use `requests` (blocking).
        """

        def __init__(self):
            try:
                from msal import ConfidentialClientApplication
            except Exception:
                ConfidentialClientApplication = None  # type: ignore

            self.client_id = os.getenv("ONEDRIVE_CLIENT_ID")
            self.client_secret = os.getenv("ONEDRIVE_CLIENT_SECRET")
            self.tenant_id = os.getenv("ONEDRIVE_TENANT_ID")

            if all([self.client_id, self.client_secret, self.tenant_id, ConfidentialClientApplication]):
                self.app = ConfidentialClientApplication(
                    self.client_id,
                    authority=f"https://login.microsoftonline.com/{self.tenant_id}",
                    client_credential=self.client_secret,
                )
                self.token = self._get_token()
                logger.info("OneDrive client initialized")
            else:
                self.app = None
                self.token = None
                logger.warning("OneDrive client not initialized - missing credentials or MSAL")

        def _get_token(self) -> str:
            """Acquire an app-only access token for Graph."""
            result = self.app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
            if "access_token" in result:
                return result["access_token"]
            raise RuntimeError(f"Could not acquire token: {result.get('error_description')}")

        def _get_headers(self) -> Dict:
            """Return authorization headers for Graph requests."""
            return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

        async def read_json(self, file_path: str) -> Dict:
            """Read a JSON file from OneDrive and return parsed content."""
            if not self.token:
                raise RuntimeError("OneDrive client not initialized")

            onedrive_path = file_path.replace("\\", "/")
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{onedrive_path}:/content"
            response = requests.get(url, headers=self._get_headers())
            if response.status_code == 200:
                return response.json()
            raise RuntimeError(f"Failed to read file: {response.status_code} - {response.text}")

        async def upload_file(self, local_path: Path, remote_path: str) -> None:
            """Upload local file to OneDrive path `remote_path`."""
            if not self.token:
                raise RuntimeError("OneDrive client not initialized")

            with open(local_path, "rb") as f:
                content = f.read()

            onedrive_path = remote_path.replace("\\", "/")
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{onedrive_path}:/content"
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/octet-stream"}
            response = requests.put(url, headers=headers, data=content)
            if response.status_code not in (200, 201):
                raise RuntimeError(f"Failed to upload file: {response.status_code} - {response.text}")
            logger.info(f"Uploaded {local_path} to OneDrive: {remote_path}")

        async def upload_folder(self, local_folder: Path, remote_folder: str = None) -> None:
            """Recursively upload a local folder to OneDrive under `remote_folder`."""
            if not self.token:
                raise RuntimeError("OneDrive client not initialized")

            if remote_folder is None:
                remote_folder = f"/AgenticMLBuilder/output/{local_folder.name}"

            for file_path in local_folder.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_folder)
                    remote_file_path = f"{remote_folder}/{relative_path}".replace("\\", "/")
                    await self.upload_file(file_path, remote_file_path)

            logger.info(f"Uploaded folder {local_folder} to OneDrive: {remote_folder}")

        async def download_file(self, remote_path: str, local_path: Path) -> None:
            """Download a file from OneDrive to local `local_path`."""
            if not self.token:
                raise RuntimeError("OneDrive client not initialized")

            onedrive_path = remote_path.replace("\\", "/")
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{onedrive_path}:/content"
            response = requests.get(url, headers=self._get_headers())
            if response.status_code == 200:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded {remote_path} to {local_path}")
                return
            raise RuntimeError(f"Failed to download file: {response.status_code} - {response.text}")
