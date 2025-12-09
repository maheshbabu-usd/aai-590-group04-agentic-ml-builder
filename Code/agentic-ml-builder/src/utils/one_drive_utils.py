# ============================================================================
# one_drive_utils.py
# ============================================================================
"""
Utilities for interacting with Microsoft OneDrive via Microsoft Graph.

Module responsibilities
- Authenticate using client credentials and MSAL.
- Read JSON files and upload/download files or folders to OneDrive.
- Provide simple convenience wrappers used by the Agentic ML Builder.

Environment variables
- `ONEDRIVE_CLIENT_ID` - Application (client) id for the Azure AD app
- `ONEDRIVE_CLIENT_SECRET` - Client secret for the Azure AD app
- `ONEDRIVE_TENANT_ID` - Tenant id where the Azure AD app is registered

Usage example
```
from utils.one_drive_utils import OneDriveClient

client = OneDriveClient()
content = await client.read_json('path/to/file.json')
```
Note: the client methods are declared `async` for compatibility with
async codepaths; they perform blocking HTTP calls using `requests`.
Consider switching to an async HTTP client if non-blocking behavior is
required.
"""
import logging
import os
from pathlib import Path
from typing import Dict
from msal import ConfidentialClientApplication
import requests
import json

logger = logging.getLogger(__name__)

class OneDriveClient:
    """Client for OneDrive file operations using Microsoft Graph.

    Responsibilities:
    - Acquire an app-only access token using MSAL (`ConfidentialClientApplication`).
    - Provide helper methods to read JSON files, upload files/folders and
      download files from the signed-in user's OneDrive.

    Behavior notes:
    - If required environment variables are not set the client will not be
      initialized and method calls will raise `RuntimeError`.
    - Methods are declared `async` to allow use from async code, but they
      currently perform synchronous HTTP calls via `requests`.
    """

    def __init__(self):
        self.client_id = os.getenv("ONEDRIVE_CLIENT_ID")
        self.client_secret = os.getenv("ONEDRIVE_CLIENT_SECRET")
        self.tenant_id = os.getenv("ONEDRIVE_TENANT_ID")

        if all([self.client_id, self.client_secret, self.tenant_id]):
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
            logger.warning("OneDrive client not initialized - missing credentials")

    def _get_token(self) -> str:
        """Get access token"""
        result = self.app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )

        if "access_token" in result:
            return result["access_token"]
        else:
            raise Exception(f"Could not acquire token: {result.get('error_description')}")

    def _get_headers(self) -> Dict:
        """Get authorization headers"""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def read_json(self, file_path: str) -> Dict:
        """Read JSON file from OneDrive"""
        if not self.token:
            raise RuntimeError("OneDrive client not initialized")

        # Convert path to OneDrive path
        onedrive_path = file_path.replace('\\', '/')
        url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{onedrive_path}:/content"

        response = requests.get(url, headers=self._get_headers())

        if response.status_code == 200:
            return json.loads(response.text)
        else:
            raise Exception(f"Failed to read file: {response.status_code} - {response.text}")

    async def upload_file(self, local_path: Path, remote_path: str):
        """Upload file to OneDrive"""
        if not self.token:
            raise RuntimeError("OneDrive client not initialized")

        with open(local_path, 'rb') as f:
            content = f.read()

        # Convert path to OneDrive path
        onedrive_path = remote_path.replace('\\', '/')
        url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{onedrive_path}:/content"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/octet-stream",
        }

        response = requests.put(url, headers=headers, data=content)

        if response.status_code in [200, 201]:
            logger.info(f"Uploaded {local_path} to OneDrive: {remote_path}")
        else:
            raise Exception(f"Failed to upload file: {response.status_code} - {response.text}")

    async def upload_folder(self, local_folder: Path, remote_folder: str = None):
        """Upload entire folder to OneDrive"""
        if not self.token:
            raise RuntimeError("OneDrive client not initialized")

        if remote_folder is None:
            remote_folder = f"/AgenticMLBuilder/output/{local_folder.name}"

        for file_path in local_folder.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_folder)
                remote_file_path = f"{remote_folder}/{relative_path}".replace('\\', '/')
                await self.upload_file(file_path, remote_file_path)

        logger.info(f"Uploaded folder {local_folder} to OneDrive: {remote_folder}")

    async def download_file(self, remote_path: str, local_path: Path):
        """Download file from OneDrive"""
        if not self.token:
            raise RuntimeError("OneDrive client not initialized")

        onedrive_path = remote_path.replace('\\', '/')
        url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{onedrive_path}:/content"

        response = requests.get(url, headers=self._get_headers())

        if response.status_code == 200:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {remote_path} to {local_path}")
        else:
            raise Exception(f"Failed to download file: {response.status_code} - {response.text}")
