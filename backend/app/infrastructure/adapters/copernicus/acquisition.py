import logging
import os
import time

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class CDSECredentialsMissingError(Exception):
    """Raised when Copernicus CDSE credentials are not configured in environment."""


class CDSEAuthenticationError(Exception):
    """Raised when CDSE token request fails."""


class CopernicusAcquisitionClient:
    """
    Handles authenticated token acquisition and asset downloading from Copernicus Data Space.
    Auth URL: https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.token_url = self.settings.CDSE_TOKEN_URL
        self._cached_token: str | None = None
        self._token_expires_at: float = 0.0

    def is_configured(self) -> bool:
        """Check if CDSE credentials are provided in settings."""
        has_client = bool(self.settings.CDSE_CLIENT_ID and self.settings.CDSE_CLIENT_SECRET)
        has_user = bool(self.settings.CDSE_USERNAME and self.settings.CDSE_PASSWORD)
        return has_client or has_user

    async def get_access_token(self) -> str:
        """
        Acquire or refresh an OAuth2 token using either client_credentials or password grant.
        """
        now = time.time()
        # Return valid cached token if within expiry buffer
        if self._cached_token and now < (self._token_expires_at - 60):
            return self._cached_token

        if not self.is_configured():
            raise CDSECredentialsMissingError(
                "Copernicus Data Space credentials not configured. Please set CDSE_CLIENT_ID and "
                "CDSE_CLIENT_SECRET (or CDSE_USERNAME and CDSE_PASSWORD) in your environment variables."
            )

        data: dict[str, str] = {}
        if self.settings.CDSE_CLIENT_ID and self.settings.CDSE_CLIENT_SECRET:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.settings.CDSE_CLIENT_ID,
                "client_secret": self.settings.CDSE_CLIENT_SECRET,
            }
        else:
            data = {
                "grant_type": "password",
                "username": self.settings.CDSE_USERNAME,
                "password": self.settings.CDSE_PASSWORD,
                "client_id": "cdse-public",
            }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                res = await client.post(self.token_url, data=data)
                if res.status_code != 200:
                    logger.error("CDSE Auth failed (%d): %s", res.status_code, res.text[:200])
                    raise CDSEAuthenticationError(
                        f"Failed to authenticate with CDSE Keycloak: HTTP {res.status_code}"
                    )
                
                token_data = res.json()
                self._cached_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 600)
                self._token_expires_at = now + expires_in
                return self._cached_token
        except httpx.RequestError as exc:
            raise CDSEAuthenticationError(f"Network error connecting to CDSE token endpoint: {exc}")

    async def download_asset(
        self,
        download_url: str,
        target_path: str,
        max_bytes: int = 100 * 1024 * 1024,  # Safety cap: 100 MB default
    ) -> str:
        """
        Download a SAR asset (or tile / preview chip) with authentication and streaming.
        """
        token = await self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            async with client.stream("GET", download_url, headers=headers) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Asset download failed with status {response.status_code}")

                total_downloaded = 0
                with open(target_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        total_downloaded += len(chunk)
                        if total_downloaded > max_bytes:
                            raise ValueError(
                                f"Downloaded asset exceeded maximum permitted size ({max_bytes} bytes)"
                            )
                        f.write(chunk)

        logger.info("Successfully acquired SAR asset to %s (%d bytes)", target_path, total_downloaded)
        return target_path
