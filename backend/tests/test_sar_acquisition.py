import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.config import Settings
from app.infrastructure.adapters.copernicus.acquisition import (
    CopernicusAcquisitionClient,
    CDSEAuthenticationError,
    CDSECredentialsMissingError,
)


def test_copernicus_is_configured_logic():
    # Empty
    s_empty = Settings(CDSE_USERNAME="", CDSE_PASSWORD="", CDSE_CLIENT_ID="", CDSE_CLIENT_SECRET="")
    client_empty = CopernicusAcquisitionClient(settings=s_empty)
    assert not client_empty.is_configured()

    # Partial user/password
    s_partial = Settings(CDSE_USERNAME="user@test.com", CDSE_PASSWORD="", CDSE_CLIENT_ID="", CDSE_CLIENT_SECRET="")
    client_partial = CopernicusAcquisitionClient(settings=s_partial)
    assert not client_partial.is_configured()

    # Full username/password
    s_user = Settings(CDSE_USERNAME="user@test.com", CDSE_PASSWORD="pass123", CDSE_CLIENT_ID="", CDSE_CLIENT_SECRET="")
    client_user = CopernicusAcquisitionClient(settings=s_user)
    assert client_user.is_configured()

    # Full client id/secret
    s_client = Settings(CDSE_USERNAME="", CDSE_PASSWORD="", CDSE_CLIENT_ID="cid-123", CDSE_CLIENT_SECRET="csec-456")
    client_creds = CopernicusAcquisitionClient(settings=s_client)
    assert client_creds.is_configured()


@pytest.mark.asyncio
async def test_copernicus_credentials_missing_error():
    s_empty = Settings(CDSE_USERNAME="", CDSE_PASSWORD="", CDSE_CLIENT_ID="", CDSE_CLIENT_SECRET="")
    client = CopernicusAcquisitionClient(settings=s_empty)
    with pytest.raises(CDSECredentialsMissingError):
        await client.get_access_token()


@pytest.mark.asyncio
async def test_copernicus_auth_error_sanitization():
    s = Settings(CDSE_USERNAME="user@test.com", CDSE_PASSWORD="bad_password")
    client = CopernicusAcquisitionClient(settings=s)

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {
        "error": "invalid_grant",
        "error_description": "Invalid user credentials",
    }
    mock_resp.text = '{"error": "invalid_grant", "error_description": "Invalid user credentials"}'

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(CDSEAuthenticationError) as exc_info:
            await client.get_access_token()
        assert "Invalid user credentials" in str(exc_info.value)
        assert "400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_copernicus_s3_credentials_rotation_on_403(tmp_path):
    s = Settings(CDSE_USERNAME="user@test.com", CDSE_PASSWORD="good_password")
    client = CopernicusAcquisitionClient(settings=s)
    client.s3_session_file = str(tmp_path / ".sar_s3_session.json")

    # Mock get_access_token
    client.get_access_token = AsyncMock(return_value="mock_jwt_token")

    # Setup httpx mock responses for S3 keys manager:
    # 1. POST -> 403 Max number of credentials reached
    # 2. GET -> 200 with existing credentials list
    # 3. DELETE -> 204
    # 4. POST -> 200 with new credentials
    resp_403 = MagicMock()
    resp_403.status_code = 403
    resp_403.text = '{"detail": "Max number of credentials reached."}'

    resp_list = MagicMock()
    resp_list.status_code = 200
    resp_list.json.return_value = {
        "credentials": [
            {"access_id": "old_key_1", "description": "Key 1"},
            {"access_id": "old_key_2", "description": "Key 2"},
        ]
    }

    resp_delete = MagicMock()
    resp_delete.status_code = 204

    resp_success = MagicMock()
    resp_success.status_code = 200
    resp_success.json.return_value = {
        "access_id": "new_key_3",
        "secret": "SECRET_NEW",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get, \
         patch("httpx.AsyncClient.delete", new_callable=AsyncMock) as mock_del:

        mock_post.side_effect = [resp_403, resp_success]
        mock_get.return_value = resp_list
        mock_del.return_value = resp_delete

        creds = await client.get_s3_credentials()

        assert creds["access_id"] == "new_key_3"
        assert creds["secret"] == "SECRET_NEW"
        mock_del.assert_called_once()
        assert "old_key_1" in mock_del.call_args[0][0]
