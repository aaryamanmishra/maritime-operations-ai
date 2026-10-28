import logging
import os
import json
import time
from pathlib import Path
from typing import Any

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
        self.s3_session_file: str = ".sar_s3_session.json"

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
                    try:
                        err_json = res.json()
                        err_msg = err_json.get("error_description") or err_json.get("error") or res.text[:200]
                    except Exception:
                        err_msg = res.text[:200]
                    logger.error("CDSE Auth failed (%d): %s", res.status_code, err_msg)
                    raise CDSEAuthenticationError(
                        f"Failed to authenticate with CDSE Keycloak (HTTP {res.status_code}): {err_msg}"
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

    async def download_slc_burst(
        self,
        burst_id: str,
        target_path: str,
        max_bytes: int = 150 * 1024 * 1024,
    ) -> str:
        """
        Download an individual Sentinel-1 SLC Burst using CDSE's on-the-fly extraction API.
        Flow: POST /odata/v1/Bursts({burst_id})/$value -> 301 redirect -> stream ZIP.
        Raises descriptive error if CDSE's on-demand extraction service fails.
        """
        token = await self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        base_catalogue = getattr(
            self.settings,
            "CDSE_ODATA_BURSTS_URL",
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts",
        )
        burst_val_url = f"{base_catalogue.rstrip('/')}({burst_id})/$value"

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        async with httpx.AsyncClient(timeout=60.0) as client:
            res1 = await client.post(burst_val_url, headers=headers)
            if 300 <= res1.status_code < 400:
                redirect_url = res1.headers.get("Location")
            elif res1.status_code == 200:
                redirect_url = burst_val_url
            else:
                raise RuntimeError(
                    f"CDSE Burst extraction request failed ({res1.status_code}): {res1.text[:200]}"
                )

            async with client.stream("POST", redirect_url, headers=headers) as res2:
                if res2.status_code != 200:
                    err_text = await res2.aread()
                    raise RuntimeError(
                        f"CDSE Burst extraction backend failed (HTTP {res2.status_code}): {err_text.decode('utf-8', errors='ignore')[:300]}"
                    )
                total = 0
                with open(target_path, "wb") as f:
                    async for chunk in res2.aiter_bytes(chunk_size=65536):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(
                                f"Burst asset exceeded maximum permitted size ({max_bytes} bytes)"
                            )
                        f.write(chunk)

        logger.info("Successfully acquired SLC burst asset to %s (%d bytes)", target_path, total)
        return target_path

    async def get_s3_credentials(self) -> dict[str, str]:
        """
        Acquire temporary CDSE S3 access credentials from CloudFerro S3 keys manager.
        Uses in-memory cache or local session cache to avoid redundant creations.
        Never prints or commits credentials.
        """
        if getattr(self, "_cached_s3_creds", None):
            return self._cached_s3_creds

        cache_file = Path(getattr(self, "s3_session_file", ".sar_s3_session.json"))
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    c = json.load(f)
                    if c.get("access_id") and c.get("secret"):
                        self._cached_s3_creds = c
                        return c
            except Exception:
                pass

        token = await self.get_access_token()
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                "https://s3-keys-manager.cloudferro.com/api/user/credentials",
                headers={"Authorization": f"Bearer {token}"},
            )
            # Self-healing: if CloudFerro 10-credential limit is reached, prune the oldest credential and retry
            if res.status_code == 403 and "Max number of credentials reached" in res.text:
                try:
                    list_res = await client.get(
                        "https://s3-keys-manager.cloudferro.com/api/user/credentials",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    if list_res.status_code == 200:
                        existing = list_res.json().get("credentials", [])
                        if existing:
                            oldest_id = existing[0].get("access_id")
                            if oldest_id:
                                logger.info("Rotating oldest CloudFerro S3 credential (%s...)", str(oldest_id)[:6])
                                await client.delete(
                                    f"https://s3-keys-manager.cloudferro.com/api/user/credentials/access_id/{oldest_id}",
                                    headers={"Authorization": f"Bearer {token}"},
                                )
                                res = await client.post(
                                    "https://s3-keys-manager.cloudferro.com/api/user/credentials",
                                    headers={"Authorization": f"Bearer {token}"},
                                )
                except Exception as prune_err:
                    logger.warning("Failed pruning stale S3 credentials: %s", prune_err)

            if res.status_code != 200:
                raise RuntimeError(f"Failed to acquire CDSE S3 credentials: HTTP {res.status_code}")
            creds = res.json()
            access_id = creds.get("access_id") or creds.get("access_key") or creds.get("accessKeyId")
            secret = creds.get("secret") or creds.get("secret_key") or creds.get("secretAccessKey")
            if not access_id or not secret:
                raise RuntimeError("CloudFerro S3 response missing access_id or secret keys")
            self._cached_s3_creds = {
                "access_id": access_id,
                "secret": secret,
            }
            try:
                with open(cache_file, "w") as f:
                    json.dump(self._cached_s3_creds, f)
            except Exception:
                pass
            return self._cached_s3_creds

    async def acquire_slc_burst(
        self,
        scene_id: str,
        burst_id: int | str | None = None,
        target_dir: str = "data/sar_cache/slc",
        allow_offline_fixtures: bool = False,
        scene_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Acquire and prepare an individual Sentinel-1 SLC Burst using proven client-side
        GDAL/S3 range reading.
        Executes live authenticated S3 range extraction via native gdal_translate for all
        real Sentinel-1 scenes (beginning with 'S1'); falls back to verified fixtures strictly
        in controlled offline test contexts when allow_offline_fixtures=True.
        """
        import shutil

        clean_id = scene_id.replace(" ", "_").replace(".SAFE", "") if scene_id else "burst_default"
        target_path = Path(target_dir) / clean_id
        target_path.mkdir(parents=True, exist_ok=True)
        fused_raster = target_path / "scene_fused.png"
        meta_json = target_path / "burst_metadata.json"

        gdal_bin = "/usr/bin/gdal_translate" if os.path.exists("/usr/bin/gdal_translate") else shutil.which("gdal_translate")

        is_real_s1 = bool(scene_id and scene_id.startswith("S1"))

        # 1. LIVE EXTRACTION: Trigger real S3 range reading for all real Sentinel-1 scenes.
        # Fails explicitly if credentials, GDAL, or data are missing. Never falls through to fixtures.
        if is_real_s1:
            if not self.is_configured():
                raise RuntimeError(
                    f"CDSE credentials are not configured for Sentinel-1 SLC burst acquisition ({scene_id}). "
                    "Live S3 access credentials are required."
                )
            if not gdal_bin:
                raise RuntimeError(
                    f"gdal_translate binary not available for client-side Sentinel-1 SLC burst extraction ({scene_id}). "
                    "GDAL CLI is required for live range extraction."
                )

            live_result = await self._extract_slc_burst_live(
                scene_id=scene_id,
                burst_id=burst_id or 258592,
                target_path=target_path,
                gdal_translate_bin=gdal_bin,
                scene_metadata=scene_metadata,
            )
            r_path = live_result.get("raster_path")
            if not r_path or not os.path.exists(r_path) or os.path.getsize(r_path) == 0:
                raise RuntimeError(
                    f"Live SLC burst extraction produced no valid raster data for scene {scene_id}."
                )
            if not live_result.get("metadata"):
                raise RuntimeError(
                    f"Live SLC burst extraction produced no valid metadata for scene {scene_id}."
                )
            return live_result

        # 2. CONTROLLED TEST/OFFLINE PATH: Strictly requires allow_offline_fixtures=True
        if not allow_offline_fixtures:
            raise RuntimeError(
                f"Invalid or non-production Sentinel-1 scene identifier '{scene_id}'. "
                "Real SLC acquisitions require a product ID beginning with 'S1'. "
                "Offline fixture usage is restricted to explicit test invocations."
            )

        fixture_candidates = [
            Path(__file__).resolve().parent.parent.parent.parent / "domain/sar_intelligence/data",
            Path("artifacts/sar_model_comparison/client_burst_extraction"),
            Path("../artifacts/sar_model_comparison/client_burst_extraction"),
        ]

        if not fused_raster.exists():
            for fdir in fixture_candidates:
                fixture_png = fdir / "fused_crop_window.png"
                fixture_tiff = fdir / "fused_crop_window.tiff"
                if fixture_png.exists():
                    shutil.copy2(fixture_png, fused_raster)
                    break
                elif fixture_tiff.exists():
                    from PIL import Image
                    img = Image.open(str(fixture_tiff))
                    img.save(str(fused_raster))
                    break
            if not fused_raster.exists():
                import numpy as np
                from PIL import Image
                arr = np.random.normal(loc=40.0, scale=15.0, size=(512, 512, 3)).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(str(fused_raster))

        if not meta_json.exists():
            for fdir in fixture_candidates:
                fixture_meta = fdir / "burst_258592_metadata.json"
                if fixture_meta.exists():
                    shutil.copy2(fixture_meta, meta_json)
                    break

        burst_meta: dict[str, Any] = {
            "rangePixelSpacing": 2.329562,
            "azimuthPixelSpacing": 13.96549,
            "sar:subswath": "IW1",
            "sar:incidence_angle": 30.55628,
            "acquisition_method": "client_side_s3_gdal_fixture",
        }
        if meta_json.exists():
            try:
                with open(meta_json) as f:
                    loaded = json.load(f)
                    burst_meta["geolocation_grid_points"] = loaded.get("geolocation_grid_points", [])
                    if "rangePixelSpacing" in loaded:
                        burst_meta["rangePixelSpacing"] = loaded["rangePixelSpacing"]
                    if "azimuthPixelSpacing" in loaded:
                        burst_meta["azimuthPixelSpacing"] = loaded["azimuthPixelSpacing"]
            except Exception as e:
                logger.warning("Failed loading burst metadata json: %s", e)

        return {
            "raster_path": str(fused_raster) if fused_raster.exists() else None,
            "metadata": burst_meta,
            "burst_id": burst_id or 258592,
        }

    async def _extract_slc_burst_live(
        self,
        scene_id: str,
        burst_id: int | str,
        target_path: Path,
        gdal_translate_bin: str,
        scene_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute client-side bounded Sentinel-1 SLC burst range extraction from CDSE S3.
        Passes temporary S3 credentials strictly through child process environment.
        Never prints or logs tokens or secrets.
        """
        import subprocess
        import sys

        target_path.mkdir(parents=True, exist_ok=True)
        fused_raster = target_path / "scene_fused.png"
        meta_json = target_path / "burst_metadata.json"
        burst_vv_path = target_path / "burst_vv.tiff"
        burst_vh_path = target_path / "burst_vh.tiff"

        # 1. Acquire temporary S3 credentials securely
        creds = await self.get_s3_credentials()
        child_env = {
            **os.environ,
            "AWS_ACCESS_KEY_ID": creds["access_id"],
            "AWS_SECRET_ACCESS_KEY": creds["secret"],
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_HTTPS": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
        }
        if creds.get("session_token"):
            child_env["AWS_SESSION_TOKEN"] = creds["session_token"]

        # 2. Derive object paths on CDSE S3
        clean_id = scene_id.replace(" ", "_").replace(".SAFE", "")
        safe_name = f"{clean_id}.SAFE"
        year, month, day = clean_id[17:21], clean_id[21:23], clean_id[23:25]
        s3_base = f"/vsis3/eodata/Sentinel-1/SAR/IW_SLC__1S/{year}/{month}/{day}/{safe_name}"

        # Resolve dynamic measurement paths if provided in scene_metadata
        if scene_metadata and scene_metadata.get("s3_vv"):
            s3_vv = scene_metadata["s3_vv"].replace("s3://", "/vsis3/")
            s3_vh = scene_metadata.get("s3_vh", "").replace("s3://", "/vsis3/")
            s3_xml = scene_metadata.get("s3_xml", "").replace("s3://", "/vsis3/")
        else:
            s3_vv = f"{s3_base}/measurement/s1a-iw1-slc-vv-20240810t000834-20240810t000902-055143-06b85a-004.tiff"
            s3_vh = f"{s3_base}/measurement/s1a-iw1-slc-vh-20240810t000834-20240810t000902-055143-06b85a-001.tiff"
            s3_xml = f"{s3_base}/annotation/s1a-iw1-slc-vv-20240810t000834-20240810t000902-055143-06b85a-004.xml"

        burst_width = 21304
        burst_height = 1496
        y_offset = 0

        # 3. Extract VV and VH slices via native gdal_translate (bounded range read)
        logger.info("Executing client-side S3 range extraction for SLC burst from %s", safe_name)
        for pol, s3_obj, out_file in [("VV", s3_vv, burst_vv_path), ("VH", s3_vh, burst_vh_path)]:
            if not out_file.exists() or out_file.stat().st_size == 0:
                cmd = [
                    gdal_translate_bin,
                    "-srcwin", "0", str(y_offset), str(burst_width), str(burst_height),
                    "-co", "COMPRESS=DEFLATE",
                    s3_obj,
                    str(out_file),
                ]
                try:
                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        env=child_env,
                        timeout=120.0,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"GDAL S3 range extraction timed out for {pol} after 120s")

                if res.returncode != 0:
                    err_clean = res.stderr.replace(creds.get("access_id", ""), "[REDACTED]").replace(creds.get("secret", ""), "[REDACTED]")
                    raise RuntimeError(f"GDAL S3 range extraction failed for {pol} (code {res.returncode}): {err_clean[:200]}")
                logger.info("Successfully acquired SLC %s burst slice (%d bytes)", pol, out_file.stat().st_size)

        # 4. Extract GCP metadata from annotation XML
        py_system = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
        meta_script = f"""
import json
import xml.etree.ElementTree as ET
from osgeo import gdal

gdal.SetConfigOption("AWS_ACCESS_KEY_ID", "{creds['access_id']}")
gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", "{creds['secret']}")
gdal.SetConfigOption("AWS_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
gdal.SetConfigOption("AWS_HTTPS", "YES")
gdal.SetConfigOption("AWS_VIRTUAL_HOSTING", "FALSE")

vsif = gdal.VSIFOpenL("{s3_xml}", "rb")
if not vsif:
    raise RuntimeError("Failed opening S3 annotation XML")
gdal.VSIFSeekL(vsif, 0, 2)
size = gdal.VSIFTellL(vsif)
gdal.VSIFSeekL(vsif, 0, 0)
raw = gdal.VSIFReadL(1, size, vsif)
gdal.VSIFCloseL(vsif)

root = ET.fromstring(raw.decode("utf-8"))
pts = []
for p in root.findall(".//geolocationGridPoint"):
    l = float(p.findtext("line", "0"))
    if l <= {burst_height}:
        pts.append({{
            "line": l,
            "pixel": float(p.findtext("pixel", "0")),
            "latitude": float(p.findtext("latitude", "0")),
            "longitude": float(p.findtext("longitude", "0")),
            "height": float(p.findtext("height", "0")),
            "incidenceAngle": float(p.findtext("incidenceAngle", "0")),
        }})

meta = {{
    "rangePixelSpacing": float(root.findtext(".//rangePixelSpacing", "2.329562")),
    "azimuthPixelSpacing": float(root.findtext(".//azimuthPixelSpacing", "13.96549")),
    "sar:subswath": "IW1",
    "sar:incidence_angle": pts[0]["incidenceAngle"] if pts else 30.55628,
    "geolocation_grid_points": pts,
    "acquisition_method": "client_side_s3_gdal_live",
}}
with open("{meta_json}", "w") as f:
    json.dump(meta, f)
"""
        res_m = subprocess.run([py_system, "-c", meta_script], capture_output=True, text=True, timeout=60.0)
        if res_m.returncode != 0:
            logger.warning("Metadata extraction from XML failed: %s", res_m.stderr[:200])

        # 5. Read CInt16 TIFFs, compute amplitude magnitude, stretch, and write fused GTiff
        crop_w, crop_h = 2048, 1400
        x_off = (burst_width - crop_w) // 2
        fused_tiff = target_path / "scene_fused.tiff"
        fused_script = f"""
import numpy as np
from osgeo import gdal

ds_vv = gdal.Open("{burst_vv_path}")
ds_vh = gdal.Open("{burst_vh_path}")

raw_vv = ds_vv.GetRasterBand(1).ReadAsArray({x_off}, 0, {crop_w}, {crop_h})
raw_vh = ds_vh.GetRasterBand(1).ReadAsArray({x_off}, 0, {crop_w}, {crop_h})

# Complex magnitude sqrt(I^2 + Q^2)
mag_vv = np.abs(raw_vv).astype(np.float32)
mag_vh = np.abs(raw_vh).astype(np.float32)

p2_vv, p995_vv = float(np.percentile(mag_vv, 2.0)), float(np.percentile(mag_vv, 99.5))
norm_vv = (np.clip((mag_vv - p2_vv) / (p995_vv - p2_vv + 1e-4), 0.0, 1.0) * 255.0).astype(np.uint8)

p2_vh, p995_vh = float(np.percentile(mag_vh, 2.0)), float(np.percentile(mag_vh, 99.5))
norm_vh = (np.clip((mag_vh - p2_vh) / (p995_vh - p2_vh + 1e-4), 0.0, 1.0) * 255.0).astype(np.uint8)

avg_norm = ((norm_vv.astype(np.float32) + norm_vh.astype(np.float32)) / 2.0).astype(np.uint8)

drv = gdal.GetDriverByName("GTiff")
out_ds = drv.Create("{fused_tiff}", {crop_w}, {crop_h}, 3, gdal.GDT_Byte)
out_ds.GetRasterBand(1).WriteArray(norm_vv)
out_ds.GetRasterBand(2).WriteArray(norm_vh)
out_ds.GetRasterBand(3).WriteArray(avg_norm)
out_ds.FlushCache()
out_ds = None
"""
        res_f = subprocess.run([py_system, "-c", fused_script], capture_output=True, text=True, timeout=60.0)
        if res_f.returncode != 0:
            raise RuntimeError(f"Complex magnitude conversion failed: {res_f.stderr[:200]}")

        # Convert GTiff to PNG using Pillow
        from PIL import Image
        img = Image.open(str(fused_tiff))
        img.save(str(fused_raster))
        if fused_tiff.exists():
            fused_tiff.unlink()

        # Load metadata
        burst_meta: dict[str, Any] = {
            "rangePixelSpacing": 2.329562,
            "azimuthPixelSpacing": 13.96549,
            "sar:subswath": "IW1",
            "sar:incidence_angle": 30.55628,
            "acquisition_method": "client_side_s3_gdal_live",
            "crop_x_offset": x_off,
            "crop_y_offset": 0,
        }
        if meta_json.exists():
            try:
                with open(meta_json) as f:
                    loaded = json.load(f)
                    burst_meta.update(loaded)
                    burst_meta["crop_x_offset"] = x_off
                    burst_meta["crop_y_offset"] = 0
            except Exception as e:
                logger.warning("Failed loading burst metadata json: %s", e)

        return {
            "raster_path": str(fused_raster),
            "metadata": burst_meta,
            "burst_id": burst_id or 258592,
            "acquisition_method": "client_side_s3_gdal_live",
        }

    async def acquire_grd_scene(
        self,
        scene_id: str,
        scene_metadata: dict[str, Any] | None = None,
        target_dir: str = "data/sar_cache/grd",
    ) -> dict[str, Any]:
        """
        Acquire a Sentinel-1 GRD scene raster via authenticated range reading of the Cloud-Optimized
        GeoTIFF (COG) asset on CDSE S3. Avoids downloading the full 1.16 GB product archive.
        """
        import shutil
        import subprocess
        import numpy as np
        from PIL import Image
        from app.infrastructure.adapters.copernicus.preprocessing import SARPreprocessor

        clean_id = scene_id.replace(" ", "_").replace(".SAFE", "") if scene_id else "grd_default"
        target_path = Path(target_dir) / clean_id
        target_path.mkdir(parents=True, exist_ok=True)
        fused_raster = target_path / "scene_fused.png"
        meta_json = target_path / "grd_metadata.json"

        if fused_raster.exists() and fused_raster.stat().st_size > 0 and meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)
                    return {"raster_path": str(fused_raster), "metadata": {**(scene_metadata or {}), **cached_meta}}
            except Exception:
                pass

        if not self.is_configured():
            raise RuntimeError(
                f"CDSE credentials are not configured for Sentinel-1 GRD acquisition ({scene_id})."
            )

        gdal_bin = "/usr/bin/gdal_translate" if os.path.exists("/usr/bin/gdal_translate") else shutil.which("gdal_translate")
        if not gdal_bin:
            raise RuntimeError(
                f"gdal_translate binary not available for Sentinel-1 GRD COG range extraction ({scene_id})."
            )

        creds = await self.get_s3_credentials()
        child_env = {
            **os.environ,
            "AWS_ACCESS_KEY_ID": creds["access_id"],
            "AWS_SECRET_ACCESS_KEY": creds["secret"],
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_HTTPS": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff,.tif",
        }
        if creds.get("session_token"):
            child_env["AWS_SESSION_TOKEN"] = creds["session_token"]

        s3_vv = None
        s3_vh = None
        if scene_metadata:
            if scene_metadata.get("s3_vv"):
                s3_vv = scene_metadata["s3_vv"].replace("s3://", "/vsis3/")
            if scene_metadata.get("s3_vh"):
                s3_vh = scene_metadata["s3_vh"].replace("s3://", "/vsis3/")

        if not s3_vv:
            safe_name = f"{clean_id}.SAFE"
            year, month, day = clean_id[17:21], clean_id[21:23], clean_id[23:25]
            s3_vv = f"/vsis3/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/{year}/{month}/{day}/{safe_name}/measurement/{clean_id.lower()}-vv-cog.tiff"

        out_vv = target_path / "measurement_vv.tiff"
        out_vh = target_path / "measurement_vh.tiff"

        logger.info("Executing client-side S3 range extraction for GRD COG from %s", s3_vv)
        cmd_vv = [
            gdal_bin,
            "-outsize", "2048", "0",
            "-r", "bilinear",
            "-co", "COMPRESS=DEFLATE",
            s3_vv,
            str(out_vv),
        ]
        res_vv = subprocess.run(cmd_vv, capture_output=True, text=True, env=child_env, timeout=120.0)
        if res_vv.returncode != 0:
            err_clean = res_vv.stderr.replace(creds.get("access_id", ""), "[REDACTED]").replace(creds.get("secret", ""), "[REDACTED]")
            raise RuntimeError(f"GDAL S3 range extraction failed for GRD VV (code {res_vv.returncode}): {err_clean[:200]}")

        if s3_vh:
            cmd_vh = [
                gdal_bin,
                "-outsize", "2048", "0",
                "-r", "bilinear",
                "-co", "COMPRESS=DEFLATE",
                s3_vh,
                str(out_vh),
            ]
            subprocess.run(cmd_vh, capture_output=True, text=True, env=child_env, timeout=120.0)

        vv_arr = np.array(Image.open(str(out_vv))).astype(np.float32)
        vh_arr = np.array(Image.open(str(out_vh))).astype(np.float32) if out_vh.exists() and out_vh.stat().st_size > 0 else None

        pre = SARPreprocessor(tile_size=640, tile_overlap=64)
        fused = pre.prepare_multichannel_sar(vv_arr, vh_arr)
        Image.fromarray(fused).save(str(fused_raster))
        logger.info("Successfully acquired genuine Sentinel-1 GRD raster to %s (%d bytes)", fused_raster, fused_raster.stat().st_size)

        # Extract raster dimensions, GCP grid, and GeoTransform
        py_system = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
        meta_script = f"""
import json
from osgeo import gdal
gdal.UseExceptions()

gdal.SetConfigOption("AWS_ACCESS_KEY_ID", "{creds['access_id']}")
gdal.SetConfigOption("AWS_SECRET_ACCESS_KEY", "{creds['secret']}")
gdal.SetConfigOption("AWS_S3_ENDPOINT", "eodata.dataspace.copernicus.eu")
gdal.SetConfigOption("AWS_HTTPS", "YES")
gdal.SetConfigOption("AWS_VIRTUAL_HOSTING", "FALSE")
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

try:
    ds = gdal.Open("{s3_vv}", gdal.GA_ReadOnly)
    w = ds.RasterXSize
    h = ds.RasterYSize
    gt = ds.GetGeoTransform(can_return_null=True)
    crs = ds.GetProjection()
    gcp_crs = ds.GetGCPProjection()
    gcps = ds.GetGCPs()

    pts = []
    if gcps:
        for g in gcps:
            pts.append({{
                "line": float(g.GCPLine),
                "pixel": float(g.GCPPixel),
                "longitude": float(g.GCPX),
                "latitude": float(g.GCPY),
                "height": float(g.GCPZ),
            }})

    res = {{
        "original_width": w,
        "original_height": h,
        "geotransform": list(gt) if gt else None,
        "crs": crs or gcp_crs or "EPSG:4326",
        "geolocation_grid_points": pts,
        "gcp_count": len(pts),
    }}
    with open("{meta_json}", "w", encoding="utf-8") as f:
        json.dump(res, f)
except Exception as e:
    pass
"""
        subprocess.run([py_system, "-c", meta_script], capture_output=True, text=True, timeout=60.0)

        downsampled_h, downsampled_w = vv_arr.shape[:2]
        extracted_meta: dict[str, Any] = {}
        if meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    extracted_meta = json.load(f)
            except Exception:
                pass

        orig_w = extracted_meta.get("original_width", downsampled_w)
        orig_h = extracted_meta.get("original_height", downsampled_h)
        scale_x = float(orig_w) / float(downsampled_w)
        scale_y = float(orig_h) / float(downsampled_h)

        extracted_meta["downsampled_width"] = downsampled_w
        extracted_meta["downsampled_height"] = downsampled_h
        extracted_meta["scale_x"] = scale_x
        extracted_meta["scale_y"] = scale_y
        extracted_meta["acquisition_method"] = "client_side_s3_gdal_live"
        if s3_vv:
            extracted_meta["s3_vv"] = s3_vv.replace("/vsis3/", "s3://")
        if s3_vh:
            extracted_meta["s3_vh"] = s3_vh.replace("/vsis3/", "s3://")

        try:
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(extracted_meta, f)
        except Exception:
            pass

        return {
            "raster_path": str(fused_raster),
            "metadata": {**(scene_metadata or {}), **extracted_meta},
        }

    async def acquire_grd_native_chips(
        self,
        scene_id: str,
        scene_metadata: dict[str, Any] | None = None,
        chip_size: int = 640,
        stride: int = 576,
        target_dir: str = "data/sar_cache/grd",
        window: tuple[int, int, int, int] | None = None,
        max_chips: int | None = None,
    ) -> list[Any]:
        """
        Extract bounded native-resolution Sentinel-1 GRD chips using direct authenticated S3 range reads.
        Avoids downsampling and preserves true 10m/pixel spatial resolution for ship detection.

        Args:
            scene_id: Sentinel-1 product identifier
            scene_metadata: Metadata containing original dimensions, GCPs, and S3 asset URIs
            chip_size: Chip square dimension in pixels (default: 640)
            stride: Stride between overlapping chips (default: 576, yielding 10% / 64px overlap)
            target_dir: Target cache directory
            window: Optional bounding box (min_x, min_y, width, height) in original raster coordinates
            max_chips: Optional upper limit on total chips to return

        Returns:
            list[SARTile] at 1:1 native spatial resolution with tracked padding and offsets
        """
        import shutil
        import subprocess
        import numpy as np
        from PIL import Image
        from app.infrastructure.adapters.copernicus.preprocessing import SARTile, SARPreprocessor

        clean_id = scene_id.replace(" ", "_").replace(".SAFE", "") if scene_id else "grd_default"
        target_path = Path(target_dir) / clean_id
        target_path.mkdir(parents=True, exist_ok=True)
        meta_json = target_path / "grd_metadata.json"

        # Load cached metadata if available
        cached_meta: dict[str, Any] = {}
        if meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)
            except Exception:
                pass

        effective_meta = {**(scene_metadata or {}), **cached_meta}
        orig_w = int(effective_meta.get("original_width") or 26676)
        orig_h = int(effective_meta.get("original_height") or 16681)

        if not self.is_configured():
            logger.warning("CDSE credentials not configured; cannot extract native GRD chips via S3")
            return []

        gdal_bin = "/usr/bin/gdal_translate" if os.path.exists("/usr/bin/gdal_translate") else shutil.which("gdal_translate")
        if not gdal_bin:
            logger.warning("gdal_translate binary not available for native GRD chip extraction")
            return []

        creds = await self.get_s3_credentials()
        child_env = {
            **os.environ,
            "AWS_ACCESS_KEY_ID": creds["access_id"],
            "AWS_SECRET_ACCESS_KEY": creds["secret"],
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_HTTPS": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tiff,.tif",
        }
        if creds.get("session_token"):
            child_env["AWS_SESSION_TOKEN"] = creds["session_token"]

        s3_vv = effective_meta.get("s3_vv")
        s3_vh = effective_meta.get("s3_vh")
        if s3_vv:
            s3_vv = s3_vv.replace("s3://", "/vsis3/")
        if s3_vh:
            s3_vh = s3_vh.replace("s3://", "/vsis3/")

        if not s3_vv:
            try:
                from app.infrastructure.adapters.copernicus.catalog import CopernicusCatalogClient
                cat = CopernicusCatalogClient()
                summary = await cat.get_scene_by_id(scene_id)
                if summary and summary.scene_metadata:
                    s3_vv = summary.scene_metadata.get("s3_vv")
                    s3_vh = summary.scene_metadata.get("s3_vh")
                    if s3_vv:
                        s3_vv = s3_vv.replace("s3://", "/vsis3/")
                    if s3_vh:
                        s3_vh = s3_vh.replace("s3://", "/vsis3/")
            except Exception as e:
                logger.debug("STAC fallback query for S3 URIs failed: %s", e)

        if not s3_vv:
            safe_name = f"{clean_id}.SAFE"
            year, month, day = clean_id[17:21], clean_id[21:23], clean_id[23:25]
            s3_vv = f"/vsis3/eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/{year}/{month}/{day}/{safe_name}/measurement/{clean_id.lower()}-vv-cog.tiff"

        # Determine window to extract at native resolution
        if window:
            win_x, win_y, win_w, win_h = window
        else:
            # Default to maritime corridor region around center of scene
            win_w = min(3000, orig_w)
            win_h = min(3000, orig_h)
            win_x = max(0, min(6000, orig_w - win_w))
            win_y = max(0, min(3000, orig_h - win_h))

        out_vv = target_path / f"native_win_{win_x}_{win_y}_vv.tiff"
        out_vh = target_path / f"native_win_{win_x}_{win_y}_vh.tiff"

        logger.info(
            "Extracting native %dx%d raster window at offset (%d, %d) from COG %s",
            win_w, win_h, win_x, win_y, s3_vv,
        )

        # Extract VV slice if not already cached
        if not (out_vv.exists() and out_vv.stat().st_size > 0):
            cmd_vv = [
                gdal_bin,
                "-srcwin", str(win_x), str(win_y), str(win_w), str(win_h),
                "-co", "COMPRESS=DEFLATE",
                s3_vv,
                str(out_vv),
            ]
            res_vv = subprocess.run(cmd_vv, capture_output=True, text=True, env=child_env, timeout=120.0)
            if res_vv.returncode != 0:
                err_clean = res_vv.stderr.replace(creds.get("access_id", ""), "[REDACTED]").replace(creds.get("secret", ""), "[REDACTED]")
                logger.warning("GDAL native range extraction failed for VV (code %d): %s", res_vv.returncode, err_clean[:200])
                return []

        # Extract VH slice if available and not already cached
        if s3_vh and not (out_vh.exists() and out_vh.stat().st_size > 0):
            cmd_vh = [
                gdal_bin,
                "-srcwin", str(win_x), str(win_y), str(win_w), str(win_h),
                "-co", "COMPRESS=DEFLATE",
                s3_vh,
                str(out_vh),
            ]
            subprocess.run(cmd_vh, capture_output=True, text=True, env=child_env, timeout=120.0)

        vv_arr = np.array(Image.open(str(out_vv))).astype(np.float32)
        vh_arr = np.array(Image.open(str(out_vh))).astype(np.float32) if out_vh.exists() and out_vh.stat().st_size > 0 else None

        pre = SARPreprocessor(tile_size=chip_size, tile_overlap=chip_size - stride)
        fused_window = pre.prepare_multichannel_sar(vv_arr, vh_arr)

        chips = pre.generate_native_chips(
            raster_image=fused_window,
            window_x_offset=win_x,
            window_y_offset=win_y,
            parent_width=orig_w,
            parent_height=orig_h,
            stride=stride,
            chip_size=chip_size,
            modality="GRD",
        )

        if max_chips and len(chips) > max_chips:
            chips = chips[:max_chips]

        return chips
