import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from app.core.config import Settings
from app.domain.sar_intelligence.models import SARSceneSummary

logger = logging.getLogger(__name__)


class CopernicusCatalogClient:
    """
    Dedicated client for Copernicus Data Space STAC catalog API.
    Endpoint: https://stac.dataspace.copernicus.eu/v1/search
    Collection: sentinel-1-grd
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.stac_url = self.settings.CDSE_STAC_URL.rstrip("/")
        if not self.stac_url.endswith("/search"):
            self.search_url = f"{self.stac_url}/search"
        else:
            self.search_url = self.stac_url

    async def search_scenes(
        self,
        bbox: tuple[float, float, float, float],
        start_time: datetime,
        end_time: datetime,
        limit: int = 20,
        modality: str = "GRD",
    ) -> list[SARSceneSummary]:
        """
        Search Sentinel-1 scenes over a bounding box and time window.
        Modality: 'GRD' (default: sentinel-1-grd) or 'SLC' (opt-in: sentinel-1-slc).
        No cloud cover constraints (SAR penetration through clouds).
        """
        mod = modality.upper()
        if mod == "SLC":
            target_collection = "sentinel-1-slc"
        else:
            target_collection = "sentinel-1-grd"

        # Format ISO UTC timestamps
        start_iso = start_time.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_time.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        datetime_range = f"{start_iso}/{end_iso}"

        payload = {
            "collections": [target_collection],
            "bbox": list(bbox),
            "datetime": datetime_range,
            "limit": limit,
        }

        logger.info(
            "Searching Copernicus STAC catalog at %s for collection=%s, bbox=%s, timerange=%s",
            self.search_url,
            target_collection,
            bbox,
            datetime_range,
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.search_url,
                    json=payload,
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )

                if response.status_code != 200:
                    logger.error(
                        "Copernicus STAC error %d: %s",
                        response.status_code,
                        response.text[:300],
                    )
                    return []

                data = response.json()
                features = data.get("features", [])
                logger.info("Retrieved %d Sentinel-1 %s scenes from Copernicus STAC", len(features), mod)

                scenes: list[SARSceneSummary] = []
                for feat in features:
                    scene_summary = self._parse_stac_feature(feat, modality=mod)
                    if scene_summary:
                        scenes.append(scene_summary)

                return scenes

        except httpx.RequestError as exc:
            logger.warning("Network error contacting Copernicus STAC: %s", exc)
            return []

    async def get_scene_by_id(self, scene_id: str, modality: str = "GRD") -> SARSceneSummary | None:
        """Fetch full STAC feature metadata for a specific Sentinel-1 scene ID."""
        mod = modality.upper()
        collection = "sentinel-1-slc" if mod == "SLC" else "sentinel-1-grd"
        payload = {
            "collections": [collection],
            "ids": [scene_id],
            "limit": 1,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.search_url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    features = data.get("features", [])
                    if features:
                        return self._parse_stac_feature(features[0], modality=mod)
        except Exception as e:
            logger.warning("Failed to fetch scene by ID %s: %s", scene_id, e)
        return None

    def _parse_stac_feature(self, feat: dict[str, Any], modality: str = "GRD") -> SARSceneSummary | None:
        try:
            scene_id = feat["id"]
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            assets = feat.get("assets", {})

            dt_str = props.get("datetime") or props.get("start_datetime")
            if not dt_str:
                return None
            
            # Parse datetime
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except ValueError:
                dt = datetime.now(UTC)

            platform = props.get("platform") or "Sentinel-1"
            orbit_pass = props.get("sat:orbit_state")
            orbit_number = props.get("sat:relative_orbit") or props.get("sat:absolute_orbit")
            polarizations = props.get("sar:polarizations")
            if isinstance(polarizations, list):
                polarization_str = "+".join(polarizations)
            else:
                polarization_str = "VV+VH"

            default_proc = "LEVEL1_SLC" if modality.upper() == "SLC" else "LEVEL1_GRD"
            processing_level = props.get("processing:level") or default_proc

            # Check thumbnail URL
            thumbnail_url = None
            if "thumbnail" in assets:
                thumbnail_url = assets["thumbnail"].get("href")

            # Check download asset URL (supports 'Product', 'product', or polarization alternates)
            download_url = None
            if "Product" in assets:
                download_url = assets["Product"].get("href")
            elif "product" in assets:
                download_url = assets["product"].get("href")
            elif "vv" in assets and "alternate" in assets["vv"] and "https" in assets["vv"]["alternate"]:
                download_url = assets["vv"]["alternate"]["https"].get("href")
            elif "iw1-vv" in assets and "alternate" in assets["iw1-vv"] and "https" in assets["iw1-vv"]["alternate"]:
                download_url = assets["iw1-vv"]["alternate"]["https"].get("href")

            # Preserve direct S3 measurement asset URIs for range/windowed extraction
            for k, v in assets.items():
                if not isinstance(v, dict):
                    continue
                href = v.get("href", "")
                if k in ("vv", "iw1-vv") and "s3_vv" not in props:
                    props["s3_vv"] = href
                elif k in ("vh", "iw1-vh") and "s3_vh" not in props:
                    props["s3_vh"] = href
                elif ("schema-product" in k or "manifest" in k) and "s3_xml" not in props:
                    props["s3_xml"] = href

            return SARSceneSummary(
                scene_id=scene_id,
                acquisition_time=dt,
                platform=platform,
                orbit_pass=orbit_pass,
                orbit_number=orbit_number,
                footprint=geom,
                polarization=polarization_str,
                processing_level=processing_level,
                modality=modality.upper(),
                source="Copernicus CDSE",
                quicklook_url=thumbnail_url,
                download_url=download_url,
                scene_metadata=props,
            )
        except Exception as exc:
            logger.warning("Failed to parse STAC feature %s: %s", feat.get("id"), exc)
            return None

    async def search_slc_bursts(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        burst_id: int | None = None,
        swath_id: str | None = None,
        polarization: str = "VV",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Query Copernicus OData /Bursts endpoint for individual SLC bursts.
        Allows targeted burst metadata discovery without loading full SAFE catalogs.
        """
        filters: list[str] = []
        if burst_id is not None:
            filters.append(f"BurstId eq {burst_id}")
        if swath_id:
            filters.append(f"SwathIdentifier eq '{swath_id}'")
        if polarization:
            filters.append(f"PolarisationChannels eq '{polarization}'")
        if start_time:
            s_iso = start_time.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            filters.append(f"ContentDate/Start gt {s_iso}")
        if end_time:
            e_iso = end_time.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            filters.append(f"ContentDate/Start lt {e_iso}")
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            polygon_wkt = (
                f"SRID=4326;POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
                f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
            )
            filters.append(f"OData.CSC.Intersects(area=geography'{polygon_wkt}')")

        filter_str = " and ".join(filters)
        params: dict[str, Any] = {"$top": limit}
        if filter_str:
            params["$filter"] = filter_str

        bursts_url = getattr(
            self.settings,
            "CDSE_ODATA_BURSTS_URL",
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts",
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.get(bursts_url, params=params)
                if res.status_code == 200:
                    return res.json().get("value", [])
                logger.warning("CDSE Bursts query returned %d: %s", res.status_code, res.text[:200])
                return []
        except Exception as exc:
            logger.error("Failed querying CDSE Bursts: %s", exc)
            return []
