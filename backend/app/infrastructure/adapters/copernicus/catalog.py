import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
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

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.stac_url = self.settings.CDSE_STAC_URL.rstrip("/")
        if not self.stac_url.endswith("/search"):
            self.search_url = f"{self.stac_url}/search"
        else:
            self.search_url = self.stac_url

    async def search_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        start_time: datetime,
        end_time: datetime,
        limit: int = 20,
    ) -> List[SARSceneSummary]:
        """
        Search Sentinel-1 GRD scenes over a bounding box and time window.
        No cloud cover constraints (SAR penetration through clouds).
        """
        # Format ISO UTC timestamps
        start_iso = start_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = end_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        datetime_range = f"{start_iso}/{end_iso}"

        payload = {
            "collections": ["sentinel-1-grd"],
            "bbox": list(bbox),
            "datetime": datetime_range,
            "limit": limit,
        }

        logger.info(
            "Searching Copernicus STAC catalog at %s for bbox=%s, timerange=%s",
            self.search_url,
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
                logger.info("Retrieved %d Sentinel-1 scenes from Copernicus STAC", len(features))

                scenes: List[SARSceneSummary] = []
                for feat in features:
                    scene_summary = self._parse_stac_feature(feat)
                    if scene_summary:
                        scenes.append(scene_summary)

                return scenes

        except httpx.RequestError as exc:
            logger.warning("Network error contacting Copernicus STAC: %s", exc)
            return []

    def _parse_stac_feature(self, feat: Dict[str, Any]) -> Optional[SARSceneSummary]:
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
                dt = datetime.now(timezone.utc)

            platform = props.get("platform") or "Sentinel-1"
            orbit_pass = props.get("sat:orbit_state")
            orbit_number = props.get("sat:relative_orbit") or props.get("sat:absolute_orbit")
            polarizations = props.get("sar:polarizations")
            if isinstance(polarizations, list):
                polarization_str = "+".join(polarizations)
            else:
                polarization_str = "VV+VH"

            processing_level = props.get("processing:level") or "LEVEL1_GRD"

            # Check thumbnail URL
            thumbnail_url = None
            if "thumbnail" in assets:
                thumbnail_url = assets["thumbnail"].get("href")

            # Check download asset URL
            download_url = None
            if "Product" in assets:
                download_url = assets["Product"].get("href")
            elif "vv" in assets and "alternate" in assets["vv"] and "https" in assets["vv"]["alternate"]:
                download_url = assets["vv"]["alternate"]["https"].get("href")

            return SARSceneSummary(
                scene_id=scene_id,
                acquisition_time=dt,
                platform=platform,
                orbit_pass=orbit_pass,
                orbit_number=orbit_number,
                footprint=geom,
                polarization=polarization_str,
                processing_level=processing_level,
                source="Copernicus CDSE",
                quicklook_url=thumbnail_url,
                download_url=download_url,
                scene_metadata=props,
            )
        except Exception as exc:
            logger.warning("Failed to parse STAC feature %s: %s", feat.get("id"), exc)
            return None
