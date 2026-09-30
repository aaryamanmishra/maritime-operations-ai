import React, { useState, useEffect } from "react";
import {
  SARSceneSummary,
  SARDetectionItem,
  SARJobResponse,
  searchSARScenes,
  startSARJob,
  fetchSARJobStatus,
  fetchAnalyzedScenes,
  fetchSceneDetections,
} from "../services/sarApi";

interface DarkVesselControlPanelProps {
  onSceneSelect: (scene: SARSceneSummary | null) => void;
  onDetectionsLoaded: (detections: SARDetectionItem[]) => void;
  onSelectDetection: (detection: SARDetectionItem | null) => void;
  selectedDetection: SARDetectionItem | null;
}

export const DarkVesselControlPanel: React.FC<DarkVesselControlPanelProps> = ({
  onSceneSelect,
  onDetectionsLoaded,
  onSelectDetection,
  selectedDetection,
}) => {
  // Search parameters
  const [minLon, setMinLon] = useState<number>(-2.0);
  const [minLat, setMinLat] = useState<number>(49.5);
  const [maxLon, setMaxLon] = useState<number>(2.5);
  const [maxLat, setMaxLat] = useState<number>(51.5);

  const [startTime, setStartTime] = useState<string>(() => {
    const d = new Date();
    d.setDate(d.getDate() - 30);
    return d.toISOString().slice(0, 10);
  });
  const [endTime, setEndTime] = useState<string>(() => {
    return new Date().toISOString().slice(0, 10);
  });

  // State
  const [searching, setSearching] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [scenes, setScenes] = useState<SARSceneSummary[]>([]);
  const [selectedScene, setSelectedScene] = useState<SARSceneSummary | null>(null);

  // Job execution state
  const [activeJob, setActiveJob] = useState<SARJobResponse | null>(null);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const [detections, setDetections] = useState<SARDetectionItem[]>([]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setSearching(true);
    setSearchError(null);
    setScenes([]);
    setSelectedScene(null);
    onSceneSelect(null);
    setDetections([]);
    onDetectionsLoaded([]);

    try {
      const results = await searchSARScenes({
        bbox: [minLon, minLat, maxLon, maxLat],
        start_time: `${startTime}T00:00:00Z`,
        end_time: `${endTime}T23:59:59Z`,
        limit: 15,
      });
      setScenes(results);
      if (results.length === 0) {
        setSearchError("No Sentinel-1 observations found in selected area and time range.");
      }
    } catch (err: any) {
      setSearchError(err.message || "Failed to query Copernicus STAC catalog.");
    } finally {
      setSearching(false);
    }
  };

  const handleSelectScene = (scene: SARSceneSummary) => {
    setSelectedScene(scene);
    onSceneSelect(scene);
    setJobError(null);
    setActiveJob(null);
    setDetections([]);
    onDetectionsLoaded([]);

    // Check if scene was previously analyzed
    fetchSceneDetections(scene.scene_id)
      .then((existingDets) => {
        if (existingDets && existingDets.length > 0) {
          setDetections(existingDets);
          onDetectionsLoaded(existingDets);
        }
      })
      .catch(() => {
        // Not analyzed yet
      });
  };

  const handleAnalyzeScene = async () => {
    if (!selectedScene) return;
    setAnalyzing(true);
    setJobError(null);

    try {
      const job = await startSARJob(selectedScene.scene_id, selectedScene);
      setActiveJob(job);

      // Poll job status until complete or failed
      const pollInterval = setInterval(async () => {
        try {
          const status = await fetchSARJobStatus(job.job_id);
          setActiveJob(status);

          if (status.status === "COMPLETE") {
            clearInterval(pollInterval);
            setAnalyzing(false);
            setDetections(status.detections);
            onDetectionsLoaded(status.detections);
          } else if (status.status === "FAILED") {
            clearInterval(pollInterval);
            setAnalyzing(false);
            setJobError(status.error_detail || status.message || "SAR analysis failed.");
          }
        } catch (pollErr: any) {
          clearInterval(pollInterval);
          setAnalyzing(false);
          setJobError(pollErr.message || "Error polling SAR job status.");
        }
      }, 2000);
    } catch (err: any) {
      setAnalyzing(false);
      setJobError(err.message || "Failed to start SAR analysis job.");
    }
  };

  const matchedCount = detections.filter((d) => d.match_status === "AIS-MATCHED").length;
  const unmatchedCount = detections.filter((d) => d.match_status === "AIS-UNMATCHED").length;

  return (
    <div className="absolute top-4 right-4 z-20 w-96 max-h-[calc(100vh-8rem)] overflow-y-auto bg-slate-900/95 backdrop-blur border border-slate-700/80 rounded-lg shadow-2xl p-4 text-xs text-slate-200">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-violet-400 animate-pulse" />
          <h2 className="font-semibold text-sm text-slate-100 uppercase tracking-wide">
            Dark-Vessel Intelligence
          </h2>
        </div>
        <span className="text-[10px] text-violet-400 bg-violet-950/60 px-2 py-0.5 rounded border border-violet-800/50">
          Copernicus SAR
        </span>
      </div>

      {/* Observation Notice */}
      <div className="mt-2 text-[10px] text-slate-400 leading-tight">
        Synthetic Aperture Radar observations from Sentinel-1 GRD. Target correlation with AIS broadcasts within ±15 minutes.
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="mt-3 space-y-2">
        <div className="text-[11px] font-medium text-slate-300">Observation Search Area (BBox)</div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-slate-400">Min Lon / Lat</label>
            <div className="flex gap-1">
              <input
                type="number"
                step="0.1"
                value={minLon}
                onChange={(e) => setMinLon(parseFloat(e.target.value))}
                className="w-full bg-slate-950/80 border border-slate-700 rounded px-1.5 py-1 text-slate-200"
              />
              <input
                type="number"
                step="0.1"
                value={minLat}
                onChange={(e) => setMinLat(parseFloat(e.target.value))}
                className="w-full bg-slate-950/80 border border-slate-700 rounded px-1.5 py-1 text-slate-200"
              />
            </div>
          </div>
          <div>
            <label className="text-[10px] text-slate-400">Max Lon / Lat</label>
            <div className="flex gap-1">
              <input
                type="number"
                step="0.1"
                value={maxLon}
                onChange={(e) => setMaxLon(parseFloat(e.target.value))}
                className="w-full bg-slate-950/80 border border-slate-700 rounded px-1.5 py-1 text-slate-200"
              />
              <input
                type="number"
                step="0.1"
                value={maxLat}
                onChange={(e) => setMaxLat(parseFloat(e.target.value))}
                className="w-full bg-slate-950/80 border border-slate-700 rounded px-1.5 py-1 text-slate-200"
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 mt-2">
          <div>
            <label className="text-[10px] text-slate-400">Start Date</label>
            <input
              type="date"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-700 rounded px-2 py-1 text-slate-200"
            />
          </div>
          <div>
            <label className="text-[10px] text-slate-400">End Date</label>
            <input
              type="date"
              value={endTime}
              onChange={(e) => setEndTime(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-700 rounded px-2 py-1 text-slate-200"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={searching}
          className="w-full mt-2 py-1.5 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 text-white font-medium rounded transition flex items-center justify-center gap-1.5"
        >
          {searching ? "Searching STAC Catalog..." : "Search Sentinel-1 Scenes"}
        </button>
      </form>

      {searchError && (
        <div className="mt-3 p-2 bg-rose-950/60 border border-rose-700 text-rose-300 rounded text-[11px]">
          {searchError}
        </div>
      )}

      {/* Found Scenes List */}
      {scenes.length > 0 && (
        <div className="mt-3">
          <div className="text-[11px] font-medium text-slate-300 mb-1 flex justify-between">
            <span>Available Sentinel-1 GRD Scenes ({scenes.length})</span>
          </div>
          <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
            {scenes.map((sc) => {
              const isSelected = selectedScene?.scene_id === sc.scene_id;
              const dateStr = new Date(sc.acquisition_time).toUTCString();
              return (
                <div
                  key={sc.scene_id}
                  onClick={() => handleSelectScene(sc)}
                  className={`p-2 rounded border cursor-pointer transition ${
                    isSelected
                      ? "bg-violet-950/60 border-violet-500"
                      : "bg-slate-950/40 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex justify-between items-center text-[11px] font-semibold text-slate-200">
                    <span>{sc.platform}</span>
                    <span className="text-[10px] text-violet-400 font-mono">
                      {sc.orbit_pass || "ORBIT"}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    OBSERVATION TIME: <span className="text-slate-300">{dateStr}</span>
                  </div>
                  <div className="text-[9px] text-slate-500 truncate mt-0.5 font-mono">
                    ID: {sc.scene_id}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Selected Scene Action */}
      {selectedScene && (
        <div className="mt-3 p-3 bg-slate-950/70 border border-violet-900/60 rounded">
          <div className="flex justify-between items-center mb-1">
            <span className="font-semibold text-[11px] text-violet-300">Selected Scene</span>
            <span className="text-[10px] text-slate-400">{selectedScene.platform}</span>
          </div>
          <div className="text-[10px] text-slate-400 font-mono truncate mb-2">
            {selectedScene.scene_id}
          </div>

          <button
            type="button"
            onClick={handleAnalyzeScene}
            disabled={analyzing}
            className="w-full py-1.5 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 disabled:opacity-50 text-white font-medium rounded transition flex items-center justify-center gap-1.5"
          >
            {analyzing ? "Analyzing SAR Observations..." : "Analyze Scene & Correlate AIS"}
          </button>
        </div>
      )}

      {/* Job Execution Status */}
      {activeJob && (
        <div className="mt-3 p-2.5 bg-slate-950/80 border border-slate-800 rounded">
          <div className="flex justify-between items-center text-[10px] mb-1">
            <span className="font-mono text-violet-400">{activeJob.status}</span>
            <span className="text-slate-400">{activeJob.progress_percent}%</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-violet-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${activeJob.progress_percent}%` }}
            />
          </div>
          <div className="text-[10px] text-slate-400 mt-1.5 truncate">
            {activeJob.message}
          </div>
        </div>
      )}

      {jobError && (
        <div className="mt-3 p-2 bg-rose-950/60 border border-rose-700 text-rose-300 rounded text-[11px]">
          {jobError}
        </div>
      )}

      {/* Explicit No Detections State */}
      {selectedScene && !analyzing && activeJob?.status === "COMPLETE" && detections.length === 0 && (
        <div className="mt-3 p-3 bg-slate-950/60 border border-slate-800 rounded text-center">
          <div className="text-slate-400 font-medium text-[11px]">No Radar Contacts Detected</div>
          <div className="text-slate-500 text-[10px] mt-0.5">
            YOLO26s found 0 vessel detections above confidence threshold in this scene.
          </div>
        </div>
      )}

      {/* Detection Results */}
      {detections.length > 0 && (
        <div className="mt-4 pt-3 border-t border-slate-800">
          <div className="flex justify-between items-center mb-2">
            <span className="font-semibold text-slate-200">Detections Summary</span>
            <span className="font-mono text-xs text-violet-400 font-bold">
              {detections.length} Contacts
            </span>
          </div>

          {/* Counts badges */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            <div className="p-2 bg-emerald-950/50 border border-emerald-800/60 rounded text-center">
              <div className="text-sm font-bold text-emerald-400">{matchedCount}</div>
              <div className="text-[10px] text-emerald-300 uppercase tracking-wider font-semibold">
                AIS-MATCHED
              </div>
            </div>
            <div className="p-2 bg-amber-950/50 border border-amber-800/60 rounded text-center">
              <div className="text-sm font-bold text-amber-400">{unmatchedCount}</div>
              <div className="text-[10px] text-amber-300 uppercase tracking-wider font-semibold">
                AIS-UNMATCHED
              </div>
            </div>
          </div>

          {/* Detections List */}
          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {detections.map((det) => {
              const isSelected = selectedDetection?.detection_id === det.detection_id;
              const isMatched = det.match_status === "AIS-MATCHED";
              return (
                <div
                  key={det.detection_id}
                  onClick={() => onSelectDetection(det)}
                  className={`p-2 rounded border cursor-pointer transition ${
                    isSelected
                      ? "bg-slate-800 border-violet-400 shadow-md"
                      : "bg-slate-950/50 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span className="font-mono text-[10px] text-slate-300">
                      {det.latitude.toFixed(4)}°, {det.longitude.toFixed(4)}°
                    </span>
                    <span
                      className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                        isMatched
                          ? "bg-emerald-900/80 text-emerald-300 border border-emerald-700/60"
                          : "bg-amber-900/80 text-amber-300 border border-amber-700/60"
                      }`}
                    >
                      {det.match_status}
                    </span>
                  </div>

                  <div className="flex justify-between items-center text-[10px] text-slate-400 mt-1">
                    <span>Conf: {(det.confidence * 100).toFixed(0)}%</span>
                    {isMatched && det.candidate_mmsi ? (
                      <span className="text-emerald-400 font-mono">
                        MMSI: {det.candidate_mmsi} ({det.distance_km} km)
                      </span>
                    ) : (
                      <span className="text-slate-500">No AIS candidate</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Selected Detection Detail Provenance */}
          {selectedDetection && (
            <div className="mt-3 p-2.5 bg-slate-950 border border-slate-700 rounded text-[10px] space-y-1">
              <div className="font-semibold text-slate-200 border-b border-slate-800 pb-1">
                Contact Observation Provenance
              </div>
              <div>ID: <span className="font-mono text-slate-300">{selectedDetection.detection_id}</span></div>
              <div>Coordinates: <span className="font-mono text-slate-300">{selectedDetection.latitude.toFixed(5)}°, {selectedDetection.longitude.toFixed(5)}°</span></div>
              <div>Status: <span className={selectedDetection.match_status === "AIS-MATCHED" ? "text-emerald-400 font-bold" : "text-amber-400 font-bold"}>{selectedDetection.match_status}</span></div>
              {selectedDetection.candidate_mmsi && (
                <>
                  <div>Matched MMSI: <span className="font-mono text-emerald-300">{selectedDetection.candidate_mmsi}</span></div>
                  <div>AIS Offset: <span className="font-mono text-slate-300">{selectedDetection.distance_km} km / {selectedDetection.time_difference_seconds}s</span></div>
                </>
              )}
              {selectedDetection.provenance?.explanation && (
                <div className="text-[9px] text-slate-400 italic pt-1 border-t border-slate-800">
                  {selectedDetection.provenance.explanation}
                </div>
              )}
            </div>
          )}

          {/* Ethical Disclaimer */}
          <div className="mt-3 p-2 bg-slate-950/40 border border-slate-800 rounded text-[9px] text-slate-500 leading-tight">
            <strong>Operational Notice:</strong> AIS-unmatched radar contacts represent surface observations with no correlating AIS broadcast under configured spatiotemporal thresholds (±15 min, 3.0 km). No criminality or intentional disabling is inferred.
          </div>
        </div>
      )}
    </div>
  );
};
