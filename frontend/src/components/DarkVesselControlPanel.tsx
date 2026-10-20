import React, { useState } from "react";
import {
  SARSceneSummary,
  SARDetectionItem,
  SARJobResponse,
  SARModality,
  searchSARScenes,
  startSARJob,
  fetchSARJobStatus,
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
  const [modality, setModality] = useState<SARModality>("GRD");

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
  const [showClutterDiagnostics, setShowClutterDiagnostics] = useState<boolean>(true);

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
        modality,
      });
      setScenes(results);
      if (results.length === 0) {
        setSearchError(`No Sentinel-1 ${modality} observations found in selected area and time range.`);
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

    const sceneModality = (selectedScene.modality as SARModality) || modality || "GRD";

    try {
      const job = await startSARJob(selectedScene.scene_id, selectedScene, sceneModality);
      setActiveJob(job);

      if (job.status === "COMPLETE") {
        setAnalyzing(false);
        setDetections(job.detections || []);
        onDetectionsLoaded(job.detections || []);
        return;
      }
      if (job.status === "FAILED") {
        setAnalyzing(false);
        setJobError(job.error_detail || job.message || "SAR analysis failed.");
        return;
      }

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

  // Explicit operational classification:
  // Operational vessels exclude terrestrial clutter (is_operational_vessel: false).
  // Backwards compatibility: undefined is_operational_vessel (e.g. legacy GRD) is treated as operational.
  const operationalDetections = detections.filter((d) => d.is_operational_vessel !== false);
  const clutterCandidates = detections.filter((d) => d.is_operational_vessel === false);

  const operationalCount =
    activeJob?.detections_count !== undefined
      ? activeJob.detections_count
      : operationalDetections.length;

  const matchedCount = operationalDetections.filter((d) => d.match_status === "AIS-MATCHED").length;
  const darkVesselCount = operationalDetections.filter((d) => d.match_status === "AIS-UNMATCHED").length;
  const clutterCount = clutterCandidates.length;

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
        Synthetic Aperture Radar observations from Sentinel-1 ({modality}). Correlated with AIS broadcasts within ±15 minutes with geographic surface validation.
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="mt-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium text-slate-300">Observation Search Area (BBox) & Modality</span>
          <div className="flex gap-1 bg-slate-950 p-0.5 rounded border border-slate-800">
            <button
              type="button"
              onClick={() => setModality("GRD")}
              className={`px-2 py-0.5 text-[10px] font-semibold rounded transition ${
                modality === "GRD"
                  ? "bg-violet-700 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              GRD (Default)
            </button>
            <button
              type="button"
              onClick={() => setModality("SLC")}
              className={`px-2 py-0.5 text-[10px] font-semibold rounded transition ${
                modality === "SLC"
                  ? "bg-violet-700 text-white"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              SLC
            </button>
          </div>
        </div>

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
          {searching ? "Searching STAC Catalog..." : `Search Sentinel-1 Scenes (${modality})`}
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
            <span>Available Sentinel-1 {modality} Scenes ({scenes.length})</span>
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
                      {sc.orbit_pass || "ORBIT"} &bull; {sc.modality || modality}
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
            <span className="text-[10px] text-slate-400">
              {selectedScene.platform} ({selectedScene.modality || modality})
            </span>
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
            <span className="font-mono text-violet-400">
              {activeJob.status} ({activeJob.modality || modality})
            </span>
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
      {selectedScene && !analyzing && activeJob?.status === "COMPLETE" && operationalCount === 0 && (
        <div className="mt-3 p-3 bg-slate-950/60 border border-slate-800 rounded text-center">
          <div className="text-slate-300 font-medium text-[11px]">0 Operational Vessels Detected</div>
          <div className="text-slate-400 text-[10px] mt-0.5">
            {clutterCount > 0
              ? `${clutterCount} candidate(s) rejected as terrestrial radar clutter by geographic-context validation.`
              : "YOLO26s found 0 vessel detections above confidence threshold in this scene."}
          </div>
        </div>
      )}

      {/* Detection Results */}
      {detections.length > 0 && (
        <div className="mt-4 pt-3 border-t border-slate-800">
          <div className="flex justify-between items-center mb-2">
            <span className="font-semibold text-slate-200">Operational Summary</span>
            <span className="font-mono text-xs text-violet-400 font-bold" data-testid="operational-vessels-count">
              {operationalCount} Operational Vessels
            </span>
          </div>

          {/* Counts badges */}
          <div className={`grid ${clutterCount > 0 ? "grid-cols-3" : "grid-cols-2"} gap-1.5 mb-3`}>
            <div className="p-2 bg-emerald-950/50 border border-emerald-800/60 rounded text-center">
              <div className="text-sm font-bold text-emerald-400" data-testid="matched-count">{matchedCount}</div>
              <div className="text-[9px] text-emerald-300 uppercase tracking-wider font-semibold">
                AIS-MATCHED
              </div>
            </div>
            <div className="p-2 bg-amber-950/50 border border-amber-800/60 rounded text-center">
              <div className="text-sm font-bold text-amber-400" data-testid="unmatched-count">{darkVesselCount}</div>
              <div className="text-[9px] text-amber-300 uppercase tracking-wider font-semibold">
                DARK VESSELS
              </div>
            </div>
            {clutterCount > 0 && (
              <div className="p-2 bg-rose-950/50 border border-rose-800/60 rounded text-center">
                <div className="text-sm font-bold text-rose-400" data-testid="clutter-count">{clutterCount}</div>
                <div className="text-[9px] text-rose-300 uppercase tracking-wider font-semibold">
                  CLUTTER
                </div>
              </div>
            )}
          </div>

          {/* Operational Detections List */}
          {operationalDetections.length > 0 && (
            <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1">
              <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                Operational Surface Contacts ({operationalDetections.length})
              </div>
              {operationalDetections.map((det) => {
                const isSelected = selectedDetection?.detection_id === det.detection_id;
                const isMatched = det.match_status === "AIS-MATCHED";
                const isAmbiguousCoastal = det.surface_context === "AMBIGUOUS_COASTAL";
                const isUnknown = det.surface_context === "UNKNOWN";

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
                      <div className="flex items-center gap-1">
                        {isAmbiguousCoastal && (
                          <span
                            className="text-[9px] font-semibold px-1 py-0.5 rounded bg-amber-950/90 text-amber-300 border border-amber-600/80"
                            title="Nearshore boundary uncertainty (≤10 km from coastline)"
                          >
                            COASTAL AMBIGUITY (≤10 km)
                          </span>
                        )}
                        {isUnknown && (
                          <span
                            className="text-[9px] font-semibold px-1 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-600"
                            title="Surface context unverified"
                          >
                            UNKNOWN CONTEXT
                          </span>
                        )}
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
                    </div>

                    <div className="flex justify-between items-center text-[10px] text-slate-400 mt-1">
                      <span>
                        Conf: {(det.confidence * 100).toFixed(0)}% &bull; {det.modality || "GRD"}
                      </span>
                      {isMatched && det.candidate_mmsi ? (
                        <span className="text-emerald-400 font-mono">
                          MMSI: {det.candidate_mmsi} ({det.distance_km} km)
                        </span>
                      ) : (
                        <span className="text-amber-400/90 font-medium">Dark Vessel Target</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Rejected Terrestrial Clutter Diagnostic Section */}
          {clutterCount > 0 && (
            <div className="mt-3 pt-2 border-t border-slate-800">
              <div className="flex justify-between items-center mb-1.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-semibold text-rose-400 uppercase tracking-wider">
                    Rejected Terrestrial Clutter ({clutterCount})
                  </span>
                  <span className="text-[9px] bg-rose-950/80 text-rose-300 px-1.5 py-0.2 rounded border border-rose-800/60">
                    Non-Maritime
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setShowClutterDiagnostics((prev) => !prev)}
                  className="text-[10px] text-slate-400 hover:text-slate-200 underline"
                >
                  {showClutterDiagnostics ? "Hide" : "Show"}
                </button>
              </div>

              <div className="text-[9px] text-slate-400 leading-tight mb-2">
                Screened by Geographic-Context Validator. Excluded from operational vessel totals.
              </div>

              {showClutterDiagnostics && (
                <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                  {clutterCandidates.map((det) => {
                    const isSelected = selectedDetection?.detection_id === det.detection_id;
                    const inlandDist = det.provenance?.geographic_context?.distance_to_coast_km;

                    return (
                      <div
                        key={det.detection_id}
                        onClick={() => onSelectDetection(det)}
                        className={`p-2 rounded border cursor-pointer transition ${
                          isSelected
                            ? "bg-slate-900 border-rose-500 shadow-md"
                            : "bg-slate-950/70 border-rose-950/80 hover:border-rose-800/80"
                        }`}
                      >
                        <div className="flex justify-between items-center">
                          <span className="font-mono text-[10px] text-slate-400">
                            {det.latitude.toFixed(4)}°, {det.longitude.toFixed(4)}°
                          </span>
                          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800/70">
                            REJECTED CLUTTER
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-[10px] text-slate-400 mt-1">
                          <span>Conf: {(det.confidence * 100).toFixed(0)}% &bull; {det.modality || "SLC"}</span>
                          <span className="text-rose-400/90 font-mono text-[9px]">
                            {inlandDist ? `${inlandDist.toFixed(1)} km inland` : "Continental Clutter"}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Selected Detection Detail Provenance */}
          {selectedDetection && (
            <div className="mt-3 p-2.5 bg-slate-950 border border-slate-700 rounded text-[10px] space-y-1.5" data-testid="detection-provenance-panel">
              <div className="flex justify-between items-center border-b border-slate-800 pb-1">
                <span className="font-semibold text-slate-200">Contact Observation Provenance</span>
                <span
                  className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                    selectedDetection.is_operational_vessel === false
                      ? "bg-rose-950 text-rose-300 border border-rose-800"
                      : "bg-emerald-950 text-emerald-300 border border-emerald-800"
                  }`}
                >
                  {selectedDetection.is_operational_vessel === false
                    ? "REJECTED (Terrestrial Clutter)"
                    : "OPERATIONAL VESSEL"}
                </span>
              </div>

              <div>
                ID: <span className="font-mono text-slate-300">{selectedDetection.detection_id}</span>
              </div>
              <div>
                Coordinates:{" "}
                <span className="font-mono text-slate-300">
                  {selectedDetection.latitude.toFixed(5)}°, {selectedDetection.longitude.toFixed(5)}°
                </span>
              </div>
              <div className="flex justify-between">
                <span>
                  Modality: <span className="font-semibold text-slate-200">{selectedDetection.modality || activeJob?.modality || "GRD"}</span>
                </span>
                <span>
                  Confidence: <span className="font-mono text-slate-200">{(selectedDetection.confidence * 100).toFixed(1)}%</span>
                </span>
              </div>
              <div>
                Model: <span className="font-mono text-slate-300">{selectedDetection.model_version}</span>
              </div>

              {/* Surface Context & Geolocation */}
              <div className="pt-1 border-t border-slate-800/80 space-y-1">
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Surface Context:</span>
                  <span
                    className={`font-semibold ${
                      selectedDetection.surface_context === "TERRESTRIAL_CLUTTER"
                        ? "text-rose-400"
                        : selectedDetection.surface_context === "AMBIGUOUS_COASTAL"
                        ? "text-amber-400"
                        : selectedDetection.surface_context === "UNKNOWN"
                        ? "text-slate-400"
                        : "text-cyan-400"
                    }`}
                  >
                    {selectedDetection.surface_context || "PLAUSIBLE_WATER"}
                  </span>
                </div>

                {selectedDetection.provenance?.geographic_context?.reason && (
                  <div className="text-[9px] text-slate-400 italic bg-slate-900/60 p-1 rounded border border-slate-800">
                    {selectedDetection.provenance.geographic_context.reason}
                  </div>
                )}

                {selectedDetection.provenance?.geographic_context?.distance_to_coast_km !== undefined && (
                  <div className="text-[9px] text-slate-400">
                    Coast Distance:{" "}
                    <span className="font-mono text-slate-300">
                      {selectedDetection.provenance.geographic_context.distance_to_coast_km.toFixed(1)} km
                    </span>
                    {selectedDetection.provenance?.geographic_context?.data_source && (
                      <span className="text-slate-500 ml-1">
                        ({selectedDetection.provenance.geographic_context.data_source})
                      </span>
                    )}
                  </div>
                )}

                {/* Geolocation Grid Provenance */}
                {selectedDetection.provenance?.geolocation_method && (
                  <div className="text-[9px] text-slate-400">
                    Geolocation:{" "}
                    <span className="font-mono text-violet-300">
                      {selectedDetection.provenance.geolocation_method}
                    </span>
                  </div>
                )}

                {selectedDetection.provenance?.range_pixel_spacing_m !== undefined &&
                  selectedDetection.provenance?.azimuth_pixel_spacing_m !== undefined && (
                    <div className="text-[9px] text-slate-400">
                      Pixel Spacing:{" "}
                      <span className="font-mono text-slate-300">
                        Rg {selectedDetection.provenance.range_pixel_spacing_m.toFixed(2)} m &bull; Az{" "}
                        {selectedDetection.provenance.azimuth_pixel_spacing_m.toFixed(2)} m
                      </span>
                    </div>
                  )}
              </div>

              {/* AIS Correlation */}
              <div className="pt-1 border-t border-slate-800/80">
                <div className="flex justify-between items-center">
                  <span>Match Status:</span>
                  <span
                    className={
                      selectedDetection.match_status === "AIS-MATCHED"
                        ? "text-emerald-400 font-bold"
                        : "text-amber-400 font-bold"
                    }
                  >
                    {selectedDetection.match_status}
                  </span>
                </div>
                {selectedDetection.candidate_mmsi && (
                  <div className="mt-1">
                    <div>
                      Matched MMSI:{" "}
                      <span className="font-mono text-emerald-300">{selectedDetection.candidate_mmsi}</span>
                    </div>
                    <div>
                      AIS Offset:{" "}
                      <span className="font-mono text-slate-300">
                        {selectedDetection.distance_km} km / {selectedDetection.time_difference_seconds}s
                      </span>
                    </div>
                  </div>
                )}
                {selectedDetection.provenance?.explanation && (
                  <div className="text-[9px] text-slate-400 italic pt-1">
                    {selectedDetection.provenance.explanation}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Operational Notice */}
          <div className="mt-3 p-2 bg-slate-950/40 border border-slate-800 rounded text-[9px] text-slate-500 leading-tight">
            <strong>Operational Notice:</strong> AIS-unmatched radar contacts over plausible water represent potential dark vessels. Lack of AIS broadcast is never treated as proof of clutter.
          </div>
        </div>
      )}
    </div>
  );
};
