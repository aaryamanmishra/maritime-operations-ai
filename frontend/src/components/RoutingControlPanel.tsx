import React, { useState, useEffect } from "react";
import {
  GeoCoordinate,
  VesselCharacteristics,
  FuelParameters,
  RouteOptimizationRequest,
  RouteOptimizationResponse,
  fetchVesselDefaults,
  optimizeRoute,
} from "../services/routingApi";

interface RoutingControlPanelProps {
  onRouteCalculated: (response: RouteOptimizationResponse | null) => void;
  activePickingMode: "origin" | "destination" | null;
  onSetPickingMode: (mode: "origin" | "destination" | null) => void;
  origin: GeoCoordinate | null;
  destination: GeoCoordinate | null;
  onUpdateOrigin: (coord: GeoCoordinate) => void;
  onUpdateDestination: (coord: GeoCoordinate) => void;
}

export const RoutingControlPanel: React.FC<RoutingControlPanelProps> = ({
  onRouteCalculated,
  activePickingMode,
  onSetPickingMode,
  origin,
  destination,
  onUpdateOrigin,
  onUpdateDestination,
}) => {
  const [vesselType, setVesselType] = useState<string>("cargo");
  const [lengthM, setLengthM] = useState<number>(190.0);
  const [beamM, setBeamM] = useState<number>(28.0);
  const [draftM, setDraftM] = useState<number>(10.5);
  const [designSpeedKn, setDesignSpeedKn] = useState<number>(14.5);
  const [fuelPrice, setFuelPrice] = useState<number>(620.0);
  const [fuelType, setFuelType] = useState<"VLSFO" | "MGO" | "HFO">("VLSFO");
  const [priority, setPriority] = useState<"balanced" | "fuel_min" | "time_min">("balanced");
  const [departureTime, setDepartureTime] = useState<string>(
    new Date().toISOString().slice(0, 16)
  );

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RouteOptimizationResponse | null>(null);
  const [showSegments, setShowSegments] = useState<boolean>(false);

  // Set default initial coordinates (Brest to Dover)
  useEffect(() => {
    if (!origin) {
      onUpdateOrigin({ latitude: 48.38, longitude: -4.5 });
    }
    if (!destination) {
      onUpdateDestination({ latitude: 51.12, longitude: 1.3 });
    }
  }, []);

  const handleVesselTypeChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = e.target.value;
    setVesselType(selected);
    try {
      const defaults = await fetchVesselDefaults();
      const d = defaults[selected];
      if (d) {
        setLengthM(d.length_m);
        setBeamM(d.beam_m);
        setDraftM(d.draft_m);
        setDesignSpeedKn(d.design_speed_kn);
        setFuelPrice(d.fuel_price_per_tonne);
        setFuelType(d.fuel_type as any);
      }
    } catch (err) {
      console.warn("Using offline vessel defaults", err);
    }
  };

  const handleCalculateRoute = async () => {
    if (!origin || !destination) {
      setError("Please specify both origin and destination coordinates.");
      return;
    }

    setLoading(true);
    setError(null);

    const req: RouteOptimizationRequest = {
      origin,
      destination,
      departure_time: new Date(departureTime).toISOString(),
      vessel: {
        vessel_type: vesselType as any,
        length_m: lengthM,
        beam_m: beamM,
        draft_m: draftM,
        design_speed_kn: designSpeedKn,
      },
      fuel: {
        fuel_price_per_tonne: fuelPrice,
        fuel_type: fuelType,
      },
      optimization_priority: priority,
    };

    try {
      const res = await optimizeRoute(req);
      setResult(res);
      onRouteCalculated(res);
    } catch (err: any) {
      setError(err.message || "Route calculation failed");
      onRouteCalculated(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="absolute top-4 right-4 z-20 w-96 max-h-[calc(100vh-8rem)] overflow-y-auto bg-slate-900/95 backdrop-blur border border-slate-700/80 rounded-lg shadow-2xl p-4 text-xs text-slate-200">
      <div className="flex items-center justify-between pb-2 border-b border-slate-700">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
          <h2 className="font-semibold text-sm text-slate-150 uppercase tracking-wide">
            AI Weather-Aware Routing
          </h2>
        </div>
        <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/50">
          TFT Surrogate
        </span>
      </div>

      {error && (
        <div className="mt-3 p-2 bg-rose-950/60 border border-rose-700 text-rose-300 rounded text-[11px]">
          {error}
        </div>
      )}

      {/* Origin & Destination Controls */}
      <div className="mt-3 space-y-2">
        <label className="text-[11px] font-medium text-slate-300">Voyage Waypoints</label>
        
        {/* Origin */}
        <div className="flex items-center gap-2 bg-slate-950/60 p-2 rounded border border-slate-800">
          <div className="w-2 h-2 rounded-full bg-emerald-400" />
          <div className="flex-1">
            <span className="text-[10px] text-slate-400">Origin: </span>
            <span className="font-mono text-slate-200">
              {origin ? `${origin.latitude.toFixed(3)}°, ${origin.longitude.toFixed(3)}°` : "Not set"}
            </span>
          </div>
          <button
            type="button"
            onClick={() => onSetPickingMode(activePickingMode === "origin" ? null : "origin")}
            className={`px-2 py-1 rounded text-[10px] font-medium transition ${
              activePickingMode === "origin"
                ? "bg-emerald-600 text-white animate-pulse"
                : "bg-slate-800 hover:bg-slate-700 text-slate-300"
            }`}
          >
            {activePickingMode === "origin" ? "Click Map..." : "Set on Map"}
          </button>
        </div>

        {/* Destination */}
        <div className="flex items-center gap-2 bg-slate-950/60 p-2 rounded border border-slate-800">
          <div className="w-2 h-2 rounded-full bg-rose-400" />
          <div className="flex-1">
            <span className="text-[10px] text-slate-400">Destination: </span>
            <span className="font-mono text-slate-200">
              {destination ? `${destination.latitude.toFixed(3)}°, ${destination.longitude.toFixed(3)}°` : "Not set"}
            </span>
          </div>
          <button
            type="button"
            onClick={() => onSetPickingMode(activePickingMode === "destination" ? null : "destination")}
            className={`px-2 py-1 rounded text-[10px] font-medium transition ${
              activePickingMode === "destination"
                ? "bg-rose-600 text-white animate-pulse"
                : "bg-slate-800 hover:bg-slate-700 text-slate-300"
            }`}
          >
            {activePickingMode === "destination" ? "Click Map..." : "Set on Map"}
          </button>
        </div>
      </div>

      {/* Departure Time */}
      <div className="mt-3">
        <label className="text-[11px] font-medium text-slate-300">Departure Time (UTC)</label>
        <input
          type="datetime-local"
          value={departureTime}
          onChange={(e) => setDepartureTime(e.target.value)}
          className="mt-1 w-full bg-slate-950 border border-slate-800 rounded p-1.5 text-xs text-slate-200 font-mono"
        />
      </div>

      {/* Vessel Profile */}
      <div className="mt-3 space-y-2 border-t border-slate-800 pt-3">
        <div className="flex items-center justify-between">
          <label className="text-[11px] font-medium text-slate-300">Vessel Parameters</label>
          <select
            value={vesselType}
            onChange={handleVesselTypeChange}
            className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200"
          >
            <option value="cargo">Container / Cargo</option>
            <option value="tanker">Oil / Chemical Tanker</option>
            <option value="passenger">Passenger / Ferry</option>
            <option value="fishing">Commercial Fishing</option>
            <option value="tug">Tug / Supply</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="text-[10px] text-slate-400">Design Speed (kn)</span>
            <input
              type="number"
              step="0.1"
              value={designSpeedKn}
              onChange={(e) => setDesignSpeedKn(parseFloat(e.target.value) || 12.0)}
              className="mt-0.5 w-full bg-slate-950 border border-slate-800 rounded p-1 font-mono text-slate-200"
            />
          </div>
          <div>
            <span className="text-[10px] text-slate-400">Length (m)</span>
            <input
              type="number"
              value={lengthM}
              onChange={(e) => setLengthM(parseFloat(e.target.value) || 100.0)}
              className="mt-0.5 w-full bg-slate-950 border border-slate-800 rounded p-1 font-mono text-slate-200"
            />
          </div>
          <div>
            <span className="text-[10px] text-slate-400">Beam (m)</span>
            <input
              type="number"
              value={beamM}
              onChange={(e) => setBeamM(parseFloat(e.target.value) || 20.0)}
              className="mt-0.5 w-full bg-slate-950 border border-slate-800 rounded p-1 font-mono text-slate-200"
            />
          </div>
          <div>
            <span className="text-[10px] text-slate-400">Draft (m)</span>
            <input
              type="number"
              step="0.1"
              value={draftM}
              onChange={(e) => setDraftM(parseFloat(e.target.value) || 8.0)}
              className="mt-0.5 w-full bg-slate-950 border border-slate-800 rounded p-1 font-mono text-slate-200"
            />
          </div>
        </div>
      </div>

      {/* Fuel & Priority */}
      <div className="mt-3 grid grid-cols-2 gap-2 border-t border-slate-800 pt-3">
        <div>
          <span className="text-[10px] text-slate-400">Fuel Price ($/Tonne)</span>
          <input
            type="number"
            value={fuelPrice}
            onChange={(e) => setFuelPrice(parseFloat(e.target.value) || 600.0)}
            className="mt-0.5 w-full bg-slate-950 border border-slate-800 rounded p-1 font-mono text-slate-200"
          />
        </div>
        <div>
          <span className="text-[10px] text-slate-400">Priority</span>
          <select
            value={priority}
            onChange={(e) => setPriority(e.target.value as any)}
            className="mt-0.5 w-full bg-slate-950 border border-slate-800 rounded p-1 text-slate-200"
          >
            <option value="balanced">Balanced</option>
            <option value="fuel_min">Minimize Fuel</option>
            <option value="time_min">Minimize Time</option>
          </select>
        </div>
      </div>

      {/* Calculate Button */}
      <button
        type="button"
        disabled={loading}
        onClick={handleCalculateRoute}
        className="mt-4 w-full bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900/50 text-white font-medium py-2 rounded transition flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            <span>Optimizing SeaRoute & Weather...</span>
          </>
        ) : (
          <span>Calculate Weather-Aware Route</span>
        )}
      </button>

      {/* Route Results Display */}
      {result && (
        <div className="mt-4 border-t border-slate-700 pt-3 space-y-3">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-100 text-xs">Route Comparison</span>
            <span className={`text-[10px] px-2 py-0.5 rounded font-mono ${
              result.route_diverged
                ? "bg-amber-950 text-amber-300 border border-amber-800"
                : "bg-slate-800 text-slate-300"
            }`}>
              {result.route_diverged ? "Weather Detour Active" : "Base Corridor Followed"}
            </span>
          </div>

          {/* Metric Comparison Table */}
          <div className="bg-slate-950/70 p-2.5 rounded border border-slate-800 space-y-2">
            <div className="grid grid-cols-3 gap-1 text-[11px]">
              <div>
                <span className="text-[9px] text-slate-400 uppercase">Distance</span>
                <p className="font-mono text-slate-100">{result.weather_aware_route.summary.total_distance_nm} nm</p>
                <span className="text-[9px] text-slate-500">Base: {result.base_route.summary.total_distance_nm}</span>
              </div>
              <div>
                <span className="text-[9px] text-slate-400 uppercase">Duration</span>
                <p className="font-mono text-slate-100">{result.weather_aware_route.summary.estimated_duration_hours} hrs</p>
                <span className="text-[9px] text-slate-500">Base: {result.base_route.summary.estimated_duration_hours}</span>
              </div>
              <div>
                <span className="text-[9px] text-slate-400 uppercase">Max Wave</span>
                <p className="font-mono text-slate-100">{result.weather_aware_route.summary.max_wave_height_m} m</p>
                <span className="text-[9px] text-slate-500">Base: {result.base_route.summary.max_wave_height_m}</span>
              </div>
            </div>

            <div className="border-t border-slate-800/80 pt-1.5 grid grid-cols-2 gap-2 text-[11px]">
              <div>
                <span className="text-[9px] text-slate-400 uppercase">Est. Fuel (Tonnes)*</span>
                <p className="font-mono text-cyan-400 font-semibold">
                  {result.weather_aware_route.summary.total_estimated_fuel_tonnes} T
                </p>
              </div>
              <div>
                <span className="text-[9px] text-slate-400 uppercase">Est. Cost (USD)*</span>
                <p className="font-mono text-emerald-400 font-semibold">
                  ${result.weather_aware_route.summary.total_estimated_cost_usd.toLocaleString()}
                </p>
              </div>
            </div>
          </div>

          {/* Provenance and Disclaimer Notice */}
          <div className="p-2 bg-slate-950/80 rounded border border-slate-800 text-[9px] text-slate-400 leading-relaxed">
            <span className="text-amber-400 font-semibold">*ESTIMATE DISCLAIMER: </span>
            {result.disclaimer}
          </div>

          {/* Toggle Segments Table */}
          <button
            type="button"
            onClick={() => setShowSegments(!showSegments)}
            className="w-full py-1 text-[10px] text-cyan-400 hover:text-cyan-300 border border-cyan-900/60 rounded bg-cyan-950/30"
          >
            {showSegments ? "Hide Segment Provenance" : `View ${result.weather_aware_route.segments.length} Leg Segments`}
          </button>

          {showSegments && (
            <div className="max-h-48 overflow-y-auto space-y-1 pr-1 font-mono text-[9px]">
              {result.weather_aware_route.segments.map((seg) => (
                <div key={seg.segment_index} className="p-1.5 bg-slate-950 rounded border border-slate-800/80 flex items-center justify-between">
                  <div>
                    <span className="text-cyan-400">Leg #{seg.segment_index + 1}</span> ({seg.distance_nm} nm @ {seg.bearing_deg}°)
                    <div className="text-slate-400">
                      Waves: {seg.weather.wave_height_m}m | Wind: {seg.weather.wind_speed_mps.toFixed(1)} m/s
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-emerald-400">{seg.predicted_speed_kn} kn</span>
                    <div className="text-slate-500">-{seg.speed_loss_kn} kn loss</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
