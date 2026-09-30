# AI Weather-Aware Maritime Routing Specification

## 1. Objectives & Hybrid Architecture

The weather routing subsystem computes navigable, fuel-efficient, and weather-conscious maritime transit trajectories between global origins and destinations.

### Architectural Workflow

```
AIS historical operations
+
vessel characteristics (length, beam, draft, block coeff)
+
environmental conditions (Hs, wave period, relative angle, current, wind)
+
route/segment context (corridor geometry, step duration)
        ↓
[domain.ml_performance]
Deep-Learning Vessel-Performance Model (Candidate: Temporal Fusion Transformer / Temporal Sequence Model)
        ↓
Predicted Vessel Performance / Speed Degradation (ΔV or Speed Loss Factor)
        ↓
[domain.weather_routing]
Physics / Hydrodynamic / Energy Model (Holtrop-Mennen & Kwon baseline equations)
        ↓
Route Segment Cost (Time, Effective Brake Power, Estimated Fuel Consumption)
        ↓
Weather-Aware Optimization (Multi-objective A* on Base SeaRoute Network)
```

### Operational Directives
- **SeaRoute as Base Maritime Network**: The Eurostat / Searoute maritime passage network provides the structural graph, guaranteeing open-water connectivity without crossing landmasses, shallow archipelagos, or navigating through closed canals.
- **Deep Learning for Performance, Not Direct Fuel Guessing**: The deep-learning model predicts operational speed degradation ($\Delta V$) or delivered power ratio under environmental load. It does NOT directly predict final fuel consumption without legitimate labeled fuel sensor logs (which are proprietary and absent from open AIS).
- **Physics-Informed Energy Conversion**: Predicted speed loss and required shaft power feed into established naval architecture formulations (SFOC engine load curves) to calculate estimated fuel burn and bunker cost.
- **Honest Estimation & Disclaimers**: All outputs are explicitly marked as computational estimates. The system never claims certified fuel savings or guaranteed-safe routing.

---

## 2. Dedicated Domain Boundary: `backend/app/domain/ml_performance/`

The ML performance component is strictly isolated from route graph traversal:
- **Interface Protocol**:
  ```python
  class VesselPerformancePredictor(typing.Protocol):
      async def predict_speed_degradation(
          self,
          vessel_profile: VesselParticulars,
          operational_context: OperationalSegmentContext,
          weather_context: Sequence[MarineWeatherVector]
      ) -> PerformanceDegradationResult:
          """
          Returns predicted speed degradation (knots lost),
          calm water vs. actual speed ratio, and power penalty index.
          """
          ...
  ```
- **Inputs**:
  - Vessel characteristics: Length between perpendiculars ($L_{bp}$), beam ($B$), design draft ($T$), block coefficient ($C_b$), deadweight tonnage (DWT).
  - Environmental vectors: Significant wave height ($H_s$), wave peak period ($T_p$), relative wave heading ($\mu_{wave} = |\theta_{vessel} - \theta_{wave}|$), ocean surface current speed along/transverse to heading ($V_{c,\parallel}, V_{c,\perp}$), wind speed and relative angle ($V_w, \mu_w$).
  - Segment context: Planned target calm-water speed ($V_{service}$), segment length.
- **Outputs**:
  - Predicted actual speed over ground ($V_{pred}$).
  - Speed loss magnitude ($\Delta V$).
  - Added resistance/power surge factor ($\Delta P / P_{calm}$).
  - Model confidence / epistemic uncertainty bounds.

---

## 3. Deep Learning Architecture Candidates & Selection Methodology

### 3.1 Candidate Model: Temporal Fusion Transformer (TFT)
- **Rationale**: Maritime voyages are inherently sequential temporal processes where weather evolution over past legs impacts subsequent engine thermal efficiency and sea-keeping dynamics. TFT natively supports:
  - Multi-horizon forecasting.
  - Static vessel metadata inputs (vessel dimensions, engine classification) alongside time-varying exogenous environmental inputs ($H_s, T_p, \mu_{wave}, \vec{V}_c, \vec{V}_w$).
  - Variable Selection Networks to quantify the exact relative importance of wind vs. waves vs. current on vessel performance.
  - Interpretable self-attention weights illustrating why specific sea-state encounters triggered severe predicted speed drops.

### 3.2 Candidate Model: Bidirectional Temporal Convolutional Network (TCN) / GRU with Self-Attention
- **Rationale**: Highly efficient inference latency ($<5\text{ ms}$ per evaluation) on commodity CPU hardware, zero recursive state degradation, and easy deployment to ONNX runtime.

### 3.3 Selection & Audit Criteria
The architecture is **NOT locked** prior to the training dataset audit. The final model choice will be dictated by:
1. Data availability (availability of open high-resolution AIS trajectories with aligned weather hindcasts).
2. Target definition accuracy (speed over ground degradation derived from AIS delta vs. calm-water baseline).
3. Compute constraints during production route evaluation (must evaluate hundreds of graph candidate edges within $< 2\text{ seconds}$).
4. Validation results and generalization metrics across multiple ship classes (Container, Tanker, Bulk Carrier).

---

## 4. Base Maritime Network (SeaRoute) & Multi-Objective Optimization

### 4.1 Base Maritime Graph
- The foundational topological network is derived from the **SeaRoute / Eurostat maritime network GeoJSON**.
- Waypoints and edges represent certified maritime lanes, TSS (Traffic Separation Schemes), and international transit corridors.
- Depth attributes prevent routing vessels with draft $T$ into waters shallower than $1.15 \times T$.

### 4.2 Multi-Objective Cost Function
For a voyage trajectory $\Pi = (e_1, e_2, \dots, e_k)$:
$$\text{Cost}(\Pi) = \sum_{e \in \Pi} \left[ w_t \cdot \Delta t(e) + w_f \cdot \text{EstimatedFuel}(e) + w_r \cdot \text{RiskPenalty}(e) \right]$$

Where:
- $\Delta t(e) = \frac{\text{Distance}(e)}{V_{pred}(e)}$: Segment duration dictated by the DL-predicted degraded speed.
- $\text{EstimatedFuel}(e) = P_B(e) \cdot \text{SFOC}(P_B(e)) \cdot \Delta t(e) \times 10^{-6}\text{ [MT]}$.
- $P_B(e)$: Brake power computed by naval architecture equations utilizing the DL-predicted resistance factor.
- $\text{RiskPenalty}(e)$: Heavy exponential penalty applied if $H_s > H_{s,threshold}$ (e.g. $> 5.5\text{ m}$) or wind $> 40\text{ knots}$.

---

## 5. Production API & Implementation Verification (Phase 3)

### 5.1 Endpoints
- `POST /api/v1/routing/optimize`: Accepts origin/destination coordinates, departure timestamp, vessel characteristics, fuel parameters, and optimization priority. Returns both `base_route` and `weather_aware_route` with complete segment breakdown, weather provenance, and comparison summary.
- `GET /api/v1/routing/defaults?vessel_type={type}`: Returns realistic standard naval dimensions, displacement, power, and fuel cost parameters for cargo, tanker, passenger, fishing, tug, service, or other vessels.

### 5.2 Mandatory Disclaimer
All responses include:
> *"All fuel, energy, and speed loss values are ESTIMATES based on naval architecture formulations (Holtrop-Mennen calm-water resistance, Kwon wave speed degradation, Blendermann wind coefficients, and IMO Fourth GHG Study SFOC curves) coupled with a Temporal Fusion Transformer surrogate trained on MARIS-Forecast NOAA Track A data. This software is not certified for nautical navigation, safety of life at sea (SOLAS), or statutory compliance."*

