# ADR-005: Hybrid Navigational Graph and Hydrodynamic Weather Routing Engine

## Status
Accepted

## Context
Weather routing for ocean-going vessels requires calculating navigational routes that balance transit time, fuel consumption, and navigational safety against environmental stressors (waves, currents, winds). 

Pure geometric shortest path algorithms (e.g. great-circle / orthodromic paths) ignore landmass barriers, shallow bathymetry, Traffic Separation Schemes (TSS), and sea ice. Conversely, continuous 2D grid searches across the entire global ocean are computationally prohibitive on zero-budget commodity servers.

## Decision
We implement a **Two-Tier Hybrid Routing Engine**:
1. **Navigational Base Graph**: Routes are constrained to a validated global maritime network derived from open maritime passage datasets (Searoute maritime network / World Maritime Routes GeoJSON). Pathfinding across this graph uses Dijkstra / A* with Great-Circle edge weights to ensure valid open-water passage without land crossing.
2. **Dynamic Weather-Cost Function**: Along candidate edges, the route is discretized into voyage segments. For each segment, the engine samples spatiotemporally interpolated marine weather from Open-Meteo Marine (significant wave height $H_s$, wave peak period $T_p$, relative wave direction $\theta_w$, ocean current vector $\vec{V}_c$, and wind vector $\vec{V}_w$).
3. **Hydrodynamic Vessel Performance & Resistance Formulation**:
   - Total resistance $R_T$ is modeled using documented naval architecture formulations:
     $$R_T = R_{calm} + R_{aw} + R_{wind}$$
     where $R_{calm}$ is calm-water resistance (empirical Holtrop-Mennen formulation), $R_{aw}$ is added resistance in waves (modified Kwon / ISO 15016 empirical estimation), and $R_{wind}$ is aerodynamic superstructure drag.
   - Shaft power $P_B$ and Specific Fuel Oil Consumption (SFOC) curves yield fuel burn estimates per leg.
4. **Honest Estimation & Transparency**:
   - Fuel, power, and cost outputs are explicitly labeled: *"Estimated fuel consumption based on Kwon/Holtrop approximation model. Not certified for safety-critical navigation."*
   - Never generate fabricated fuel savings percentages or guarantee weather avoidance.

## Consequences
### Positive
- Computationally feasible execution (sub-2-second route evaluation on commodity CPU).
- Eliminates risk of proposing routes crossing land masses or hazardous shoals.
- Transparent, scientifically grounded formulas open to inspection and audit.

### Negative / Trade-offs
- Constrained to the fidelity of the maritime navigation network graph. In open oceans, waypoint deviation angles are limited to graph edges rather than arbitrary continuous isochrone curves unless graph density is increased.
