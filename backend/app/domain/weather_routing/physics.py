import math

from app.domain.weather_routing.models import (
    FuelParameters,
    MarineWeatherConditions,
    VesselCharacteristics,
)


def calculate_relative_angle(vessel_bearing_deg: float, direction_deg: float) -> float:
    """
    Compute relative encounter angle in degrees [-180, 180].
    0° = head sea/wind (coming from directly ahead)
    180° / -180° = following sea/wind (coming from directly astern)
    """
    diff = (direction_deg - vessel_bearing_deg) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def estimate_displacement(vessel: VesselCharacteristics) -> float:
    """Estimate vessel displacement in metric tonnes if not explicitly provided."""
    if vessel.displacement_t is not None:
        return vessel.displacement_t
    # Block coefficient C_b estimates by vessel type
    c_b_map = {
        "tanker": 0.82,
        "cargo": 0.72,
        "passenger": 0.60,
        "fishing": 0.55,
        "tug": 0.50,
        "service": 0.55,
        "other": 0.65,
    }
    c_b = c_b_map.get(vessel.vessel_type, 0.68)
    sea_water_density = 1.025  # tonnes/m^3
    return vessel.length_m * vessel.beam_m * vessel.draft_m * c_b * sea_water_density


def estimate_mcr_power_kw(vessel: VesselCharacteristics, displacement_t: float) -> float:
    """Estimate installed Maximum Continuous Rating (MCR) in kW if not provided."""
    if vessel.propulsion_power_kw is not None:
        return vessel.propulsion_power_kw
    # Admiralty coefficient formula: P = (Delta^(2/3) * V^3) / C_admiralty
    # Typical admiralty coefficients: 450-600
    v_kn = vessel.design_speed_kn
    c_adm = 520.0
    power_kw = (math.pow(displacement_t, 2.0 / 3.0) * math.pow(v_kn, 3.0)) / c_adm
    return max(power_kw, 250.0)


def calculate_segment_physics(
    vessel: VesselCharacteristics,
    fuel: FuelParameters,
    weather: MarineWeatherConditions,
    bearing_deg: float,
    predicted_speed_kn: float,
    duration_hours: float,
) -> dict[str, float]:
    """
    Transparent naval architecture and energy estimation layer.
    
    References:
    - Kwon (2008): Approximate method for predicting speed loss in irregular waves
    - Blendermann (1994): Estimation of wind resistance for ships
    - IMO MEPC (2018): 4th IMO GHG Study Engine Load and SFOC specifications
    
    Notice: All returned energy and fuel figures are strictly ESTIMATES.
    """
    disp_t = estimate_displacement(vessel)
    mcr_kw = estimate_mcr_power_kw(vessel, disp_t)
    v_ms = predicted_speed_kn * 0.514444

    # 1. Calm-water Resistance R_calm (kN)
    # R_calm approx from calm power P_calm = R * V
    v_ratio = max(predicted_speed_kn / max(vessel.design_speed_kn, 1.0), 0.1)
    base_calm_power_kw = mcr_kw * 0.80 * math.pow(v_ratio, 3.0)  # Standard 80% MCR design point
    r_calm_kn = base_calm_power_kw / max(v_ms, 0.5)

    # 2. Added Resistance from Waves R_aw (kN) - Kwon (2008) formulation
    # Relative wave encounter angle: 0° head, 90° beam, 180° following
    rel_wave_deg = abs(calculate_relative_angle(bearing_deg, weather.wave_direction_deg))
    cos_wave = math.cos(math.radians(rel_wave_deg))
    
    # Head seas increase resistance (+), following seas reduce added resistance
    wave_angle_factor = 0.5 * (1.0 + cos_wave)  # 1.0 for head sea, 0.0 for following sea
    # Wave resistance scales with square of wave height Hs^2
    wave_coef = 0.045 * (disp_t / 1000.0) ** (2.0 / 3.0)
    r_wave_kn = wave_coef * (weather.wave_height_m ** 2.0) * wave_angle_factor

    # 3. Aerodynamic Wind Resistance R_wind (kN) - Blendermann formulation
    rel_wind_deg = abs(calculate_relative_angle(bearing_deg, weather.wind_direction_deg))
    cos_wind = math.cos(math.radians(rel_wind_deg))
    
    air_density = 1.225  # kg/m^3
    # Frontal projected area approximation: A_T approx Beam * (Freeboard + Deckhouse)
    frontal_area_m2 = vessel.beam_m * (vessel.draft_m * 0.8 + 8.0)
    # Relative wind speed along ship axis
    v_wind_rel_ms = weather.wind_speed_mps * cos_wind + v_ms
    c_da = 0.85  # Typical drag coefficient
    r_wind_kn = 0.5 * air_density * c_da * frontal_area_m2 * (v_wind_rel_ms ** 2.0) / 1000.0
    r_wind_kn = max(r_wind_kn, 0.0)

    # 4. Total Resistance (kN)
    r_total_kn = r_calm_kn + r_wave_kn + r_wind_kn

    # 5. Effective Power (P_E) and Brake Power (P_B)
    eta_prop = max(vessel.propulsive_efficiency, 0.4)
    eta_trans = 0.98  # Shaft transmission efficiency
    power_kw = (r_total_kn * v_ms) / (eta_prop * eta_trans)
    # Cap power at installed MCR
    power_kw = min(max(power_kw, 50.0), mcr_kw * 1.05)

    # 6. Specific Fuel Oil Consumption (SFOC) Load Curve (IMO 4th GHG Study)
    # Engines operate at peak efficiency around 70-85% MCR (~165-175 g/kWh for VLSFO)
    load_factor = max(min(power_kw / mcr_kw, 1.0), 0.15)
    
    # Base SFOC by fuel type at 80% MCR (g/kWh)
    base_sfoc_map = {
        "VLSFO": 175.0,
        "MGO": 165.0,
        "HFO": 185.0,
        "LNG": 140.0,
        "METHANOL": 330.0,
    }
    base_sfoc = base_sfoc_map.get(fuel.fuel_type, 175.0)
    # Quadratic correction for partial load penalty
    sfoc_load_penalty = 1.0 + 0.25 * math.pow(load_factor - 0.75, 2.0)
    sfoc_g_per_kwh = base_sfoc * sfoc_load_penalty

    # 7. Estimated Energy (kWh), Fuel (Tonnes), and Cost (USD)
    energy_kwh = power_kw * duration_hours
    fuel_tonnes = (energy_kwh * sfoc_g_per_kwh) * 1e-6
    fuel_cost_usd = fuel_tonnes * fuel.fuel_price_per_tonne

    # Current speed contribution along heading
    rel_curr_deg = calculate_relative_angle(bearing_deg, weather.ocean_current_direction_deg)
    # Current moving with ship adds to speed over ground, against opposes
    curr_kn = weather.ocean_current_velocity_mps * 1.94384
    curr_contrib_kn = curr_kn * math.cos(math.radians(rel_curr_deg))

    return {
        "r_calm_kn": round(r_calm_kn, 2),
        "r_wave_kn": round(r_wave_kn, 2),
        "r_wind_kn": round(r_wind_kn, 2),
        "r_total_kn": round(r_total_kn, 2),
        "brake_power_kw": round(power_kw, 1),
        "energy_kwh": round(energy_kwh, 1),
        "fuel_tonnes": round(fuel_tonnes, 4),
        "fuel_cost_usd": round(fuel_cost_usd, 2),
        "sfoc_g_per_kwh": round(sfoc_g_per_kwh, 1),
        "current_speed_contribution_kn": round(curr_contrib_kn, 2),
    }
