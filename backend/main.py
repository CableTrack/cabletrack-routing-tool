"""
CableTrack routing tool prototype backend.

Wraps searoute-py for land-avoiding route generation, then layers on
segment-based speed zones, transit time, ETA allowance, and fuel/cost
estimation -- none of which searoute-py knows about, all of it is ours.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""
import json
import math
import os
import time

import httpx
import searoute as sr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="CableTrack Routing Tool Prototype")

# ---------------------------------------------------------------------------
# Port search data -- reuse searoute-py's own bundled port list (3,962 ports)
# rather than sourcing/maintaining a separate UN/LOCODE dataset.
# ---------------------------------------------------------------------------
_PORTS_PATH = os.path.join(os.path.dirname(sr.__file__), "data", "ports.geojson")
with open(_PORTS_PATH) as _f:
    _PORTS_RAW = json.load(_f)["features"]

PORTS = [
    {
        "name": feat["properties"].get("name"),
        "country": feat["properties"].get("cty"),
        "code": feat["properties"].get("port"),
        "lon": feat["geometry"]["coordinates"][0],
        "lat": feat["geometry"]["coordinates"][1],
    }
    for feat in _PORTS_RAW
    if feat["properties"].get("name")
]


def haversine_nm(a, b):
    """Great-circle distance in nautical miles between [lon, lat] points a and b."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1, math.sqrt(h)))
    earth_radius_nm = 3440.065
    return earth_radius_nm * c


# ---------------------------------------------------------------------------
# /api/ports/search -- autocomplete for the origin/destination search boxes
# ---------------------------------------------------------------------------
@app.get("/api/ports/search")
def search_ports(q: str = "", limit: int = 12):
    q = q.strip().lower()
    if len(q) < 2:
        return []
    matches = [p for p in PORTS if q in p["name"].lower()]
    # Prefer names that start with the query, then alphabetical
    matches.sort(key=lambda p: (not p["name"].lower().startswith(q), p["name"]))
    return matches[:limit]


# ---------------------------------------------------------------------------
# /api/route -- the land-avoiding route itself, via searoute-py
# ---------------------------------------------------------------------------
class RouteRequest(BaseModel):
    origin: list[float]       # [lon, lat]
    destination: list[float]  # [lon, lat]


@app.post("/api/route")
def get_route(req: RouteRequest):
    route = sr.searoute(req.origin, req.destination, units="naut", return_passages=True)
    return {
        "coordinates": route.geometry.coordinates,
        "distance_nm": route.properties["length"],
        "passages_used": route.properties.get("passages"),
    }


# ---------------------------------------------------------------------------
# /api/stats -- segment-based speed zones, transit time, fuel and cost.
# This is entirely our own logic; searoute-py has no involvement here.
# Called both right after /api/route, and again whenever the user has
# dragged/added waypoints on the map (recalculate button).
# ---------------------------------------------------------------------------
class StatsRequest(BaseModel):
    coordinates: list[list[float]]   # [[lon, lat], ...] -- current (possibly edited) route
    port_speed_kn: float = 5
    approach_speed_kn: float = 7
    open_sea_speed_kn: float = 12
    port_radius_nm: float = 3
    approach_radius_nm: float = 15

    # Distance/track allowance: real vessels don't sail the plotted polyline
    # exactly -- course corrections, drift, avoidance manoeuvres. Applied as
    # a straight percentage on top of the geometric route distance, same as
    # a "steaming margin" in a passage plan.
    distance_allowance_pct: float = 3

    # Current factor: NOT real current data (no ocean-current feed is wired
    # up here) -- a user-supplied assumption of net current effect on speed
    # over the passage. Positive = following current (faster), negative =
    # adverse current (slower). Applied uniformly across all speed zones,
    # which is a simplification; a real current-aware version would vary
    # this per segment from actual current data.
    current_factor_pct: float = 0

    # Weather allowance: NOT a live weather/routing feed -- a user-supplied
    # percentage added to transit time for assumed weather-related slowdown.
    weather_allowance_pct: float = 10

    fuel_burn_t_per_day: float = 8.5
    fuel_price_per_t: float = 620
    vessel_day_rate: float = 12000

    # Number of transits this route represents -- e.g. a cable install that
    # needs the vessel to make the same passage 3 times. Multiplies the
    # per-transit fuel/vessel cost; distance and per-transit timing figures
    # below are unaffected (they describe a single transit).
    trips: int = 1

    # A flat, user-supplied cost the tool has no way to derive itself
    # (permits, standby days, mobilisation, etc.) -- added once to the
    # total, not multiplied by trips. additional_cost_note is a short
    # free-text reason, capped defensively server-side (the frontend also
    # enforces this via maxlength on the input).
    additional_cost: float = 0
    additional_cost_note: str = ""


def classify_zone(dist_to_nearest_end_nm: float, port_radius: float, approach_radius: float):
    if dist_to_nearest_end_nm <= port_radius:
        return "port"
    if dist_to_nearest_end_nm <= approach_radius:
        return "approach"
    return "open_sea"


@app.post("/api/stats")
def get_stats(req: StatsRequest):
    coords = req.coordinates
    if len(coords) < 2:
        return {"error": "Need at least 2 coordinates"}

    ends = (coords[0], coords[-1])
    speeds = {
        "port": req.port_speed_kn,
        "approach": req.approach_speed_kn,
        "open_sea": req.open_sea_speed_kn,
    }
    # Current factor adjusts effective speed uniformly across zones (see
    # StatsRequest docstring above for the "this is an assumption, not real
    # current data" caveat). Guard against a factor extreme enough to zero
    # or invert a speed.
    current_multiplier = max(0.05, 1 + req.current_factor_pct / 100)
    effective_speeds = {k: v * current_multiplier for k, v in speeds.items()}

    distance_multiplier = 1 + req.distance_allowance_pct / 100

    zone_totals = {"port": 0.0, "approach": 0.0, "open_sea": 0.0}
    zone_hours = {"port": 0.0, "approach": 0.0, "open_sea": 0.0}
    total_nm = 0.0

    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        seg_nm = haversine_nm(p1, p2)
        midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        dist_to_nearest_end = min(haversine_nm(midpoint, ends[0]), haversine_nm(midpoint, ends[1]))
        zone = classify_zone(dist_to_nearest_end, req.port_radius_nm, req.approach_radius_nm)

        zone_totals[zone] += seg_nm
        zone_hours[zone] += (seg_nm * distance_multiplier) / effective_speeds[zone]
        total_nm += seg_nm

    effective_distance_nm = total_nm * distance_multiplier
    transit_hours = sum(zone_hours.values())
    planning_transit_hours = transit_hours * (1 + req.weather_allowance_pct / 100)
    planning_transit_days = planning_transit_hours / 24

    fuel_required_t = req.fuel_burn_t_per_day * planning_transit_days
    fuel_cost_per_trip = fuel_required_t * req.fuel_price_per_t
    vessel_cost_per_trip = req.vessel_day_rate * planning_transit_days
    cost_per_trip = fuel_cost_per_trip + vessel_cost_per_trip

    trips = max(1, int(req.trips or 1))
    fuel_cost = fuel_cost_per_trip * trips
    vessel_cost = vessel_cost_per_trip * trips

    additional_cost = req.additional_cost or 0
    additional_cost_note = (req.additional_cost_note or "")[:200]

    total_cost = (cost_per_trip * trips) + additional_cost

    def fmt_hm(hours):
        h = int(hours)
        m = round((hours - h) * 60)
        if m == 60:
            h += 1
            m = 0
        return f"{h}h {m:02d}m"

    return {
        "total_distance_nm": round(total_nm, 1),
        "distance_allowance_pct": req.distance_allowance_pct,
        "effective_distance_nm": round(effective_distance_nm, 1),
        "current_factor_pct": req.current_factor_pct,
        "zones": {
            zone: {
                "distance_nm": round(zone_totals[zone], 1),
                "speed_kn": speeds[zone],
                "effective_speed_kn": round(effective_speeds[zone], 2),
                "hours": round(zone_hours[zone], 2),
            }
            for zone in ("open_sea", "approach", "port")
        },
        "transit_hours": round(transit_hours, 2),
        "transit_hm": fmt_hm(transit_hours),
        "weather_allowance_pct": req.weather_allowance_pct,
        "planning_transit_hours": round(planning_transit_hours, 2),
        "planning_transit_hm": fmt_hm(planning_transit_hours),
        "planning_transit_days": round(planning_transit_days, 2),
        "fuel_burn_t_per_day": req.fuel_burn_t_per_day,
        "fuel_required_t": round(fuel_required_t, 2),
        "fuel_price_per_t": req.fuel_price_per_t,
        "fuel_cost": round(fuel_cost, 2),
        "vessel_day_rate": req.vessel_day_rate,
        "vessel_cost": round(vessel_cost, 2),
        "trips": trips,
        "cost_per_trip": round(cost_per_trip, 2),
        "additional_cost": round(additional_cost, 2),
        "additional_cost_note": additional_cost_note,
        "total_cost": round(total_cost, 2),
    }


# ---------------------------------------------------------------------------
# /api/tides -- tidal range lookup via the TideCheck API (tidecheck.com).
# Not routing data, not from searoute-py -- a separate third-party service
# used purely for initial planning context around a port. The API key stays
# server-side (env var) so it's never exposed in the page source. Free tier
# is 50 requests/day, so results are cached in-process for a few hours per
# rounded coordinate to avoid burning through the quota on repeat lookups
# of the same port.
# ---------------------------------------------------------------------------
TIDECHECK_API_KEY = os.environ.get("TIDECHECK_API_KEY", "")
TIDECHECK_BASE = "https://tidecheck.com/api"
_TIDE_CACHE_TTL_S = 6 * 60 * 60  # 6 hours
_tide_cache: dict[str, tuple[float, dict]] = {}


class TideRequest(BaseModel):
    lat: float
    lon: float


@app.post("/api/tides")
def get_tides(req: TideRequest):
    if not TIDECHECK_API_KEY:
        return {"error": "Tide lookups aren't configured on this server."}

    cache_key = f"{round(req.lat, 2)},{round(req.lon, 2)}"
    now = time.time()
    cached = _tide_cache.get(cache_key)
    if cached and (now - cached[0]) < _TIDE_CACHE_TTL_S:
        return cached[1]

    headers = {"X-API-Key": TIDECHECK_API_KEY}
    try:
        with httpx.Client(timeout=10) as client:
            nearest_resp = client.get(
                f"{TIDECHECK_BASE}/stations/nearest",
                params={"lat": req.lat, "lng": req.lon},
                headers=headers,
            )
            nearest_resp.raise_for_status()
            stations = nearest_resp.json()
            if not stations:
                return {"error": "No tide station found near this location."}
            station = stations[0]

            tides_resp = client.get(
                f"{TIDECHECK_BASE}/station/{station['id']}/tides",
                params={"days": 3, "datum": "LAT"},
                headers=headers,
            )
            tides_resp.raise_for_status()
            tide_data = tides_resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return {"error": "Tide data daily limit reached -- try again tomorrow."}
        if e.response.status_code == 401:
            return {"error": "Tide API key rejected -- check TIDECHECK_API_KEY."}
        return {"error": f"Tide lookup failed ({e.response.status_code})."}
    except httpx.HTTPError:
        return {"error": "Tide lookup failed -- couldn't reach TideCheck."}

    extremes = tide_data.get("extremes", [])
    heights = [e["height"] for e in extremes]

    result = {
        "station_name": station.get("name"),
        "station_region": station.get("region"),
        "station_country": station.get("country"),
        "station_distance_km": round(station.get("distanceKm", 0), 1),
        "tidal_range_m": round(max(heights) - min(heights), 2) if heights else None,
        "datum": tide_data.get("datum"),
        "next_extremes": extremes[:4],
    }
    _tide_cache[cache_key] = (now, result)
    return result


# ---------------------------------------------------------------------------
# /api/windfarms -- worldwide offshore wind farm locations via OpenStreetMap's
# Overpass API (overpass-api.de), free and keyless. Offshore wind farms are
# identified by the `seamark:production_area:category=wind_farm` tag, which
# marine charts use specifically for offshore production areas -- this keeps
# the result to a few hundred offshore sites worldwide rather than pulling in
# every onshore wind farm on the planet (tens of thousands).
#
# Status (operational / under construction / planned) is inferred from OSM's
# lifecycle tagging conventions: a `proposed:power`/`proposed:plant:source`
# prefix or a `seamark:production_area:condition` of "planned"/"proposed"
# means planned; a `construction:power` prefix or a condition mentioning
# "construction" means under construction; anything else is treated as
# operational. This is OSM community data, not an authoritative register --
# coverage of very recently announced projects will be incomplete.
#
# This must run server-side: Overpass does not send CORS headers permitting
# browser fetches from arbitrary origins (confirmed -- a direct client-side
# fetch from this app's own domain fails with a generic "Failed to fetch",
# while the same query works fine from a page already on overpass-api.de).
# Cached in-process for a day since offshore wind farm locations change
# rarely; this also avoids hammering the shared public Overpass instance on
# every page load.
# ---------------------------------------------------------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_WINDFARM_CACHE_TTL_S = 24 * 60 * 60  # 24 hours
_windfarm_cache: tuple[float, list] | None = None

_WINDFARM_QUERY = """
[out:json][timeout:60];
(
  way["seamark:production_area:category"="wind_farm"];
  relation["seamark:production_area:category"="wind_farm"];
  node["seamark:production_area:category"="wind_farm"];
);
out tags center 3000;
"""


def _classify_windfarm_status(tags: dict) -> str:
    condition = (tags.get("seamark:production_area:condition") or "").lower()
    if "proposed:power" in tags or "planned" in condition or "proposed" in condition:
        return "planned"
    if "construction:power" in tags or "construction" in condition:
        return "construction"
    return "operational"


@app.get("/api/windfarms")
def get_windfarms():
    global _windfarm_cache
    now = time.time()
    if _windfarm_cache and (now - _windfarm_cache[0]) < _WINDFARM_CACHE_TTL_S:
        return {"windfarms": _windfarm_cache[1], "cached": True}

    try:
        with httpx.Client(timeout=70) as client:
            resp = client.post(OVERPASS_URL, data={"data": _WINDFARM_QUERY})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError:
        # Serve stale cache rather than nothing if Overpass is unreachable.
        if _windfarm_cache:
            return {"windfarms": _windfarm_cache[1], "cached": True, "stale": True}
        return {"error": "Couldn't reach the windfarm data source (OpenStreetMap Overpass API). Try again shortly."}

    windfarms = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        center = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if center.get("lat") is None or center.get("lon") is None:
            continue
        windfarms.append({
            "id": el.get("id"),
            "lat": center["lat"],
            "lon": center["lon"],
            "name": tags.get("name") or tags.get("seamark:name") or "Unnamed wind farm",
            "operator": tags.get("operator"),
            "capacity": tags.get("plant:output:electricity") or tags.get("proposed:plant:output:electricity"),
            "status": _classify_windfarm_status(tags),
        })

    _windfarm_cache = (now, windfarms)
    return {"windfarms": windfarms, "cached": False}


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
