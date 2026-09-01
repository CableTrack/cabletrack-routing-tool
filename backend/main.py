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
import uuid

import httpx
import searoute as sr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import coastal

app = FastAPI(title="CableTrack Routing Tool Prototype")

# Scoped to the main CableTrack site only, GET only -- lets the route-planner
# interstitial page on cabletrack.co.uk call /api/ports/search cross-origin
# (both to run the port autocomplete and as a wake-up ping for the free-tier
# cold start) before handing off to this tool. Not a wildcard/open policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cabletrack.co.uk"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

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
    min_clearance_nm: float = 3.0   # 0 disables the coastal-clearance pass below


@app.post("/api/route")
def get_route(req: RouteRequest):
    route = sr.searoute(req.origin, req.destination, units="naut", return_passages=True)
    coords = route.geometry.coordinates

    # searoute-py's network nodes can sit right at the coastline -- measured
    # as close as 0.06 nm off North Wales, well inside what a real vessel
    # would plan around a headland. Re-route any interior stretch tighter
    # than min_clearance_nm through a local buffered search; anything that
    # was already fine is left untouched. See coastal.py for the full
    # writeup (including why this is a bundled precomputed land grid rather
    # than a runtime dependency on the raw dataset -- that one's ~890MB in
    # RAM, more than a free-tier instance has to spare).
    coastal_hugging_fixed = False
    network_gap_fixed = False
    if req.min_clearance_nm > 0:
        coords, coastal_hugging_fixed = coastal.apply_coastal_clearance(
            coords, clearance_nm=req.min_clearance_nm
        )

        # A different problem to coastal hugging: searoute-py's network is a
        # fixed graph, and in places it's missing an edge that would let two
        # nearby, clear-water nodes connect directly -- forcing the shortest
        # path onto a long detour via a distant hub instead. Measured off
        # St David's Head: no edge from the Celtic Sea approach to the node
        # right off the headland, ~53nm apart, so the route detoured ~120nm
        # south via a hub near the Isles of Scilly and back. This looks for
        # that specific pattern (a route stretch far longer than the direct,
        # land-respecting distance between its own endpoints) and shortcuts
        # it through the same local search used above. See coastal.py.
        coords, network_gap_fixed = coastal.apply_detour_shortcuts(
            coords, clearance_nm=req.min_clearance_nm
        )

    # searoute-py's own reported length is for its original (uncorrected)
    # path -- recompute when either pass actually changed the geometry.
    if coastal_hugging_fixed or network_gap_fixed:
        distance_nm = sum(
            coastal.haversine_nm(coords[i], coords[i + 1]) for i in range(len(coords) - 1)
        )
    else:
        distance_nm = route.properties["length"]

    return {
        "coordinates": coords,
        "distance_nm": distance_nm,
        # searoute-py's own key here is "traversed_passages", not "passages" --
        # this was silently returning None before, so the "Route uses: ..."
        # note never actually showed anything.
        "passages_used": route.properties.get("traversed_passages"),
        "coastal_hugging_fixed": coastal_hugging_fixed,
        "network_gap_fixed": network_gap_fixed,
    }


# ---------------------------------------------------------------------------
# /api/barriers -- "what if this passage were closed" testing tool.
#
# searoute-py's own `restrictions` param only lets you name one of a fixed
# list of major chokepoints (suez, panama, gibraltar, etc) -- there's no
# support for an arbitrary custom closure. But the underlying maritime
# network it routes over (`sr.setup_M()`) is just a plain graph of
# (lon, lat) nodes and weighted edges, so we can block a stretch ourselves:
# find every edge whose segment crosses a barrier line the user draws, and
# remove it from the graph so Dijkstra can never use it. Restoring a
# barrier re-adds the original edges with their original attributes.
#
# Note: setting an edge's weight to float('inf') instead of removing it
# does NOT reliably work here -- networkx's bidirectional_dijkstra (which
# Marnet.shortest_path uses) doesn't always respect an infinite weight at
# the point where the two search fronts meet when a custom weight function
# is supplied, so the "blocked" edge could still end up in the returned
# path with the graph's original finite length. Confirmed by direct testing
# against the installed searoute==1.6.0. Removing the edge outright sidesteps
# this and forces a genuine, correctly-costed detour.
#
# This mutates the single shared graph instance that every /api/route call
# uses (searoute-py caches it process-wide via lru_cache), so a barrier is
# effectively global to this server -- fine for a single-user test/demo
# tool, but note it isn't per-session and won't survive a server restart.
#
# IMPORTANT: sr.setup_M must be called positionally (sr.setup_M("networkx")),
# never as a keyword (sr.setup_M(backend="networkx")) -- functools.lru_cache
# treats those as two different cache keys, so a keyword call here would
# build and mutate an entirely separate, unused Marnet graph instead of the
# one /api/route actually reads (this was the original bug that made the
# first version of this feature silently do nothing).
# ---------------------------------------------------------------------------
_active_barriers = {}  # id -> {"start": [lon,lat], "end": [lon,lat], "edges": [(u, v, orig_edge_data_dict), ...]}


def _orient(a, b, c):
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if val > 1e-12:
        return 1
    if val < -1e-12:
        return -1
    return 0


def _on_segment(a, b, c):
    return min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and \
        min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9


def _segments_intersect(p1, p2, p3, p4):
    o1, o2 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    o3, o4 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, p3):
        return True
    if o2 == 0 and _on_segment(p1, p2, p4):
        return True
    if o3 == 0 and _on_segment(p3, p4, p1):
        return True
    if o4 == 0 and _on_segment(p3, p4, p2):
        return True
    return False


def _bbox_overlap(a1, a2, b1, b2, margin=0.02):
    return not (
        max(a1[0], a2[0]) + margin < min(b1[0], b2[0]) or
        max(b1[0], b2[0]) < min(a1[0], a2[0]) - margin or
        max(a1[1], a2[1]) + margin < min(b1[1], b2[1]) or
        max(b1[1], b2[1]) < min(a1[1], a2[1]) - margin
    )


class BarrierRequest(BaseModel):
    start: list[float]  # [lon, lat]
    end: list[float]    # [lon, lat]


@app.post("/api/barriers")
def add_barrier(req: BarrierRequest):
    M = sr.setup_M("networkx")
    # Collect crossing edges first -- mutating M.edges() while iterating it
    # is unsafe, and remove_edge() needs the full attribute dict up front
    # so we can restore it exactly on delete.
    to_remove = []
    for u, v, data in M.edges(data=True):
        if not _bbox_overlap(u, v, req.start, req.end):
            continue
        if _segments_intersect(u, v, req.start, req.end):
            to_remove.append((u, v, dict(data)))
    for u, v, _ in to_remove:
        M.remove_edge(u, v)
    barrier_id = str(uuid.uuid4())[:8]
    _active_barriers[barrier_id] = {"start": req.start, "end": req.end, "edges": to_remove}
    return {"id": barrier_id, "blocked_edges": len(to_remove)}


@app.get("/api/barriers")
def list_barriers():
    return [
        {"id": bid, "start": b["start"], "end": b["end"], "blocked_edges": len(b["edges"])}
        for bid, b in _active_barriers.items()
    ]


@app.delete("/api/barriers/{barrier_id}")
def remove_barrier(barrier_id: str):
    barrier = _active_barriers.pop(barrier_id, None)
    if not barrier:
        return {"error": "Barrier not found"}
    M = sr.setup_M("networkx")
    restored = 0
    for u, v, orig_data in barrier["edges"]:
        if not M.has_edge(u, v):
            M.add_edge(u, v, **orig_data)
            restored += 1
    return {"restored_edges": restored}


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
# /api/windfarms -- worldwide offshore wind farm locations, originally sourced
# from OpenStreetMap's Overpass API (identified via the marine-chart tag
# `seamark:production_area:category=wind_farm`, which keeps the result to a
# few hundred offshore sites worldwide rather than every onshore wind farm on
# the planet). Status (operational / under construction / planned) comes from
# OSM's lifecycle tagging: a `proposed:power` prefix or a
# `seamark:production_area:condition` of "planned"/"proposed" means planned;
# a `construction:power` prefix or a condition mentioning "construction"
# means under construction; anything else is operational.
#
# Served from a bundled static snapshot (windfarms_data.json) rather than a
# live Overpass call: Render's outbound network cannot reach overpass-api.de
# at all (confirmed -- IPv6-only route gives "Network unreachable"; forcing
# IPv4 DNS then gets "Connection refused", consistent with Overpass
# blocklisting cloud/datacenter IP ranges to protect its free,
# volunteer-funded service from exactly this kind of server-to-server bulk
# use). The snapshot was captured from a browser session (not subject to
# that block) and committed to the repo -- this is OSM community data
# already described as non-authoritative, so a periodically-refreshed
# snapshot (replace windfarms_data.json and redeploy to update it) is an
# honest tradeoff, not a regression. A one-off /api/windfarms/_seed endpoint
# was used to extract the data server-side for this commit and has since
# been removed; recreate it the same way if a refresh is needed later.
# ---------------------------------------------------------------------------
_WINDFARM_DATA_PATH = os.path.join(os.path.dirname(__file__), "windfarms_data.json")


@app.get("/api/windfarms")
def get_windfarms():
    try:
        with open(_WINDFARM_DATA_PATH) as f:
            windfarms = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"error": "Windfarm data hasn't been loaded on this server yet.", "windfarms": []}
    return {"windfarms": windfarms, "cached": True}


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
