"""
Coastal clearance correction.

searoute-py's route is a shortest path over its own maritime network graph.
That graph avoids land, but its nodes can sit right at the coastline (we
measured some as close as 0.06 nm off North Wales) -- geometrically "not on
land", but not a route a real vessel would plan, which normally keeps a few
nm of sea room around headlands.

This module checks every interior point of a generated route against a
land mask, and where the route hugs the coast tighter than a target
clearance, re-routes just that stretch through a local buffered-grid A*
search that treats anything closer than the clearance distance as blocked
-- not just literal land. The rest of the route (anything already fine) is
left untouched.

Data: data/coastal_mask.npz is a pre-downsampled (1/30 degree, ~2nm cells),
bit-packed land/water grid derived from the `global_land_mask` package's
full-resolution (1/120 degree) raster. That package itself is NOT a runtime
dependency here -- its full-resolution array is ~890MB in memory, too heavy
for a small deployment. The bundled file is ~230KB on disk, ~60MB unpacked
in memory, generated once offline (see scripts/build_coastal_mask.py).
"""
import heapq
import math
import os

import numpy as np

NM_PER_DEG_LAT = 60.0
R_NM = 3440.065

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "coastal_mask.npz")
_d = np.load(_DATA_PATH)
_LAND = np.unpackbits(_d["packed"]).reshape(tuple(_d["shape"]))[: _d["shape"][0], : _d["shape"][1]].astype(bool)
_LAT0 = float(_d["lat0"][0])
_LON0 = float(_d["lon0"][0])
_RES_LAT = float(_d["res_lat"][0])   # negative: latitude decreases as row index increases
_RES_LON = float(_d["res_lon"][0])
_NLAT, _NLON = _LAND.shape


def _lat_to_row(lat):
    return int(round((lat - _LAT0) / _RES_LAT))


def _lon_to_col(lon):
    return int(round((lon - _LON0) / _RES_LON))


def _row_to_lat(row):
    return _LAT0 + row * _RES_LAT


def _col_to_lon(col):
    return _LON0 + col * _RES_LON


def haversine_nm(a, b):
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R_NM * math.asin(min(1, math.sqrt(h)))


def nearest_land_nm(lon, lat, search_radius_nm=20.0):
    """Distance in nm from (lon,lat) to the nearest land cell within
    search_radius_nm, or None if no land found in that range."""
    dlat = search_radius_nm / NM_PER_DEG_LAT
    dlon = search_radius_nm / (NM_PER_DEG_LAT * max(0.05, math.cos(math.radians(lat))))
    r0, r1 = sorted([_lat_to_row(lat - dlat), _lat_to_row(lat + dlat)])
    c0, c1 = sorted([_lon_to_col(lon - dlon), _lon_to_col(lon + dlon)])
    r0, r1 = max(0, r0), min(_NLAT - 1, r1)
    c0, c1 = max(0, c0), min(_NLON - 1, c1)
    patch = _LAND[r0:r1 + 1, c0:c1 + 1]
    idx = np.argwhere(patch)
    if idx.size == 0:
        return None
    lats = _row_to_lat(r0 + idx[:, 0])
    lons = _col_to_lon(c0 + idx[:, 1])
    lat1 = math.radians(lat)
    lat2 = np.radians(lats)
    dlat_r = lat2 - lat1
    dlon_r = np.radians(lons) - math.radians(lon)
    h = np.sin(dlat_r / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(dlon_r / 2) ** 2
    d = 2 * R_NM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    return float(np.min(d))


def regional_reroute(p_start, p_end, clearance_nm=3.0, margin_nm=25.0, target_cells=180):
    """Grid A* between p_start and p_end, treating land AND anything closer
    than clearance_nm to land as blocked. Returns [[lon,lat], ...] or None
    if no path exists within the search window."""
    from scipy.ndimage import distance_transform_edt

    lon0, lat0 = p_start
    lon1, lat1 = p_end
    mid_lat = (lat0 + lat1) / 2
    lat_min = min(lat0, lat1) - margin_nm / NM_PER_DEG_LAT
    lat_max = max(lat0, lat1) + margin_nm / NM_PER_DEG_LAT
    lon_margin = margin_nm / (NM_PER_DEG_LAT * max(0.05, math.cos(math.radians(mid_lat))))
    lon_min = min(lon0, lon1) - lon_margin
    lon_max = max(lon0, lon1) + lon_margin

    span = max(lat_max - lat_min, lon_max - lon_min)
    cell_deg = max(span / target_cells, abs(_RES_LAT))

    lats = np.arange(lat_min, lat_max, cell_deg)
    lons = np.arange(lon_min, lon_max, cell_deg)
    nlat, nlon = len(lats), len(lons)
    if nlat < 2 or nlon < 2:
        return None

    r_idx = np.clip(np.array([_lat_to_row(la) for la in lats]), 0, _NLAT - 1)
    c_idx = np.clip(np.array([_lon_to_col(lo) for lo in lons]), 0, _NLON - 1)
    land = _LAND[np.ix_(r_idx, c_idx)]

    cell_nm_lat = cell_deg * NM_PER_DEG_LAT
    cell_nm_lon = cell_deg * NM_PER_DEG_LAT * math.cos(math.radians(mid_lat))
    cell_nm = (cell_nm_lat + cell_nm_lon) / 2
    dist_nm = distance_transform_edt(~land) * cell_nm
    blocked = land | (dist_nm < clearance_nm)

    def to_cell(lon, lat):
        j = int(round((lon - lon_min) / cell_deg))
        i = int(round((lat - lat_min) / cell_deg))
        return max(0, min(nlat - 1, i)), max(0, min(nlon - 1, j))

    def nearest_unblocked(i, j):
        if not blocked[i, j]:
            return i, j
        for r in range(1, 40):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nlat and 0 <= nj < nlon and not blocked[ni, nj]:
                        return ni, nj
        return None

    start_cell = nearest_unblocked(*to_cell(lon0, lat0))
    goal_cell = nearest_unblocked(*to_cell(lon1, lat1))
    if start_cell is None or goal_cell is None:
        return None
    si, sj = start_cell
    gi, gj = goal_cell

    def h(i, j):
        return math.hypot((i - gi) * cell_nm_lat, (j - gj) * cell_nm_lon)

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    open_heap = [(h(si, sj), 0.0, (si, sj))]
    came_from = {}
    gscore = {(si, sj): 0.0}
    visited = set()
    found = False
    while open_heap:
        _, g, (ci, cj) = heapq.heappop(open_heap)
        if (ci, cj) in visited:
            continue
        visited.add((ci, cj))
        if (ci, cj) == (gi, gj):
            found = True
            break
        for di, dj in neighbors:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < nlat and 0 <= nj < nlon and not blocked[ni, nj] and (ni, nj) not in visited:
                step = math.hypot(di * cell_nm_lat, dj * cell_nm_lon)
                ng = g + step
                if ng < gscore.get((ni, nj), float("inf")):
                    gscore[(ni, nj)] = ng
                    came_from[(ni, nj)] = (ci, cj)
                    heapq.heappush(open_heap, (ng + h(ni, nj), ng, (ni, nj)))
    if not found:
        return None

    cells = [(gi, gj)]
    cur = (gi, gj)
    while cur != (si, sj):
        cur = came_from[cur]
        cells.append(cur)
    cells.reverse()

    path = [[float(lons[j]), float(lats[i])] for i, j in cells]
    path[0] = list(p_start)
    path[-1] = list(p_end)
    return path


def simplify_rdp(points, epsilon_nm=0.6):
    """Douglas-Peucker simplification using cross-track distance in nm."""
    if len(points) < 3:
        return points

    def perp_dist_nm(pt, a, b):
        if a == b:
            return haversine_nm(pt, a)
        lat0 = a[1]
        cos0 = math.cos(math.radians(lat0))

        def proj(p):
            return ((p[0] - a[0]) * cos0 * 60, (p[1] - a[1]) * 60)

        bx, by = proj(b)
        px, py = proj(pt)
        seg_len2 = bx * bx + by * by
        if seg_len2 == 0:
            return math.hypot(px, py)
        t = max(0, min(1, (px * bx + py * by) / seg_len2))
        cx, cy = t * bx, t * by
        return math.hypot(px - cx, py - cy)

    dmax, idx = 0, 0
    for i in range(1, len(points) - 1):
        d = perp_dist_nm(points[i], points[0], points[-1])
        if d > dmax:
            dmax, idx = d, i
    if dmax > epsilon_nm:
        left = simplify_rdp(points[: idx + 1], epsilon_nm)
        right = simplify_rdp(points[idx:], epsilon_nm)
        return left[:-1] + right
    return [points[0], points[-1]]


def apply_detour_shortcuts(
    coords,
    clearance_nm=3.0,
    min_saving_nm=15.0,
    min_saving_ratio=1.3,
    max_direct_nm=70.0,
    max_hops=4,
    max_passes=3,
):
    """Fix a different problem to coastal hugging: gaps in searoute-py's own
    maritime network graph. The network is a fixed set of nodes and edges --
    in places it simply has no edge directly linking two nodes that are
    close together and have clear water between them, forcing the shortest-
    path search onto a much longer detour through a distant hub node.

    We measured exactly this off St David's Head: the network has no edge
    from the Celtic Sea approach (~51.25N, 5.9W) to the St David's Head node
    (~51.16N, 4.5W) only ~53nm away -- the graph forces a detour via a hub
    node ~120nm south near the Isles of Scilly instead.

    This scans the route for stretches whose along-route distance is much
    longer than the direct distance between the same two points, and where
    a local A* search (the same engine used for coastal clearance, so the
    replacement still respects land and the clearance distance) finds a
    genuinely shorter path, splices that in. Bounded to `max_direct_nm` and
    `max_hops` (route points spanned) so this only ever fires on a single
    local network gap of the kind measured above, never on long-haul routes
    that are legitimately curving around a landmass over many points -- a
    local A* search over a span that large would either be meaningless or,
    where land blocks the direct line, come back no shorter than the
    original, but bounding the search space keeps this scoped to what it's
    actually meant to fix rather than relying on that alone."""
    coords = list(coords)
    patched = False
    for _ in range(max_passes):
        n = len(coords)
        if n < 3:
            break
        cum = [0.0]
        for k in range(1, n):
            cum.append(cum[-1] + haversine_nm(coords[k - 1], coords[k]))

        candidates = []  # (estimated_saving, i, j)
        for i in range(0, n - 2):
            for j in range(i + 2, min(n, i + 1 + max_hops)):
                direct = haversine_nm(coords[i], coords[j])
                if direct < 1e-6 or direct > max_direct_nm:
                    continue
                along = cum[j] - cum[i]
                if along - direct < min_saving_nm or along / direct < min_saving_ratio:
                    continue
                candidates.append((along - direct, i, j))

        if not candidates:
            break
        # Try candidates biggest-estimated-saving first. Most will be
        # rejected (a big apparent saving usually just means the direct
        # chord cuts across land, e.g. straight through Wales itself for
        # the route's own start/end pair) -- regional_reroute either fails
        # to find a path in that case, or finds one no shorter than the
        # original once it's actually forced around the obstacle, and the
        # savings check below discards it either way. Keep trying down the
        # list until one candidate is a genuine, land-respecting shortcut.
        candidates.sort(key=lambda c: -c[0])
        applied = False
        for _, i, j in candidates[:20]:
            along = cum[j] - cum[i]
            reroute = regional_reroute(coords[i], coords[j], clearance_nm=clearance_nm, margin_nm=40.0)
            if reroute is None:
                continue
            reroute_len = sum(haversine_nm(reroute[k], reroute[k + 1]) for k in range(len(reroute) - 1))
            if along - reroute_len < 5.0:
                # Not actually an improvement once routed properly (land
                # was probably in the way of the direct line) -- try the
                # next candidate instead of giving up altogether.
                continue
            simplified = simplify_rdp(reroute, epsilon_nm=0.6)
            coords[i:j + 1] = simplified
            patched = True
            applied = True
            break
        if not applied:
            break

    return coords, patched


def apply_coastal_clearance(coords, clearance_nm=3.0, check_radius_nm=20.0):
    """Walk a searoute-style [[lon,lat], ...] route and re-route any interior
    stretch that comes closer than clearance_nm to land. The route's actual
    endpoints (the ports themselves) are left untouched -- they're SUPPOSED
    to be close to shore. Returns (new_coords, patched: bool)."""
    if len(coords) < 3:
        return coords, False

    clearances = [None] + [nearest_land_nm(c[0], c[1], check_radius_nm) for c in coords[1:-1]] + [None]
    violating = [
        i for i in range(1, len(coords) - 1)
        if clearances[i] is not None and clearances[i] < clearance_nm
    ]
    if not violating:
        return coords, False

    # Group into contiguous runs
    runs = []
    run_start = violating[0]
    prev = violating[0]
    for i in violating[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((run_start, prev))
        run_start = i
        prev = i
    runs.append((run_start, prev))

    new_coords = list(coords)
    patched = False
    # Process runs back-to-front so earlier splices don't shift later indices
    for run_start, run_end in reversed(runs):
        anchor_before_idx = run_start - 1
        anchor_after_idx = run_end + 1
        p_before = new_coords[anchor_before_idx]
        p_after = new_coords[anchor_after_idx]

        reroute = regional_reroute(p_before, p_after, clearance_nm=clearance_nm)
        if reroute is None:
            continue  # leave this stretch as-is rather than break the route
        simplified = simplify_rdp(reroute, epsilon_nm=0.6)
        # Splice: replace [anchor_before_idx .. anchor_after_idx] with the
        # simplified reroute (which already starts/ends at those anchors).
        new_coords[anchor_before_idx:anchor_after_idx + 1] = simplified
        patched = True

    return new_coords, patched
