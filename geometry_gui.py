#!/usr/bin/env python3
"""
Interactive 4-View 3D Geometry Builder for Ansys Lumerical (.lsf), GDSII and gdsfactory
=======================================================================================
Opens a window divided into four quadrants:
  - Top-Left     : Top view (x-y)
  - Bottom-Left  : Side view / Cross section (x-z)
  - Top-Right    : 3D view (x-y-z, interactive rotation & zoom)
  - Bottom-Right : Front view / Cross section (y-z)

Features:
  - Draw connected polygon points with the mouse pointer in each 2D view
    (Top x-y, Side x-z, Front y-z).
  - Modify points at any time after drawing:
      * Drag any vertex with the left mouse button to move it.
      * Click and drag on any polygon edge midpoint (or Shift+click an edge)
        to insert a new vertex between connected points.
      * Right-click (or Ctrl+click / Delete key) on a vertex to remove it.
      * Edit exact numeric coordinates of the selected vertex in the toolbar.
  - "Generate geometry" button: Reconstructs the 3D solid from the three
    orthographic views and displays it in the 3D view.
  - "Generate file" button: Exports the drawn and reconstructed 3D geometry as
      * an Ansys Lumerical script (.lsf),
      * a GDSII layout (.gds) of the top-view (x-y) outline, written with
        klayout.db (the engine used by gdsfactory) or, if that is not
        installed, a built-in pure-Python GDSII writer,
      * a gdsfactory Python script (.py) containing an @gf.cell component and
        a LayerStack (thickness / z position); running it writes the .gds
        (usable with gdsfactory and gdsfactory+),
      * or all three at once.

Headless use (no GUI):
    python geometry_gui.py --export-preset out.lsf out.gds out.py
"""

import argparse
import datetime
import keyword
import math
import os
import re
import struct
import sys
from dataclasses import dataclass

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ======================================================================
# Geometry core (GUI independent: importable and testable without Tk)
# All coordinates are in micrometers.
# ======================================================================
X_MERGE_TOL_UM = 1e-4  # x-vertices closer than this (0.1 nm, below the 1 nm GDS grid) form one station
GDS_DBU_UM = 1e-3  # GDS database unit: 1 nm
DEFAULT_GDS_LAYER = (1, 0)
DEFAULT_CELL_NAME = "InP_waveguide"
DEFAULT_MATERIAL = "InP"

SLIDE1_PRESET = {
    # Top view (x-y): 12-vertex symmetric two-stage tapered InP waveguide
    "xy": [
        [-300.0, -0.05],
        [-110.0, -0.25],
        [-100.0, -0.75],
        [100.0, -0.75],
        [110.0, -0.25],
        [300.0, -0.05],
        [300.0, 0.05],
        [110.0, 0.25],
        [100.0, 0.75],
        [-100.0, 0.75],
        [-110.0, 0.25],
        [-300.0, 0.05],
    ],
    # Side view (x-z): 600 um long, 250 nm (0.25 um) thick
    "xz": [[-300.0, -0.125], [300.0, -0.125], [300.0, 0.125], [-300.0, 0.125]],
    # Front view (y-z): 1.5 um wide, 250 nm (0.25 um) thick
    "yz": [[-0.75, -0.125], [0.75, -0.125], [0.75, 0.125], [-0.75, 0.125]],
}


class GeometryError(ValueError):
    """Raised when the drawn views cannot be turned into a valid geometry/file."""


@dataclass
class Geometry3D:
    """Result of the 3-view reconstruction."""

    xy_poly: list  # [[x, y], ...] top-view outline
    segments: list  # longitudinal sections (dicts with x0, x1, y*_bot/top, z*_bot/top)
    faces: list  # [(vertices, kind), ...] for the 3D preview
    bbox: tuple  # (xmin, xmax, ymin, ymax, zmin, zmax)

    @property
    def z_span(self):
        return self.bbox[4], self.bbox[5]

    @property
    def z_is_uniform(self):
        zs = {(s["z0_bot"], s["z0_top"], s["z1_bot"], s["z1_top"]) for s in self.segments}
        zmin, zmax = self.z_span
        return zs == {(zmin, zmax, zmin, zmax)}


def fmt_num(value):
    """Formats a number for generated scripts without losing precision (no '1234.57' for 1234.5678)."""
    v = round(float(value), 9)
    if v == 0:
        v = 0.0  # avoid "-0"
    return f"{v:.12g}"


def polygon_area(pts):
    n = len(pts)
    return 0.5 * sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)
    )


def vertical_slice_bounds(poly_pts, u_query):
    """Finds [v_min, v_max] of a 2D polygon at horizontal coordinate u_query."""
    n = len(poly_pts)
    v_hits = []
    tol = 1e-7
    for i in range(n):
        u1, v1 = poly_pts[i]
        u2, v2 = poly_pts[(i + 1) % n]
        if abs(u1 - u_query) <= tol:
            v_hits.append(v1)
        if abs(u2 - u_query) <= tol:
            v_hits.append(v2)
        if (u1 - u_query) * (u2 - u_query) < 0:
            t = (u_query - u1) / (u2 - u1)
            v_hits.append(v1 + t * (v2 - v1))
    if not v_hits:
        # Outside the polygon's u-range: fall back to its full v-extent
        vs = [v for _, v in poly_pts]
        return min(vs), max(vs)
    return min(v_hits), max(v_hits)


def interval_slice_bounds(poly_pts, a, b):
    """
    Lower/upper v-bounds of a polygon at both ends of an interval [a, b] that
    contains no polygon vertex strictly inside. The edges spanning the interval
    are straight there, so they are evaluated exactly at a and b (no sampling
    offset, which would distort steep tapers).
    Returns ((v_lo(a), v_hi(a)), (v_lo(b), v_hi(b))).
    """
    tol = 1e-9
    lo = hi = None
    n = len(poly_pts)
    for i in range(n):
        u1, v1 = poly_pts[i]
        u2, v2 = poly_pts[(i + 1) % n]
        if abs(u2 - u1) < tol or min(u1, u2) > a + tol or max(u1, u2) < b - tol:
            continue
        slope = (v2 - v1) / (u2 - u1)
        va, vb = v1 + (a - u1) * slope, v1 + (b - u1) * slope
        mid = va + vb
        if lo is None or mid < lo[0]:
            lo = (mid, va, vb)
        if hi is None or mid > hi[0]:
            hi = (mid, va, vb)
    if lo is None:  # interval outside the polygon: fall back to sampling
        return vertical_slice_bounds(poly_pts, a), vertical_slice_bounds(poly_pts, b)
    return (lo[1], hi[1]), (lo[2], hi[2])


def reconstruct_geometry(xy_pts, xz_pts, yz_pts):
    """
    Combines the Top (x-y), Side (x-z) and Front (y-z) polygons into 3D
    longitudinal segments (one per pair of neighbouring x-vertices of the top
    view) and preview faces. Raises GeometryError for incomplete/degenerate input.
    """
    missing = [
        name
        for name, pts in (
            ("Top view (x-y)", xy_pts),
            ("Side view (x-z)", xz_pts),
            ("Front view (y-z)", yz_pts),
        )
        if len(pts) < 3
    ]
    if missing:
        raise GeometryError(
            "Please draw connected polygons (at least 3 points) in all three 2D views "
            "before generating the 3D geometry.\n\nCurrently missing ≥3 points in:\n  • "
            + "\n  • ".join(missing)
            + "\n\nTip: You can also click 'Load Slide 1 Preset' to load the example geometry."
        )
    if abs(polygon_area(xy_pts)) < 1e-9:
        raise GeometryError("The Top view (x-y) polygon has zero area (all points on one line).")

    # 1. z bounds from Side (x-z) and Front (y-z)
    xz_zs = [p[1] for p in xz_pts]
    yz_zs = [p[1] for p in yz_pts]
    z_min_global = 0.5 * (min(xz_zs) + min(yz_zs))
    z_max_global = 0.5 * (max(xz_zs) + max(yz_zs))
    if abs(z_max_global - z_min_global) < 1e-6:
        z_min_global, z_max_global = -0.125, 0.125

    # 2. Unique sorted x stations of the Top view. Only (near-)coincident
    #    x-values are merged so that short features are not distorted.
    x_stations = []
    for x_val in sorted(p[0] for p in xy_pts):
        if not x_stations or x_val - x_stations[-1] > X_MERGE_TOL_UM:
            x_stations.append(x_val)
    if len(x_stations) < 2:
        raise GeometryError("The Top view (x-y) polygon has no extent along x.")

    # Map the x-z polygon's x-range onto the top view so local z(x) variations are respected
    xz_xmin = min(p[0] for p in xz_pts)
    xz_xmax = max(p[0] for p in xz_pts)
    xy_xmin, xy_xmax = x_stations[0], x_stations[-1]
    xz_span = max(xz_zs) - min(xz_zs)
    yz_span = max(yz_zs) - min(yz_zs)

    def z_bounds_at_x(x_val):
        if xz_xmax - xz_xmin > 1e-6:
            u_mapped = xz_xmin + (x_val - xy_xmin) / (xy_xmax - xy_xmin) * (xz_xmax - xz_xmin)
            zb, zt = vertical_slice_bounds(xz_pts, u_mapped)
            # Scale by the Front view z-span if the user modified it
            if xz_span > 1e-6 and yz_span > 1e-6:
                z_center = 0.5 * (zb + zt)
                half_h = 0.5 * (zt - zb) * (yz_span / xz_span)
                return z_center - half_h, z_center + half_h
            return zb, zt
        return z_min_global, z_max_global

    segments = []
    faces_3d = []
    for x0, x1 in zip(x_stations[:-1], x_stations[1:]):
        (y0_bot, y0_top), (y1_bot, y1_top) = interval_slice_bounds(xy_pts, x0, x1)
        # The side view has its own vertices, so it is sampled just inside the section
        eps = min(1e-4, 0.01 * (x1 - x0))
        z0_bot, z0_top = z_bounds_at_x(x0 + eps)
        z1_bot, z1_top = z_bounds_at_x(x1 - eps)

        seg = {
            k: round(v, 6)
            for k, v in (
                ("x0", x0), ("x1", x1),
                ("y0_bot", y0_bot), ("y0_top", y0_top),
                ("y1_bot", y1_bot), ("y1_top", y1_top),
                ("z0_bot", z0_bot), ("z0_top", z0_top),
                ("z1_bot", z1_bot), ("z1_top", z1_top),
            )
        }
        segments.append(seg)

        # 8 corners of the hexahedral segment
        b0 = (x0, y0_bot, z0_bot)
        b1 = (x1, y1_bot, z1_bot)
        b2 = (x1, y1_top, z1_bot)
        b3 = (x0, y0_top, z0_bot)
        t0 = (x0, y0_bot, z0_top)
        t1 = (x1, y1_bot, z1_top)
        t2 = (x1, y1_top, z1_top)
        t3 = (x0, y0_top, z0_top)
        faces_3d.append(([b0, b1, b2, b3], "bottom"))
        faces_3d.append(([t0, t1, t2, t3], "top"))
        faces_3d.append(([b0, b1, t1, t0], "side"))
        faces_3d.append(([b3, b2, t2, t3], "side"))
        faces_3d.append(([b0, b3, t3, t0], "end"))
        faces_3d.append(([b1, b2, t2, t1], "end"))

    all_v = [v for f, _ in faces_3d for v in f]
    bbox = tuple(
        round(fn(v[axis] for v in all_v), 6) for axis in range(3) for fn in (min, max)
    )
    if bbox[5] - bbox[4] < 1e-6:
        raise GeometryError("The reconstructed structure has zero thickness along z.")
    return Geometry3D(
        xy_poly=[[float(x), float(y)] for x, y in xy_pts],
        segments=segments,
        faces=faces_3d,
        bbox=bbox,
    )


# ----------------------------------------------------------------------
# Ansys Lumerical (.lsf)
# ----------------------------------------------------------------------
def build_lsf(geom):
    """Builds the Ansys Lumerical .lsf script string from a reconstructed geometry."""
    f = fmt_num
    z_min, z_max = geom.z_span
    t_z_nm = (z_max - z_min) * 1000.0
    z_center_um = 0.5 * (z_min + z_max)

    lines = [
        "# =========================================================================",
        "# Ansys Lumerical Script (.lsf)",
        "# 3D Geometry Generated from Interactive Multi-View Geometry Builder",
        "#",
        "# Coordinate System:",
        "#   x : Propagation / longitudinal direction",
        "#   y : Lateral / width direction (in-plane)",
        "#   z : Vertical / thickness direction (out-of-plane)",
        "# =========================================================================",
        "",
        "clear;",
        "deleteall;",
        "",
        "# -------------------------------------------------------------------------",
        "# 1. Units & Global Parameters (SI units: meters)",
        "# -------------------------------------------------------------------------",
        "um = 1e-6;",
        "nm = 1e-9;",
        "",
        f"t_z      = {f(t_z_nm)} * nm;   # Layer thickness ({f(z_max - z_min)} um)",
        f"z_center = {f(z_center_um)} * um;   # Center z-coordinate",
        "",
        "# Material configuration",
        'mat_name = "InP - Palik";',
        "if (!materialexists(mat_name)) {",
        '    if (materialexists("InP")) {',
        '        mat_name = "InP";',
        "    } else {",
        '        mat_name = addmaterial("(n,k) Material");',
        '        setmaterial(mat_name, "name", "InP");',
        '        setmaterial("InP", "Refractive Index", 3.17);',
        '        mat_name = "InP";',
        "    }",
        "}",
        "",
        "# Toggle: 0 = Individual longitudinal sections, 1 = Unified extruded polygon",
        "use_single_polygon = 0;",
        "",
        "# -------------------------------------------------------------------------",
        "# 2. Build 3D Geometry inside 'InP_Structure' Group",
        "# -------------------------------------------------------------------------",
        "addstructuregroup;",
        'set("name", "InP_Structure");',
        'set("x", 0);',
        'set("y", 0);',
        'set("z", 0);',
        "",
        "if (use_single_polygon == 1) {",
        "    V_all = [",
    ]
    n_poly = len(geom.xy_poly)
    for idx, (vx, vy) in enumerate(geom.xy_poly):
        sep = ";" if idx < n_poly - 1 else ""
        lines.append(f"        {f(vx)} * um, {f(vy)} * um{sep}")
    lines.extend(
        [
            "    ];",
            "    addpoly;",
            '    set("name", "InP_waveguide_full");',
            '    set("x", 0);',
            '    set("y", 0);',
            '    set("z", z_center);',
            '    set("z span", t_z);',
            '    set("vertices", V_all);',
            '    set("material", mat_name);',
            '    addtogroup("InP_Structure");',
            "} else {",
        ]
    )

    for idx, seg in enumerate(geom.segments, start=1):
        x0, x1 = seg["x0"], seg["x1"]
        y0_b, y0_t = seg["y0_bot"], seg["y0_top"]
        y1_b, y1_t = seg["y1_bot"], seg["y1_top"]
        seg_len = x1 - x0
        w0 = y0_t - y0_b
        w1 = y1_t - y1_b
        seg_z_c = 0.25 * (seg["z0_bot"] + seg["z0_top"] + seg["z1_bot"] + seg["z1_top"])
        seg_z_span_nm = 0.5 * ((seg["z0_top"] - seg["z0_bot"]) + (seg["z1_top"] - seg["z1_bot"])) * 1000.0
        is_rect = abs(y0_b - y1_b) < 1e-6 and abs(y0_t - y1_t) < 1e-6
        # Short, readable object name (lengths rounded to 1 nm)
        len_tag = f"{round(seg_len, 3):g}".replace(".", "p")

        if is_rect:
            lines.extend(
                [
                    f"    # Section ({idx}): Straight waveguide x=[{f(x0)}, {f(x1)}] um, width={f(w0)} um",
                    "    addrect;",
                    f'    set("name", "section_{idx}_straight_{len_tag}um");',
                    f'    set("x", {f(0.5 * (x0 + x1))} * um);',
                    f'    set("x span", {f(seg_len)} * um);',
                    f'    set("y", {f(0.5 * (y0_b + y0_t))} * um);',
                    f'    set("y span", {f(w0)} * um);',
                    f'    set("z", {f(seg_z_c)} * um);',
                    f'    set("z span", {f(seg_z_span_nm)} * nm);',
                    '    set("material", mat_name);',
                    '    addtogroup("InP_Structure");',
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"    # Section ({idx}): Taper x=[{f(x0)}, {f(x1)}] um, width={f(w0)} um -> {f(w1)} um",
                    f"    V_sec_{idx} = [",
                    f"        {f(x0)} * um, {f(y0_b)} * um;",
                    f"        {f(x1)} * um, {f(y1_b)} * um;",
                    f"        {f(x1)} * um, {f(y1_t)} * um;",
                    f"        {f(x0)} * um, {f(y0_t)} * um",
                    "    ];",
                    "    addpoly;",
                    f'    set("name", "section_{idx}_taper_{len_tag}um");',
                    '    set("x", 0);',
                    '    set("y", 0);',
                    f'    set("z", {f(seg_z_c)} * um);',
                    f'    set("z span", {f(seg_z_span_nm)} * nm);',
                    f'    set("vertices", V_sec_{idx});',
                    '    set("material", mat_name);',
                    '    addtogroup("InP_Structure");',
                    "",
                ]
            )

    x_total = geom.segments[-1]["x1"] - geom.segments[0]["x0"]
    w_max = max(
        max(s["y0_top"] - s["y0_bot"], s["y1_top"] - s["y1_bot"]) for s in geom.segments
    )
    lines.extend(
        [
            "}",
            "",
            f'?"3D InP structure successfully generated (Total x-length: {f(x_total)} um, '
            f'max y-width: {f(w_max)} um, z-thickness: {f(t_z_nm)} nm).";',
            "",
        ]
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# GDSII (.gds)
# ----------------------------------------------------------------------
def parse_layer_spec(text):
    """Parses 'layer/datatype' (also 'layer,datatype', 'layer:datatype' or just 'layer')."""
    parts = [p for p in re.split(r"[/,:;\s]+", str(text).strip()) if p]
    if not 1 <= len(parts) <= 2:
        raise GeometryError(f"Invalid GDS layer '{text}'. Use the form 'layer/datatype', e.g. 1/0.")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise GeometryError(f"Invalid GDS layer '{text}'. Use integers, e.g. 1/0.") from None
    if len(nums) == 1:
        nums.append(0)
    if not all(0 <= n <= 65535 for n in nums):
        raise GeometryError(f"GDS layer/datatype must be between 0 and 65535 (got '{text}').")
    return nums[0], nums[1]


def sanitize_cell_name(name):
    """Makes a name that is valid both as a GDS cell name and as a Python identifier."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", str(name).strip()).strip("_")
    if not clean:
        clean = DEFAULT_CELL_NAME
    if clean[0].isdigit():
        clean = "cell_" + clean
    if keyword.iskeyword(clean):
        clean += "_"
    return clean[:120]


def polygon_to_dbu(points, dbu_um=GDS_DBU_UM):
    """Snaps a polygon (um) to the integer GDS grid and removes duplicate points."""
    pts = []
    for x, y in points:
        p = (int(round(x / dbu_um)), int(round(y / dbu_um)))
        if not pts or p != pts[-1]:
            pts.append(p)
    while len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        raise GeometryError("The Top view (x-y) polygon collapses to fewer than 3 points on the 1 nm GDS grid.")
    if len(pts) > 8190:
        raise GeometryError("GDSII polygons are limited to 8190 vertices.")
    return pts


def _gds_record(rtype, dtype, payload=b""):
    return struct.pack(">HBB", 4 + len(payload), rtype, dtype) + payload


def _gds_real8(value):
    """Encodes a float as GDSII 8-byte excess-64 base-16 real."""
    if value == 0:
        return b"\x00" * 8
    sign = 0x80 if value < 0 else 0
    value = abs(value)
    exponent = 64
    while value >= 1.0:
        value /= 16.0
        exponent += 1
    while value < 1.0 / 16.0:
        value *= 16.0
        exponent -= 1
    mantissa = int(round(value * (1 << 56)))
    if mantissa >= 1 << 56:
        mantissa >>= 4
        exponent += 1
    return bytes([sign | exponent]) + mantissa.to_bytes(7, "big")


def _gds_string(text):
    data = text.encode("ascii")
    return data + (b"\x00" if len(data) % 2 else b"")


def _write_gds_builtin(path, cell_name, layer, polygons_dbu):
    now = datetime.datetime.now()
    stamp = struct.pack(">6h", now.year, now.month, now.day, now.hour, now.minute, now.second) * 2
    out = [
        _gds_record(0x00, 0x02, struct.pack(">h", 600)),  # HEADER (version 6)
        _gds_record(0x01, 0x02, stamp),  # BGNLIB
        _gds_record(0x02, 0x06, _gds_string("GEOM_GEN")),  # LIBNAME
        _gds_record(0x03, 0x05, _gds_real8(GDS_DBU_UM) + _gds_real8(GDS_DBU_UM * 1e-6)),  # UNITS
        _gds_record(0x05, 0x02, stamp),  # BGNSTR
        _gds_record(0x06, 0x06, _gds_string(cell_name)),  # STRNAME
    ]
    for pts in polygons_dbu:
        closed = list(pts) + [pts[0]]
        out += [
            _gds_record(0x08, 0x00),  # BOUNDARY
            _gds_record(0x0D, 0x02, struct.pack(">h", layer[0])),  # LAYER
            _gds_record(0x0E, 0x02, struct.pack(">h", layer[1])),  # DATATYPE
            _gds_record(0x10, 0x03, struct.pack(f">{2 * len(closed)}i", *[c for p in closed for c in p])),  # XY
            _gds_record(0x11, 0x00),  # ENDEL
        ]
    out += [_gds_record(0x07, 0x00), _gds_record(0x04, 0x00)]  # ENDSTR, ENDLIB
    with open(path, "wb") as fh:
        fh.write(b"".join(out))


def _write_gds_klayout(path, cell_name, layer, polygons_dbu):
    import klayout.db as kdb

    layout = kdb.Layout()  # fresh layout each time: no global cell-name clashes
    layout.dbu = GDS_DBU_UM
    cell = layout.create_cell(cell_name)
    layer_idx = layout.layer(layer[0], layer[1])
    for pts in polygons_dbu:
        cell.shapes(layer_idx).insert(kdb.Polygon([kdb.Point(x, y) for x, y in pts]))
    layout.write(str(path))


def write_gds(geom, path, cell_name=DEFAULT_CELL_NAME, layer=DEFAULT_GDS_LAYER, backend="auto"):
    """
    Writes the top-view (x-y) outline as one polygon to a GDSII file (1 nm grid).
    backend: 'auto' (klayout.db if installed, else built-in), 'klayout' or 'builtin'.
    Returns the name of the backend used.
    """
    cell_name = sanitize_cell_name(cell_name)
    polygons = [polygon_to_dbu(geom.xy_poly)]
    if backend in ("auto", "klayout"):
        try:
            _write_gds_klayout(path, cell_name, layer, polygons)
            return "klayout"
        except ImportError:
            if backend == "klayout":
                raise
    _write_gds_builtin(path, cell_name, layer, polygons)
    return "builtin"


# ----------------------------------------------------------------------
# gdsfactory / gdsfactory+ Python script (.py)
# ----------------------------------------------------------------------
def build_gdsfactory_script(geom, cell_name=DEFAULT_CELL_NAME, layer=DEFAULT_GDS_LAYER, material=DEFAULT_MATERIAL):
    """Builds a standalone gdsfactory script; running it writes <script name>.gds."""
    f = fmt_num
    func = sanitize_cell_name(cell_name)
    z_min, z_max = geom.z_span

    poly_lines = "\n".join(f"    ({f(x)}, {f(y)})," for x, y in geom.xy_poly)
    sec_lines = "\n".join(
        f"    [({f(s['x0'])}, {f(s['y0_bot'])}), ({f(s['x1'])}, {f(s['y1_bot'])}), "
        f"({f(s['x1'])}, {f(s['y1_top'])}), ({f(s['x0'])}, {f(s['y0_top'])})],"
        for s in geom.segments
    )
    z_note = (
        ""
        if geom.z_is_uniform
        else (
            "# NOTE: the drawn side view has an x-dependent thickness. GDS is a 2D format,\n"
            "#       so only the overall z-range is stored in the LayerStack below.\n"
        )
    )

    return f'''#!/usr/bin/env python3
"""
gdsfactory layout script generated by geometry_gui.py ({datetime.date.today().isoformat()}).

Run it to write the GDS file next to this script (same name, .gds extension):
    python <this_script>.py
    python <this_script>.py --show    # also opens the layout (KLayout klive / gdsfactory+)

The top-view (x-y) outline is placed on GDS layer {layer[0]}/{layer[1]}; units are micrometers.
In gdsfactory+ / your own PDK, import `{func}` and use it like any other @gf.cell.
"""

import gdsfactory as gf

LAYER = ({layer[0]}, {layer[1]})
MATERIAL = "{material}"
ZMIN_UM = {f(z_min)}
THICKNESS_UM = {f(z_max - z_min)}
{z_note}
# Top-view outline (x, y) in um
POLYGON_XY_UM = [
{poly_lines}
]

# Same outline split into the longitudinal sections used for the Lumerical export
SECTIONS_XY_UM = [
{sec_lines}
]


def _build(polygons) -> gf.Component:
    c = gf.Component()
    for poly in polygons:
        c.add_polygon(poly, layer=LAYER)
    c.info["material"] = MATERIAL
    c.info["zmin_um"] = ZMIN_UM
    c.info["thickness_um"] = THICKNESS_UM
    return c


# The cells take no arguments so that the GDS cell name is exactly the function name.
@gf.cell
def {func}() -> gf.Component:
    """Structure drawn in geometry_gui.py as a single polygon."""
    return _build([POLYGON_XY_UM])


@gf.cell
def {func}_sections() -> gf.Component:
    """Same structure as one polygon per longitudinal section."""
    return _build(SECTIONS_XY_UM)


def get_layer_stack():
    """LayerStack with the drawn thickness and z position (for 3D views / simulation plugins)."""
    from gdsfactory.technology import LayerLevel, LayerStack

    return LayerStack(
        layers={{
            "core": LayerLevel(
                layer=LAYER,
                thickness=THICKNESS_UM,
                zmin=ZMIN_UM,
                material=MATERIAL,
            )
        }}
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    component = {func}()
    gdspath = component.write_gds(Path(__file__).with_suffix(".gds"))
    print(f"Wrote {{gdspath}}")
    if "--show" in sys.argv:
        component.show()
'''


# ----------------------------------------------------------------------
# Format dispatch
# ----------------------------------------------------------------------
EXPORT_FORMATS = {
    "Lumerical script (.lsf)": [".lsf"],
    "GDSII layout (.gds)": [".gds"],
    "gdsfactory script (.py)": [".py"],
    "All (.lsf + .gds + .py)": [".lsf", ".gds", ".py"],
}
_RESERVED_SCRIPT_NAMES = {"gdsfactory", "klayout", "kfactory", "gplugins", "numpy"}


def export_geometry(geom, path, cell_name=DEFAULT_CELL_NAME, layer=DEFAULT_GDS_LAYER):
    """Writes geom to path; the format is chosen from the extension. Returns a short description."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".lsf":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build_lsf(geom))
        return "Lumerical script"
    if ext == ".gds":
        backend = write_gds(geom, path, cell_name=cell_name, layer=layer)
        return f"GDSII layout (cell {sanitize_cell_name(cell_name)}, layer {layer[0]}/{layer[1]}, via {backend})"
    if ext == ".py":
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in _RESERVED_SCRIPT_NAMES:
            raise GeometryError(
                f"'{stem}.py' would shadow the '{stem}' package when run. Please choose another file name."
            )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(build_gdsfactory_script(geom, cell_name=cell_name, layer=layer))
        return "gdsfactory script"
    raise GeometryError(f"Unsupported file type '{ext}'. Use .lsf, .gds or .py.")


# ======================================================================
# Dark UI theme
# ======================================================================
if sys.platform == "darwin":
    FONT_UI, FONT_MONO = "Helvetica Neue", "Menlo"
elif sys.platform.startswith("win"):
    FONT_UI, FONT_MONO = "Segoe UI", "Consolas"
else:
    FONT_UI, FONT_MONO = "DejaVu Sans", "DejaVu Sans Mono"

THEME = {
    "app_bg": "#0f1115",       # window background / gaps between panels
    "bar_bg": "#161a20",       # header bar & tool strip
    "panel_bg": "#1a1e25",     # view panel header
    "border": "#2a303a",
    "plot_bg": "#12151a",      # 2D plot area
    "grid": "#1f242c",
    "grid_axis": "#3a414d",
    "guide": "#2f3742",
    "text": "#e6e9ef",
    "text_muted": "#8b93a1",
    "text_dim": "#5c6370",
    "accent": "#4c8dff",       # primary actions, hover
    "accent_hover": "#6aa1ff",
    "success": "#2fb37a",
    "success_hover": "#3fc78c",
    "warning": "#f0b429",
    "danger": "#ff6b6b",
    "btn_bg": "#242a33",
    "btn_hover": "#2f3642",
    "entry_bg": "#0f1216",
    "shape_fill": "#d9a21b",   # InP gold
    "shape_edge": "#ffd166",
    "vertex": "#0f1115",
}


class FlatButton(tk.Label):
    """Flat, theme-able button (tk.Button ignores colors on macOS)."""

    VARIANTS = {
        # variant: (bg, hover bg, fg)
        "primary": ("accent", "accent_hover", "#ffffff"),
        "success": ("success", "success_hover", "#ffffff"),
        "secondary": ("btn_bg", "btn_hover", "text"),
        "ghost": ("panel_bg", "btn_hover", "text_muted"),
        "danger": ("btn_bg", "btn_hover", "danger"),
    }

    def __init__(self, parent, text, command=None, variant="secondary", size=10, bold=False, padx=12, pady=5, tooltip=None):
        bg, hover, fg = (THEME.get(c, c) for c in self.VARIANTS[variant])
        if variant == "ghost":
            bg = parent.cget("bg")
        super().__init__(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=(FONT_UI, size, "bold" if bold else "normal"),
            padx=padx,
            pady=pady,
            cursor="hand2",
        )
        self._bg, self._hover, self._fg = bg, hover, fg
        self.command = command
        self.bind("<Enter>", lambda e: self.config(bg=self._hover, fg="#ffffff" if variant == "ghost" else self._fg))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg, fg=self._fg))
        self.bind("<ButtonRelease-1>", self._on_click)
        if tooltip:
            Tooltip(self, tooltip)

    def _on_click(self, event):
        # Only fire when released over the button (like a native button)
        if 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height() and self.command:
            self.command()


class ToggleButton(FlatButton):
    """Flat on/off toggle bound to a BooleanVar."""

    def __init__(self, parent, text, variable, command=None, tooltip=None):
        self.variable = variable
        self._label = text
        self._user_command = command
        super().__init__(parent, text="", command=self._toggle, tooltip=tooltip)
        self._refresh()

    def _toggle(self):
        self.variable.set(not self.variable.get())
        self._refresh()
        if self._user_command:
            self._user_command()

    def _refresh(self):
        on = self.variable.get()
        self.config(text=f"{'●' if on else '○'}  {self._label}")
        self._fg = THEME["accent_hover"] if on else THEME["text_muted"]
        self.config(fg=self._fg)


def themed_entry(parent, width, mono=True):
    return tk.Entry(
        parent,
        width=width,
        font=(FONT_MONO if mono else FONT_UI, 11),
        bg=THEME["entry_bg"],
        fg=THEME["text"],
        insertbackground=THEME["text"],
        disabledbackground=THEME["entry_bg"],
        relief=tk.FLAT,
        highlightthickness=1,
        highlightbackground=THEME["border"],
        highlightcolor=THEME["accent"],
        selectbackground=THEME["accent"],
        selectforeground="#ffffff",
    )


def caption(parent, text):
    """Small upper-case group caption used in the tool strips."""
    return tk.Label(parent, text=text.upper(), font=(FONT_UI, 9, "bold"), bg=parent.cget("bg"), fg=THEME["text_dim"])


def vsep(parent):
    return tk.Frame(parent, bg=THEME["border"], width=1)


class Tooltip:
    """Minimal hover tooltip."""

    def __init__(self, widget, text, delay_ms=500):
        self.widget, self.text, self.delay_ms = widget, text, delay_ms
        self._job = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._job = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None

    def _show(self):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            bg="#262c36",
            fg=THEME["text"],
            font=(FONT_UI, 10),
            padx=8,
            pady=5,
            highlightthickness=1,
            highlightbackground=THEME["border"],
        ).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def apply_ttk_dark_style(root):
    """Dark styling for the ttk widgets in use (combobox)."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "Dark.TCombobox",
        fieldbackground=THEME["entry_bg"],
        background=THEME["btn_bg"],
        foreground=THEME["text"],
        arrowcolor=THEME["text_muted"],
        bordercolor=THEME["border"],
        lightcolor=THEME["entry_bg"],
        darkcolor=THEME["entry_bg"],
        selectbackground=THEME["entry_bg"],
        selectforeground=THEME["text"],
        padding=4,
    )
    style.map(
        "Dark.TCombobox",
        fieldbackground=[("readonly", THEME["entry_bg"])],
        foreground=[("readonly", THEME["text"])],
        bordercolor=[("focus", THEME["accent"])],
        background=[("active", THEME["btn_hover"])],
    )
    root.option_add("*TCombobox*Listbox.background", THEME["bar_bg"])
    root.option_add("*TCombobox*Listbox.foreground", THEME["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", THEME["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", (FONT_UI, 11))


def _wheel_event(event, direction):
    """Gives X11 Button-4/5 scroll events the 'delta' attribute used by <MouseWheel> handlers."""
    event.delta = direction
    return event


class OrthoCanvas2D(tk.Frame):
    """Interactive 2D orthographic view canvas for drawing and editing connected points."""

    def __init__(
        self,
        parent,
        title,
        u_label,
        v_label,
        u_range,
        v_range,
        u_guides=None,
        v_guides=None,
        u_grid_step=50.0,
        v_grid_step=0.25,
        on_point_selected=None,
        on_geometry_changed=None,
    ):
        super().__init__(parent, bg=THEME["panel_bg"], highlightthickness=1, highlightbackground=THEME["border"])
        self.title = title
        self.u_label = u_label
        self.v_label = v_label
        self.default_u_range = list(u_range)
        self.default_v_range = list(v_range)
        self.u_range = list(u_range)
        self.v_range = list(v_range)
        self.u_guides = u_guides or []
        self.v_guides = v_guides or []
        self.u_grid_step = u_grid_step
        self.v_grid_step = v_grid_step
        self.on_point_selected = on_point_selected
        self.on_geometry_changed = on_geometry_changed

        # Polygon state: list of [u, v] in physical units (micrometers)
        self.points = []
        self.closed = False
        self.selected_idx = None
        self.dragging_idx = None
        self.hover_idx = None
        self.hover_edge_idx = None
        self.cursor_uv = None
        self.snap_enabled = True

        # Panning state (middle-drag or Space+drag)
        self._pan_start = None

        # Header bar inside view: title + point count | live cursor readout | view tools
        header = tk.Frame(self, bg=THEME["panel_bg"])
        header.pack(fill=tk.X, side=tk.TOP)

        self.title_lbl = tk.Label(
            header,
            text=self.title,
            font=(FONT_UI, 12, "bold"),
            bg=THEME["panel_bg"],
            fg=THEME["text"],
            anchor="w",
        )
        self.title_lbl.pack(side=tk.LEFT, padx=(12, 6), pady=6)

        self.count_lbl = tk.Label(
            header,
            text="empty",
            font=(FONT_UI, 9),
            bg=THEME["btn_bg"],
            fg=THEME["text_muted"],
            padx=6,
            pady=1,
        )
        self.count_lbl.pack(side=tk.LEFT, padx=(0, 10))

        self.coord_lbl = tk.Label(
            header,
            text="",
            font=(FONT_MONO, 10),
            bg=THEME["panel_bg"],
            fg=THEME["text_muted"],
        )
        self.coord_lbl.pack(side=tk.LEFT, padx=4)

        for text, cmd, tip in (
            ("Clear", self.clear_points, "Remove all points from this view"),
            ("Close shape", self.close_polygon, "Close the polygon and switch to edit mode\n(drag vertices, click an edge to insert a point)"),
            ("Fit", self.reset_view, "Reset zoom / pan of this view"),
        ):
            FlatButton(header, text, cmd, variant="ghost", size=10, padx=8, pady=3, tooltip=tip).pack(
                side=tk.RIGHT, padx=(0, 4) if text != "Clear" else (0, 8), pady=4
            )

        # Drawing Canvas
        self.canvas = tk.Canvas(
            self,
            bg=THEME["plot_bg"],
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Margins for axis labels
        self.margin_left = 54
        self.margin_right = 20
        self.margin_top = 20
        self.margin_bottom = 36

        # Bind events
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

        # Right-click (macOS uses Button-2 or Button-3 or Control-Button-1)
        self.canvas.bind("<ButtonPress-2>", self._on_right_click)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<Control-Button-1>", self._on_right_click)

        # Zoom with scroll wheel (Button-4/5 are the X11/Linux wheel events)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_mousewheel(_wheel_event(e, +1)))
        self.canvas.bind("<Button-5>", lambda e: self._on_mousewheel(_wheel_event(e, -1)))

    # ------------------------------------------------------------------
    # Coordinate transforms (physical (u, v) <-> canvas pixel (px, py))
    # ------------------------------------------------------------------
    def _plot_rect(self):
        w = max(self.canvas.winfo_width(), 100)
        h = max(self.canvas.winfo_height(), 100)
        x0 = self.margin_left
        y0 = self.margin_top
        x1 = max(x0 + 20, w - self.margin_right)
        y1 = max(y0 + 20, h - self.margin_bottom)
        return x0, y0, x1, y1

    def uv_to_px(self, u, v):
        x0, y0, x1, y1 = self._plot_rect()
        u_min, u_max = self.u_range
        v_min, v_max = self.v_range
        px = x0 + (u - u_min) / (u_max - u_min) * (x1 - x0)
        py = y1 - (v - v_min) / (v_max - v_min) * (y1 - y0)
        return px, py

    def px_to_uv(self, px, py):
        x0, y0, x1, y1 = self._plot_rect()
        u_min, u_max = self.u_range
        v_min, v_max = self.v_range
        u = u_min + (px - x0) / max(1.0, (x1 - x0)) * (u_max - u_min)
        v = v_min + (y1 - py) / max(1.0, (y1 - y0)) * (v_max - v_min)
        return u, v

    def snap_uv(self, u, v, disable_snap=False):
        if not self.snap_enabled or disable_snap:
            return round(u, 4), round(v, 4)

        u_span = self.u_range[1] - self.u_range[0]
        v_span = self.v_range[1] - self.v_range[0]

        # Snap to the nearest key guide coordinate if within 2.5% (u) / 3% (v) of span
        def nearest_guide(val, guides, tol):
            best = min(guides, key=lambda g: abs(val - g), default=None)
            return best if best is not None and abs(val - best) <= tol else None

        snapped_u = nearest_guide(u, self.u_guides, 0.025 * u_span)
        if snapped_u is None:
            # Snap to fine grid step
            fine_u = self.u_grid_step / 5.0
            snapped_u = round(u / fine_u) * fine_u

        snapped_v = nearest_guide(v, self.v_guides, 0.03 * v_span)
        if snapped_v is None:
            fine_v = self.v_grid_step / 5.0
            snapped_v = round(v / fine_v) * fine_v

        return round(snapped_u, 4), round(snapped_v, 4)

    # ------------------------------------------------------------------
    # Hit testing for vertices and edges
    # ------------------------------------------------------------------
    def _find_vertex_near(self, px, py, radius=10.0):
        best_i = None
        best_d2 = radius * radius
        for i, (u, v) in enumerate(self.points):
            vx, vy = self.uv_to_px(u, v)
            d2 = (px - vx) ** 2 + (py - vy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _find_edge_near(self, px, py, tol=7.0):
        n = len(self.points)
        if n < 2:
            return None
        num_edges = n if (self.closed or n >= 3) else n - 1
        best_edge = None
        best_dist = tol
        for i in range(num_edges):
            j = (i + 1) % n
            x1, y1 = self.uv_to_px(*self.points[i])
            x2, y2 = self.uv_to_px(*self.points[j])
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-6:
                continue
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
            if 0.15 <= t <= 0.85:
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = math.hypot(px - proj_x, py - proj_y)
                if dist <= best_dist:
                    best_dist = dist
                    best_edge = i
        return best_edge

    # ------------------------------------------------------------------
    # Mouse interaction handlers
    # ------------------------------------------------------------------
    def _on_motion(self, event):
        disable_snap = bool(event.state & 0x0001)  # Shift key disables snap
        u, v = self.px_to_uv(event.x, event.y)
        su, sv = self.snap_uv(u, v, disable_snap=disable_snap)
        self.cursor_uv = (su, sv)
        self.coord_lbl.config(
            text=f"{self.u_label}={su:g} μm, {self.v_label}={sv:g} μm"
        )

        v_idx = self._find_vertex_near(event.x, event.y)
        e_idx = None if v_idx is not None else self._find_edge_near(event.x, event.y)

        changed = (v_idx != self.hover_idx) or (e_idx != self.hover_edge_idx)
        self.hover_idx = v_idx
        self.hover_edge_idx = e_idx

        if v_idx is not None:
            self.canvas.config(cursor="hand2")
        elif e_idx is not None and self.closed:
            self.canvas.config(cursor="plus")
        else:
            self.canvas.config(cursor="crosshair")

        if changed or not self.closed:
            self.redraw()

    def _on_left_down(self, event):
        disable_snap = bool(event.state & 0x0001)
        u, v = self.px_to_uv(event.x, event.y)
        su, sv = self.snap_uv(u, v, disable_snap=disable_snap)

        # 1. Check if clicking an existing vertex
        v_idx = self._find_vertex_near(event.x, event.y)
        if v_idx is not None:
            # If user clicks the first vertex while drawing >= 3 points, close polygon AND allow dragging
            if not self.closed and v_idx == 0 and len(self.points) >= 3:
                self.closed = True
            self.selected_idx = v_idx
            self.dragging_idx = v_idx
            self._notify_point_selected()
            self.redraw()
            return

        # 2. If polygon is closed (or Shift held), clicking on an edge inserts a new point on that edge
        e_idx = self._find_edge_near(event.x, event.y)
        if e_idx is not None and (self.closed or bool(event.state & 0x0001)):
            insert_pos = e_idx + 1
            self.points.insert(insert_pos, [su, sv])
            self.selected_idx = insert_pos
            self.dragging_idx = insert_pos
            self._notify_point_selected()
            if self.on_geometry_changed:
                self.on_geometry_changed()
            self.redraw()
            return

        # 3. Otherwise, add a new connected point
        if self.closed:
            # Even if closed, if user clicks an edge it inserts; if they click in empty space,
            # insert along the nearest polygon edge so the polygon stays well-ordered!
            insert_pos = self._nearest_edge_insert_index(su, sv)
            self.points.insert(insert_pos, [su, sv])
            self.selected_idx = insert_pos
            self.dragging_idx = insert_pos
        else:
            self.points.append([su, sv])
            self.selected_idx = len(self.points) - 1
            self.dragging_idx = self.selected_idx

        self._notify_point_selected()
        if self.on_geometry_changed:
            self.on_geometry_changed()
        self.redraw()

    def _nearest_edge_insert_index(self, u, v):
        n = len(self.points)
        if n < 2:
            return n
        px, py = self.uv_to_px(u, v)
        best_idx = n
        best_dist = float("inf")
        for i in range(n):
            j = (i + 1) % n
            x1, y1 = self.uv_to_px(*self.points[i])
            x2, y2 = self.uv_to_px(*self.points[j])
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 < 1e-6:
                dist = math.hypot(px - x1, py - y1)
            else:
                t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len2))
                dist = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
            if dist < best_dist:
                best_dist = dist
                best_idx = i + 1
        return best_idx

    def _on_left_drag(self, event):
        if self.dragging_idx is None:
            return
        x0, y0, x1, y1 = self._plot_rect()
        cx = max(x0, min(x1, event.x))
        cy = max(y0, min(y1, event.y))
        disable_snap = bool(event.state & 0x0001)
        u, v = self.px_to_uv(cx, cy)
        su, sv = self.snap_uv(u, v, disable_snap=disable_snap)
        self.points[self.dragging_idx] = [su, sv]
        self.cursor_uv = (su, sv)
        self.coord_lbl.config(
            text=f"{self.u_label}={su:g} μm, {self.v_label}={sv:g} μm"
        )
        self._notify_point_selected()
        if self.on_geometry_changed:
            self.on_geometry_changed()
        self.redraw()

    def _on_left_up(self, event):
        self.dragging_idx = None
        self.redraw()

    def _on_double_click(self, event):
        if len(self.points) >= 3:
            self.closed = True
            self.redraw()

    def _on_right_click(self, event):
        v_idx = self._find_vertex_near(event.x, event.y, radius=12.0)
        if v_idx is not None:
            del self.points[v_idx]
            if not self.points:
                self.selected_idx = None
                self.closed = False
            else:
                self.selected_idx = min(v_idx, len(self.points) - 1)
            self._notify_point_selected()
            if self.on_geometry_changed:
                self.on_geometry_changed()
            self.redraw()
        elif len(self.points) >= 3 and not self.closed:
            # Right-clicking empty space finishes/closes the current polygon
            self.closed = True
            self.redraw()

    def _on_mousewheel(self, event):
        # Zoom around cursor position
        u_c, v_c = self.px_to_uv(event.x, event.y)
        factor = 0.85 if event.delta > 0 else 1.18
        u_min, u_max = self.u_range
        v_min, v_max = self.v_range
        self.u_range = [
            u_c - (u_c - u_min) * factor,
            u_c + (u_max - u_c) * factor,
        ]
        self.v_range = [
            v_c - (v_c - v_min) * factor,
            v_c + (v_max - v_c) * factor,
        ]
        self.redraw()

    def _notify_point_selected(self):
        if self.on_point_selected:
            if self.selected_idx is not None and 0 <= self.selected_idx < len(self.points):
                pt = self.points[self.selected_idx]
                self.on_point_selected(self, self.selected_idx, pt[0], pt[1])
            else:
                self.on_point_selected(self, None, None, None)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def clear_points(self):
        self.points = []
        self.closed = False
        self.selected_idx = None
        self.dragging_idx = None
        self._notify_point_selected()
        if self.on_geometry_changed:
            self.on_geometry_changed()
        self.redraw()

    def close_polygon(self):
        if len(self.points) >= 3:
            self.closed = True
            self.redraw()

    def reset_view(self):
        self.u_range = list(self.default_u_range)
        self.v_range = list(self.default_v_range)
        self.redraw()

    def set_points(self, pts, closed=True):
        self.points = [[float(p[0]), float(p[1])] for p in pts]
        self.closed = closed and len(self.points) >= 3
        self.selected_idx = None
        self._notify_point_selected()
        if self.on_geometry_changed:
            self.on_geometry_changed()
        self.redraw()

    def update_selected_point(self, new_u, new_v):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.points):
            self.points[self.selected_idx] = [float(new_u), float(new_v)]
            if self.on_geometry_changed:
                self.on_geometry_changed()
            self.redraw()

    def delete_selected_point(self):
        if self.selected_idx is not None and 0 <= self.selected_idx < len(self.points):
            del self.points[self.selected_idx]
            if not self.points:
                self.selected_idx = None
                self.closed = False
            else:
                self.selected_idx = min(self.selected_idx, len(self.points) - 1)
            self._notify_point_selected()
            if self.on_geometry_changed:
                self.on_geometry_changed()
            self.redraw()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def redraw(self):
        T = THEME
        c = self.canvas
        c.delete("all")
        x0, y0, x1, y1 = self._plot_rect()

        n = len(self.points)
        if n == 0:
            self.count_lbl.config(text="empty", fg=T["text_dim"])
        else:
            state = "closed" if self.closed else "drawing"
            self.count_lbl.config(
                text=f"{n} pts · {state}",
                fg=T["success"] if self.closed and n >= 3 else T["warning"],
            )

        # Plot area background & border
        c.create_rectangle(x0, y0, x1, y1, fill=T["plot_bg"], outline=T["border"])

        u_min, u_max = self.u_range
        v_min, v_max = self.v_range
        tick_font = (FONT_UI, 9)

        # 1. Regular grid lines (+ tick labels), zero axis emphasised
        step_u = self.u_grid_step
        u_val = math.ceil(u_min / step_u) * step_u
        while u_val <= u_max + 1e-9:
            px, _ = self.uv_to_px(u_val, 0)
            if x0 <= px <= x1:
                is_zero = abs(u_val) < 1e-9
                c.create_line(px, y0, px, y1, fill=T["grid_axis"] if is_zero else T["grid"], width=1)
                c.create_text(px, y1 + 12, text=f"{u_val:g}", font=tick_font, fill=T["text_muted"])
            u_val += step_u

        step_v = self.v_grid_step
        v_val = math.ceil(v_min / step_v) * step_v
        while v_val <= v_max + 1e-9:
            _, py = self.uv_to_px(0, v_val)
            if y0 <= py <= y1:
                is_zero = abs(v_val) < 1e-9
                c.create_line(x0, py, x1, py, fill=T["grid_axis"] if is_zero else T["grid"], width=1)
                c.create_text(x0 - 6, py, text=f"{v_val:g}", font=tick_font, fill=T["text_muted"], anchor="e")
            v_val += step_v

        # 2. Reference guide lines (key Slide 1 dimensions); skip labels that would overlap
        last_label_px = None
        for gu in sorted(self.u_guides):
            if u_min <= gu <= u_max and abs(gu) > 1e-9:
                px, _ = self.uv_to_px(gu, 0)
                c.create_line(px, y0, px, y1, fill=T["guide"], dash=(3, 4))
                if last_label_px is None or px - last_label_px > 30:
                    c.create_text(px, y0 - 8, text=f"{gu:g}", font=(FONT_UI, 8), fill=T["text_dim"])
                    last_label_px = px

        for gv in self.v_guides:
            if v_min <= gv <= v_max and abs(gv) > 1e-9:
                _, py = self.uv_to_px(0, gv)
                c.create_line(x0, py, x1, py, fill=T["guide"], dash=(3, 4))

        # Axis titles
        c.create_text(
            (x0 + x1) / 2, y1 + 27, text=f"{self.u_label} (μm)", font=(FONT_UI, 10, "bold"), fill=T["text_muted"]
        )
        c.create_text(
            16, (y0 + y1) / 2, text=f"{self.v_label}\n(μm)", font=(FONT_UI, 10, "bold"), fill=T["text_muted"],
            justify=tk.CENTER,
        )

        # Empty-view hint
        if n == 0:
            c.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text="Click to add points  ·  double-click to close",
                font=(FONT_UI, 11),
                fill=T["text_dim"],
            )

        # 3. Polygon & connected segments
        px_pts = [self.uv_to_px(u, v) for u, v in self.points]

        flat_coords = [coord for pt in px_pts for coord in pt]
        if n >= 3 and self.closed:
            # InP gold, matching Slide 1
            c.create_polygon(*flat_coords, fill=T["shape_fill"], outline=T["shape_edge"], width=2)
        elif n >= 2:
            # Still drawing: open polyline
            c.create_line(*flat_coords, fill=T["shape_edge"], width=2)

        # Highlight hovered edge when in edit mode
        if self.hover_edge_idx is not None and self.hover_edge_idx < n:
            i = self.hover_edge_idx
            j = (i + 1) % n
            c.create_line(*px_pts[i], *px_pts[j], fill=T["accent"], width=3)

        # Preview rubber-band line when still adding points
        if not self.closed and n >= 1 and self.cursor_uv is not None and self.hover_idx is None:
            cx, cy = self.uv_to_px(*self.cursor_uv)
            lx, ly = px_pts[-1]
            c.create_line(lx, ly, cx, cy, fill=T["accent"], dash=(4, 2), width=1.5)
            if n >= 2:
                fx, fy = px_pts[0]
                c.create_line(cx, cy, fx, fy, fill=T["text_dim"], dash=(2, 3), width=1)

        # 4. Vertex handles and coordinates of hovered / selected vertex
        for i, (px, py) in enumerate(px_pts):
            is_sel = i == self.selected_idx
            is_hov = i == self.hover_idx
            r = 6 if (is_sel or is_hov) else 4
            if is_sel:
                fill_c, outline_c = T["danger"], "#ffffff"
            elif is_hov:
                fill_c, outline_c = T["accent"], "#ffffff"
            else:
                fill_c, outline_c = T["vertex"], T["shape_edge"]
            c.create_oval(px - r, py - r, px + r, py + r, fill=fill_c, outline=outline_c, width=1.5)
            if is_sel or is_hov:
                u, v = self.points[i]
                tid = c.create_text(
                    px + 10, py - 12, text=f"P{i+1}  ({u:g}, {v:g})", font=(FONT_MONO, 10, "bold"), fill="#ffffff", anchor="w"
                )
                bx0, by0, bx1, by1 = c.bbox(tid)
                bg = c.create_rectangle(bx0 - 5, by0 - 2, bx1 + 5, by1 + 2, fill=fill_c, outline="")
                c.tag_lower(bg, tid)


class View3DCanvas(tk.Frame):
    """Interactive 3D viewport for displaying the reconstructed 3D waveguide geometry."""

    def __init__(self, parent):
        super().__init__(parent, bg=THEME["panel_bg"], highlightthickness=1, highlightbackground=THEME["border"])

        header = tk.Frame(self, bg=THEME["panel_bg"])
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header,
            text="3D view (x-y-z)",
            font=(FONT_UI, 12, "bold"),
            bg=THEME["panel_bg"],
            fg=THEME["text"],
            anchor="w",
        ).pack(side=tk.LEFT, padx=(12, 6), pady=6)

        # Status pill: empty / up to date / outdated
        self.state_lbl = tk.Label(header, font=(FONT_UI, 9), padx=6, pady=1)
        self.state_lbl.pack(side=tk.LEFT, padx=(0, 10))
        self.set_state("empty")

        tk.Label(
            header,
            text="drag to rotate · scroll to zoom",
            font=(FONT_UI, 10),
            bg=THEME["panel_bg"],
            fg=THEME["text_dim"],
        ).pack(side=tk.LEFT, padx=4)

        FlatButton(
            header, "Reset view", self.reset_camera, variant="ghost", size=10, padx=8, pady=3,
            tooltip="Reset rotation and zoom of the 3D view",
        ).pack(side=tk.RIGHT, padx=(0, 8), pady=4)

        self.canvas = tk.Canvas(self, bg=THEME["plot_bg"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Camera parameters
        self.azim = -55.0  # degrees
        self.elev = 26.0   # degrees
        self.zoom = 1.0
        self._drag_prev = None

        # Reconstructed 3D data
        # List of 3D prism segments or general extruded polyhedron faces
        self.faces_3d = []  # Each item: (list of (x, y, z) vertices, face_type)
        self.bbox_3d = None
        self.summary_text = ""

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._on_mousewheel(_wheel_event(e, +1)))
        self.canvas.bind("<Button-5>", lambda e: self._on_mousewheel(_wheel_event(e, -1)))

    STATES = {
        "empty": ("not generated", "text_dim"),
        "ok": ("up to date", "success"),
        "outdated": ("outdated — regenerate", "warning"),
    }

    def set_state(self, state):
        text, color = self.STATES[state]
        self.state_lbl.config(text=f"● {text}", fg=THEME[color], bg=THEME["btn_bg"])

    def reset_camera(self):
        self.azim = -55.0
        self.elev = 26.0
        self.zoom = 1.0
        self.redraw()

    def clear_3d(self):
        self.faces_3d = []
        self.bbox_3d = None
        self.summary_text = ""
        self.redraw()

    def set_geometry(self, faces_3d, bbox_3d, summary_text=""):
        self.faces_3d = faces_3d
        self.bbox_3d = bbox_3d
        self.summary_text = summary_text
        self.redraw()

    def _on_mouse_down(self, event):
        self._drag_prev = (event.x, event.y)

    def _on_mouse_drag(self, event):
        if self._drag_prev is None:
            return
        dx = event.x - self._drag_prev[0]
        dy = event.y - self._drag_prev[1]
        self._drag_prev = (event.x, event.y)
        self.azim += dx * 0.6
        self.elev = max(-85.0, min(85.0, self.elev + dy * 0.6))
        self.redraw()

    def _on_mouse_up(self, event):
        self._drag_prev = None

    def _on_mousewheel(self, event):
        factor = 1.12 if event.delta > 0 else 0.89
        self.zoom = max(0.3, min(4.0, self.zoom * factor))
        self.redraw()

    def _project_point(self, x, y, z, cx, cy, scale):
        """Normalize physical (x, y, z) to visual box [-1, 1] with schematic aspect ratio and project."""
        if self.bbox_3d is None:
            nx, ny, nz = x, y, z
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self.bbox_3d
            xmid = 0.5 * (xmin + xmax)
            ymid = 0.5 * (ymin + ymax)
            zmid = 0.5 * (zmin + zmax)
            dx = max(xmax - xmin, 1e-6)
            dy = max(ymax - ymin, 1e-6)
            dz = max(zmax - zmin, 1e-6)
            # Normalized schematic proportions so x (long), y (width), and z (thin layer)
            # are all clearly visible in 3D
            nx = (x - xmid) / (dx * 0.5) * 1.35
            ny = (y - ymid) / (dy * 0.5) * 0.55
            nz = (z - zmid) / (dz * 0.5) * 0.22

        az = math.radians(self.azim)
        el = math.radians(self.elev)
        cos_a, sin_a = math.cos(az), math.sin(az)
        cos_e, sin_e = math.cos(el), math.sin(el)

        # Rotate around Z by azim, then tilt by elev
        rx = nx * cos_a - ny * sin_a
        ry = nx * sin_a + ny * cos_a
        rz = nz

        px_3d = rx
        py_3d = -ry * sin_e + rz * cos_e
        depth = ry * cos_e + rz * sin_e

        # Perspective factor
        persp = 1.0 / max(0.4, 3.2 - depth * 0.35) * 2.8
        sx = cx + px_3d * scale * self.zoom * persp
        sy = cy - py_3d * scale * self.zoom * persp
        return sx, sy, depth

    def redraw(self):
        self.canvas.delete("all")
        w = max(self.canvas.winfo_width(), 100)
        h = max(self.canvas.winfo_height(), 100)
        cx = w * 0.5
        cy = h * 0.52
        scale = min(w, h) * 0.31

        if not self.faces_3d:
            self.canvas.create_text(
                cx,
                cy - 12,
                text="No 3D geometry yet",
                font=(FONT_UI, 14, "bold"),
                fill=THEME["text_muted"],
            )
            self.canvas.create_text(
                cx,
                cy + 14,
                text="Draw the Top (x-y), Side (x-z) and Front (y-z) profiles\n(or load the Slide 1 preset), then press  Generate geometry.",
                font=(FONT_UI, 11),
                fill=THEME["text_dim"],
                justify=tk.CENTER,
            )
            self._draw_axes_triad(w, h)
            return

        # Draw 3D bounding wireframe box & dimension labels
        xmin, xmax, ymin, ymax, zmin, zmax = self.bbox_3d
        box_corners = [
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmax, ymax, zmin),
            (xmin, ymax, zmin),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
            (xmax, ymax, zmax),
            (xmin, ymax, zmax),
        ]
        proj_corners = [self._project_point(*pt, cx, cy, scale) for pt in box_corners]
        box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in box_edges:
            self.canvas.create_line(
                proj_corners[i][0],
                proj_corners[i][1],
                proj_corners[j][0],
                proj_corners[j][1],
                fill=THEME["grid_axis"],
                dash=(2, 3),
            )

        # Depth-sort faces (Painter's algorithm)
        rendered_faces = []
        for verts, face_kind in self.faces_3d:
            proj = [self._project_point(vx, vy, vz, cx, cy, scale) for vx, vy, vz in verts]
            avg_depth = sum(p[2] for p in proj) / len(proj)
            rendered_faces.append((avg_depth, proj, face_kind))

        rendered_faces.sort(key=lambda item: item[0])

        # Draw sorted faces
        for _, proj, face_kind in rendered_faces:
            coords = [c for pt in proj for c in (pt[0], pt[1])]
            if face_kind == "top":
                fill_color = "#ffd43b"
                edge_color = "#8c6d1f"
            elif face_kind == "bottom":
                fill_color = "#cca000"
                edge_color = "#665000"
            elif face_kind == "end":
                fill_color = "#e6ac00"
                edge_color = "#5c4500"
            else:
                fill_color = "#f59f00"
                edge_color = "#704800"

            self.canvas.create_polygon(
                *coords,
                fill=fill_color,
                outline=edge_color,
                width=1.2,
            )

        # Axis dimension annotations on bounding box (on dark backdrops so they stay readable)
        p_x0, p_x1, p_y1, p_z1 = proj_corners[0], proj_corners[1], proj_corners[2], proj_corners[5]
        self._badge(0.5 * (p_x0[0] + p_x1[0]), 0.5 * (p_x0[1] + p_x1[1]) + 16, f"Δx = {xmax - xmin:g} μm", "#74c0fc")
        self._badge(0.5 * (p_x1[0] + p_y1[0]) + 30, 0.5 * (p_x1[1] + p_y1[1]) + 10, f"Δy = {ymax - ymin:g} μm", "#63e6be")
        self._badge(0.5 * (p_x1[0] + p_z1[0]) + 40, 0.5 * (p_x1[1] + p_z1[1]), f"Δz = {(zmax - zmin)*1000:g} nm", "#ffa8a8")

        # Info card at top-left of 3D canvas
        if self.summary_text:
            tid = self.canvas.create_text(
                22, 18, text=self.summary_text, font=(FONT_MONO, 10), fill=THEME["text"], anchor="nw"
            )
            bx0, by0, bx1, by1 = self.canvas.bbox(tid)
            card = self.canvas.create_rectangle(
                bx0 - 10, by0 - 8, bx1 + 10, by1 + 8, fill=THEME["panel_bg"], outline=THEME["border"]
            )
            self.canvas.tag_lower(card, tid)

        self._draw_axes_triad(w, h)

    def _badge(self, x, y, text, color):
        tid = self.canvas.create_text(x, y, text=text, font=(FONT_UI, 10, "bold"), fill=color)
        bx0, by0, bx1, by1 = self.canvas.bbox(tid)
        bg = self.canvas.create_rectangle(bx0 - 5, by0 - 2, bx1 + 5, by1 + 2, fill=THEME["app_bg"], outline="")
        self.canvas.tag_lower(bg, tid)

    def _draw_axes_triad(self, w, h):
        ox, oy = 52, h - 42
        axis_len = 28
        az = math.radians(self.azim)
        el = math.radians(self.elev)
        cos_a, sin_a = math.cos(az), math.sin(az)
        cos_e, sin_e = math.cos(el), math.sin(el)

        axes = [
            ("x", (1, 0, 0), "#74c0fc"),
            ("y", (0, 1, 0), "#63e6be"),
            ("z", (0, 0, 1), "#ffa8a8"),
        ]
        for label, (ux, uy, uz), color in axes:
            rx = ux * cos_a - uy * sin_a
            ry = ux * sin_a + uy * cos_a
            rz = uz
            sx = ox + rx * axis_len
            sy = oy - (-ry * sin_e + rz * cos_e) * axis_len
            self.canvas.create_line(ox, oy, sx, sy, fill=color, width=2, arrow=tk.LAST)
            self.canvas.create_text(
                sx + (sx - ox) * 0.28,
                sy + (sy - oy) * 0.28,
                text=label,
                font=(FONT_UI, 10, "bold"),
                fill=color,
            )


class LumericalGeometryApp:
    """Main application window combining the 4 views and .lsf / .gds / gdsfactory export."""

    def __init__(self, root):
        self.root = root
        self.root.title("Ansys Lumerical / GDS 3D Geometry Generator — Multi-View Editor")
        self.root.geometry("1320x880")
        self.root.minsize(1180, 720)
        self.root.configure(bg=THEME["app_bg"])
        try:  # dark window title bar on macOS (Tk >= 8.6.10)
            self.root.tk.call("::tk::unsupported::MacWindowStyle", "appearance", self.root, "darkaqua")
        except tk.TclError:
            pass
        apply_ttk_dark_style(self.root)

        self.active_view = None
        self.geometry = None  # last reconstructed Geometry3D
        self.geometry_outdated = False  # views edited since the last reconstruction

        self._build_ui()

    def _build_ui(self):
        T = THEME
        mod = "Cmd" if sys.platform == "darwin" else "Ctrl"

        # ==================================================================
        # Row 1 — header: app title | main workflow (Generate geometry → export)
        # ==================================================================
        header = tk.Frame(self.root, bg=T["bar_bg"], padx=14, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)

        title_box = tk.Frame(header, bg=T["bar_bg"])
        title_box.pack(side=tk.LEFT)
        tk.Label(
            title_box, text="◆", font=(FONT_UI, 18), bg=T["bar_bg"], fg=T["shape_edge"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            title_box, text="Geometry Generator", font=(FONT_UI, 15, "bold"), bg=T["bar_bg"], fg=T["text"]
        ).pack(side=tk.LEFT)
        tk.Label(
            title_box, text="  Lumerical · GDS · gdsfactory", font=(FONT_UI, 11), bg=T["bar_bg"], fg=T["text_dim"]
        ).pack(side=tk.LEFT)

        # Right side, packed right-to-left: [Generate file] [cell] [layer] [format] | [Generate geometry]
        self.btn_gen_file = FlatButton(
            header,
            "⤓  Generate file",
            self.generate_file,
            variant="success",
            size=12,
            bold=True,
            padx=16,
            pady=7,
            tooltip=f"Step 2 — rebuild from the current drawing and save\nin the selected export format  ({mod}+S)",
        )
        self.btn_gen_file.pack(side=tk.RIGHT, padx=(10, 0))

        self.ent_cell = themed_entry(header, 14)
        self.ent_cell.insert(0, DEFAULT_CELL_NAME)
        self.ent_cell.pack(side=tk.RIGHT, ipady=3)
        tk.Label(header, text="Cell", font=(FONT_UI, 10), bg=T["bar_bg"], fg=T["text_muted"]).pack(
            side=tk.RIGHT, padx=(12, 5)
        )
        Tooltip(self.ent_cell, "GDS / gdsfactory cell name")

        self.ent_layer = themed_entry(header, 5)
        self.ent_layer.insert(0, f"{DEFAULT_GDS_LAYER[0]}/{DEFAULT_GDS_LAYER[1]}")
        self.ent_layer.pack(side=tk.RIGHT, ipady=3)
        tk.Label(header, text="Layer", font=(FONT_UI, 10), bg=T["bar_bg"], fg=T["text_muted"]).pack(
            side=tk.RIGHT, padx=(12, 5)
        )
        Tooltip(self.ent_layer, "GDS layer/datatype, e.g. 1/0")

        self.format_var = tk.StringVar(value=next(iter(EXPORT_FORMATS)))
        cmb_format = ttk.Combobox(
            header,
            textvariable=self.format_var,
            values=list(EXPORT_FORMATS),
            state="readonly",
            width=22,
            style="Dark.TCombobox",
            font=(FONT_UI, 11),
        )
        cmb_format.pack(side=tk.RIGHT)
        cmb_format.bind("<<ComboboxSelected>>", lambda e: cmb_format.selection_clear())
        tk.Label(header, text="Export", font=(FONT_UI, 10), bg=T["bar_bg"], fg=T["text_muted"]).pack(
            side=tk.RIGHT, padx=(0, 6)
        )

        vsep(header).pack(side=tk.RIGHT, fill=tk.Y, padx=16, pady=2)

        self.btn_gen_geom = FlatButton(
            header,
            "▶  Generate geometry",
            self.generate_geometry,
            variant="primary",
            size=12,
            bold=True,
            padx=16,
            pady=7,
            tooltip=f"Step 1 — reconstruct the 3D solid from the three\n2D views and show it in the 3D view  ({mod}+G)",
        )
        self.btn_gen_geom.pack(side=tk.RIGHT)

        tk.Frame(self.root, bg=T["border"], height=1).pack(fill=tk.X, side=tk.TOP)

        # ==================================================================
        # Row 2 — tool strip: Project | Drawing | Selected point
        # ==================================================================
        tools = tk.Frame(self.root, bg=T["bar_bg"], padx=14, pady=6)
        tools.pack(fill=tk.X, side=tk.TOP)

        caption(tools, "Project").pack(side=tk.LEFT, padx=(0, 8))
        FlatButton(
            tools, "Load Slide 1 preset", self.load_slide1_preset, size=11, padx=10, pady=4,
            tooltip="Load the Slide 1 InP taper into all three 2D views",
        ).pack(side=tk.LEFT, padx=(0, 6))
        FlatButton(
            tools, "Clear all", self.clear_all, variant="danger", size=11, padx=10, pady=4,
            tooltip="Remove all points from every view",
        ).pack(side=tk.LEFT)

        vsep(tools).pack(side=tk.LEFT, fill=tk.Y, padx=14, pady=2)

        caption(tools, "Drawing").pack(side=tk.LEFT, padx=(0, 8))
        self.snap_var = tk.BooleanVar(value=True)
        ToggleButton(
            tools, "Snap to guides / grid", self.snap_var, command=self._toggle_snap,
            tooltip="Snap new and dragged points to guide lines and the grid\n(hold Shift to disable temporarily)",
        ).pack(side=tk.LEFT, padx=(0, 6))
        FlatButton(
            tools, "Mirror top (x-y)", self.mirror_xy_top_to_bottom, size=11, padx=10, pady=4,
            tooltip="Mirror the upper (y > 0) outline of the top view across y = 0",
        ).pack(side=tk.LEFT)

        vsep(tools).pack(side=tk.LEFT, fill=tk.Y, padx=14, pady=2)

        # Selected point coordinate editor (exact numeric modification)
        caption(tools, "Point").pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_sel_pt = tk.Label(
            tools,
            text="none selected",
            font=(FONT_UI, 11),
            bg=T["bar_bg"],
            fg=T["text_dim"],
            width=15,
            anchor="w",
        )
        self.lbl_sel_pt.pack(side=tk.LEFT)

        self.lbl_u = tk.Label(tools, text="u", font=(FONT_MONO, 10), bg=T["bar_bg"], fg=T["text_muted"])
        self.lbl_u.pack(side=tk.LEFT, padx=(4, 4))
        self.ent_u = themed_entry(tools, 9)
        self.ent_u.pack(side=tk.LEFT, ipady=2)
        self.ent_u.bind("<Return>", lambda e: self.apply_point_edit())

        self.lbl_v = tk.Label(tools, text="v", font=(FONT_MONO, 10), bg=T["bar_bg"], fg=T["text_muted"])
        self.lbl_v.pack(side=tk.LEFT, padx=(10, 4))
        self.ent_v = themed_entry(tools, 9)
        self.ent_v.pack(side=tk.LEFT, ipady=2)
        self.ent_v.bind("<Return>", lambda e: self.apply_point_edit())

        FlatButton(
            tools, "Apply", self.apply_point_edit, size=11, padx=10, pady=4,
            tooltip="Move the selected point to these coordinates (μm)  (Enter)",
        ).pack(side=tk.LEFT, padx=(8, 4))
        FlatButton(
            tools, "Delete", self.delete_selected_point, variant="danger", size=11, padx=10, pady=4,
            tooltip="Delete the selected point  (Delete / Backspace, or right-click it)",
        ).pack(side=tk.LEFT)

        tk.Frame(self.root, bg=T["border"], height=1).pack(fill=tk.X, side=tk.TOP)

        # ==================================================================
        # Status Bar (packed before the grid so it always keeps its space)
        # ==================================================================
        status_frame = tk.Frame(self.root, bg=T["bar_bg"])
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Frame(status_frame, bg=T["border"], height=1).pack(fill=tk.X, side=tk.TOP)
        self.status_dot = tk.Label(status_frame, text="●", font=(FONT_UI, 11), bg=T["bar_bg"], fg=T["accent"])
        self.status_dot.pack(side=tk.LEFT, padx=(12, 4))
        self.status_lbl = tk.Label(
            status_frame,
            text="",
            font=(FONT_UI, 11),
            bg=T["bar_bg"],
            fg=T["text"],
            anchor="w",
            pady=5,
        )
        self.status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            status_frame,
            text="click: add  ·  drag: move  ·  right-click: delete  ·  scroll: zoom  ·  Shift: no snap",
            font=(FONT_UI, 10),
            bg=T["bar_bg"],
            fg=T["text_dim"],
        ).pack(side=tk.RIGHT, padx=12)
        self.set_status(
            "Draw connected points in the Top (x-y), Side (x-z) and Front (y-z) views — or load the Slide 1 preset.",
        )

        # ==================================================================
        # Main 2x2 Quadrant Grid
        #   (0, 0) Top View (x-y)     |  (0, 1) 3D View (x-y-z)
        #   --------------------------+--------------------------
        #   (1, 0) Side View (x-z)    |  (1, 1) Front View (y-z)
        # ==================================================================
        grid_frame = tk.Frame(self.root, bg=T["app_bg"])
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        grid_frame.columnconfigure(0, weight=1, uniform="col")
        grid_frame.columnconfigure(1, weight=1, uniform="col")
        grid_frame.rowconfigure(0, weight=1, uniform="row")
        grid_frame.rowconfigure(1, weight=1, uniform="row")

        # Key guide coordinates from Slide 1
        x_guides = [-300.0, -110.0, -100.0, 0.0, 100.0, 110.0, 300.0]
        y_guides = [-0.75, -0.25, -0.05, 0.0, 0.05, 0.25, 0.75]
        z_guides = [-0.125, 0.0, 0.125]

        # 1. Top-Left: Top View (x-y)
        self.view_xy = OrthoCanvas2D(
            grid_frame,
            title="Top view (x-y)",
            u_label="x",
            v_label="y",
            u_range=[-340.0, 340.0],
            v_range=[-1.05, 1.05],
            u_guides=x_guides,
            v_guides=y_guides,
            u_grid_step=100.0,
            v_grid_step=0.25,
            on_point_selected=self._on_point_selected,
            on_geometry_changed=self._on_geometry_changed,
        )
        self.view_xy.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        # 2. Bottom-Left: Side View (x-z)
        self.view_xz = OrthoCanvas2D(
            grid_frame,
            title="Side view (x-z)",
            u_label="x",
            v_label="z",
            u_range=[-340.0, 340.0],
            v_range=[-0.28, 0.28],
            u_guides=x_guides,
            v_guides=z_guides,
            u_grid_step=100.0,
            v_grid_step=0.1,
            on_point_selected=self._on_point_selected,
            on_geometry_changed=self._on_geometry_changed,
        )
        self.view_xz.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)

        # 3. Top-Right: 3D View
        self.view_3d = View3DCanvas(grid_frame)
        self.view_3d.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)

        # 4. Bottom-Right: Front View (y-z)
        self.view_yz = OrthoCanvas2D(
            grid_frame,
            title="Front view (y-z)",
            u_label="y",
            v_label="z",
            u_range=[-1.05, 1.05],
            v_range=[-0.28, 0.28],
            u_guides=y_guides,
            v_guides=z_guides,
            u_grid_step=0.25,
            v_grid_step=0.1,
            on_point_selected=self._on_point_selected,
            on_geometry_changed=self._on_geometry_changed,
        )
        self.view_yz.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)

        # Keyboard shortcuts
        self.root.bind("<BackSpace>", lambda e: self._handle_delete_key())
        self.root.bind("<Delete>", lambda e: self._handle_delete_key())
        accel = "Command" if sys.platform == "darwin" else "Control"
        self.root.bind(f"<{accel}-g>", lambda e: self.generate_geometry())
        self.root.bind(f"<{accel}-s>", lambda e: self.generate_file())

    STATUS_COLORS = {"info": "accent", "ok": "success", "warn": "warning", "error": "danger"}

    def set_status(self, text, kind="info"):
        self.status_lbl.config(text=text)
        self.status_dot.config(fg=THEME[self.STATUS_COLORS[kind]])

    def _handle_delete_key(self):
        # Avoid intercepting BackSpace when typing in Entry widgets
        focused = self.root.focus_get()
        if isinstance(focused, tk.Entry):
            return
        self.delete_selected_point()

    def _toggle_snap(self):
        snap = self.snap_var.get()
        self.view_xy.snap_enabled = snap
        self.view_xz.snap_enabled = snap
        self.view_yz.snap_enabled = snap

    def _on_point_selected(self, view_obj, idx, u, v):
        self.active_view = view_obj
        # Deselect in other views for clarity
        for v_canvas in (self.view_xy, self.view_xz, self.view_yz):
            if v_canvas is not view_obj and v_canvas.selected_idx is not None:
                v_canvas.selected_idx = None
                v_canvas.redraw()

        if idx is None:
            self.lbl_sel_pt.config(text="none selected", fg=THEME["text_dim"])
            self.lbl_u.config(text="u")
            self.lbl_v.config(text="v")
            self.ent_u.delete(0, tk.END)
            self.ent_v.delete(0, tk.END)
        else:
            self.lbl_sel_pt.config(text=f"{view_obj.title.split()[0]} view · P{idx+1}", fg=THEME["danger"])
            self.lbl_u.config(text=view_obj.u_label)
            self.lbl_v.config(text=view_obj.v_label)
            self.ent_u.delete(0, tk.END)
            self.ent_u.insert(0, f"{u:g}")
            self.ent_v.delete(0, tk.END)
            self.ent_v.insert(0, f"{v:g}")

    def apply_point_edit(self):
        if self.active_view is None:
            return
        try:
            new_u = float(self.ent_u.get())
            new_v = float(self.ent_v.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric coordinates.")
            return
        self.active_view.update_selected_point(new_u, new_v)
        self.set_status(f"Moved point in {self.active_view.title} to ({new_u:g}, {new_v:g}) μm.")

    def delete_selected_point(self):
        if self.active_view is not None:
            self.active_view.delete_selected_point()

    def _on_geometry_changed(self):
        """Called by the 2D views whenever points are added, moved, deleted or replaced."""
        if self.geometry is not None and not self.geometry_outdated:
            self.geometry_outdated = True
            self.view_3d.set_state("outdated")
            self.set_status(
                "Views changed — the 3D view is outdated. Press Generate geometry to refresh "
                "(exports always use the current drawing).",
                "warn",
            )

    def clear_all(self):
        self.view_xy.clear_points()
        self.view_xz.clear_points()
        self.view_yz.clear_points()
        self.view_3d.clear_3d()
        self.geometry = None
        self.geometry_outdated = False
        self.view_3d.set_state("empty")
        self.set_status("Cleared all views. Draw connected points in (x-y), (x-z) and (y-z).")

    def mirror_xy_top_to_bottom(self):
        """Mirrors the upper profile points (y > 0) in Top view (x-y) across y = 0 to form a symmetric waveguide."""
        pts = self.view_xy.points
        if len(pts) < 2:
            messagebox.showinfo(
                "Mirror Top (x-y)",
                "Draw at least 2 connected points along the upper boundary (y > 0) in Top view (x-y) first.",
            )
            return
        # Keep upper half points (or all points if user only drew one side) sorted by x
        upper = [p for p in pts if p[1] >= 0]
        if len(upper) < 2:
            upper = list(pts)
        upper_sorted = sorted(upper, key=lambda p: p[0])
        lower_sorted = [[p[0], -abs(p[1])] for p in upper_sorted]
        upper_clean = [[p[0], abs(p[1])] for p in reversed(upper_sorted)]
        self.view_xy.set_points(lower_sorted + upper_clean, closed=True)
        self.set_status("Mirrored the Top view (x-y) profile symmetrically across y = 0.")

    def load_slide1_preset(self):
        """Loads the exact Slide 1 geometry into the three orthographic 2D views."""
        self.view_xy.set_points(SLIDE1_PRESET["xy"], closed=True)
        self.view_xz.set_points(SLIDE1_PRESET["xz"], closed=True)
        self.view_yz.set_points(SLIDE1_PRESET["yz"], closed=True)
        self.set_status(
            "Loaded the Slide 1 geometry into all three views. Drag any vertex to modify it, or press Generate geometry."
        )

    # ==================================================================
    # 3D Geometry Reconstruction from the Three 2D Views
    # ==================================================================
    def generate_geometry(self):
        """
        Reconstructs the 3D solid from the current 2D views and renders it in the
        3D view. Returns the Geometry3D, or None if the views are incomplete.
        """
        try:
            geom = reconstruct_geometry(self.view_xy.points, self.view_xz.points, self.view_yz.points)
        except GeometryError as exc:
            self.set_status(f"Cannot generate geometry: {exc}", "error")
            messagebox.showwarning("Cannot Generate Geometry", str(exc))
            return None

        # Mark all 3 views as closed polygons so subsequent clicks edit nicely
        self.view_xy.close_polygon()
        self.view_xz.close_polygon()
        self.view_yz.close_polygon()

        self.geometry = geom
        self.geometry_outdated = False

        xmin, xmax, ymin, ymax, zmin, zmax = geom.bbox
        summary = (
            f"Material  InP     Sections  {len(geom.segments)}\n"
            f"x  [{xmin:g}, {xmax:g}] μm   L = {xmax - xmin:g} μm\n"
            f"y  [{ymin:g}, {ymax:g}] μm   W = {ymax - ymin:g} μm\n"
            f"z  [{zmin:g}, {zmax:g}] μm   T = {(zmax - zmin)*1000:g} nm"
        )
        self.view_3d.set_geometry(geom.faces, geom.bbox, summary_text=summary)
        self.view_3d.set_state("ok")
        self.set_status(
            f"3D geometry generated — {len(geom.segments)} longitudinal section(s), "
            f"L = {xmax-xmin:g} μm, W_max = {ymax-ymin:g} μm, T = {(zmax-zmin)*1000:g} nm.",
            "ok",
        )
        return geom

    # ==================================================================
    # File export (.lsf / .gds / gdsfactory .py)
    # ==================================================================
    def build_lsf_content(self):
        """Builds the Ansys Lumerical .lsf script from the *current* drawing."""
        geom = self.generate_geometry()
        return build_lsf(geom) if geom else None

    def generate_file(self):
        """Exports the current geometry in the format selected in the export bar."""
        # Always rebuild from the current drawing so edits made after the last
        # "Generate geometry" are never lost in the exported file.
        geom = self.generate_geometry()
        if geom is None:
            return

        fmt_label = self.format_var.get()
        exts = EXPORT_FORMATS[fmt_label]
        try:
            layer = parse_layer_spec(self.ent_layer.get())
        except GeometryError as exc:
            messagebox.showerror("Invalid GDS Layer", str(exc))
            return
        cell_name = sanitize_cell_name(self.ent_cell.get())

        file_types = {
            ".lsf": ("Lumerical Script Files", "*.lsf"),
            ".gds": ("GDSII Layout Files", "*.gds"),
            ".py": ("gdsfactory Python Scripts", "*.py"),
        }
        default_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Save {fmt_label}" + (" — base name for all files" if len(exts) > 1 else ""),
            initialdir=default_dir,
            initialfile="generate_structure" + exts[0],
            defaultextension=exts[0],
            filetypes=[file_types[e] for e in exts] + [("All Files", "*.*")],
        )
        if not file_path:
            return

        root_path, chosen_ext = os.path.splitext(file_path)
        if len(exts) == 1:
            if chosen_ext.lower() != exts[0]:
                file_path = root_path + exts[0] if chosen_ext.lower() in file_types else file_path + exts[0]
            targets = [file_path]
        else:
            if chosen_ext.lower() not in file_types:
                root_path = file_path
            targets = [root_path + e for e in exts]
            # The dialog only confirmed overwriting the selected file; ask for the others.
            existing = [t for t in targets if os.path.exists(t) and os.path.abspath(t) != os.path.abspath(file_path)]
            if existing and not messagebox.askyesno(
                "Overwrite Files?",
                "These files already exist and will be overwritten:\n\n"
                + "\n".join(os.path.basename(t) for t in existing),
                parent=self.root,
            ):
                return

        written = []
        try:
            for target in targets:
                written.append(f"{os.path.basename(target)}  —  {export_geometry(geom, target, cell_name, layer)}")
        except Exception as exc:  # report any I/O or export problem instead of failing silently
            messagebox.showerror("Export Failed", f"Could not write {os.path.basename(target)}:\n\n{exc}")
            return

        folder = os.path.dirname(targets[0])
        self.set_status(f"Saved {', '.join(os.path.basename(t) for t in targets)} to {folder}", "ok")
        note = ""
        if ".py" in exts:
            note = "\n\nRun the .py with Python (gdsfactory installed) to generate its .gds; add --show to view it."
        if ".gds" in exts and not geom.z_is_uniform:
            note += "\n\nNote: the side view has an x-dependent thickness, which a 2D GDS file cannot store."
        messagebox.showinfo("Files Generated", f"Saved in {folder}:\n\n" + "\n".join(written) + note)


def export_preset_cli(paths, cell_name, layer_text):
    """Exports the Slide 1 preset geometry without opening the GUI."""
    geom = reconstruct_geometry(SLIDE1_PRESET["xy"], SLIDE1_PRESET["xz"], SLIDE1_PRESET["yz"])
    layer = parse_layer_spec(layer_text)
    for path in paths:
        print(f"{path}: {export_geometry(geom, path, cell_name, layer)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--export-preset",
        nargs="+",
        metavar="FILE",
        help="export the Slide 1 preset to FILE(s) (.lsf, .gds or .py) without opening the GUI",
    )
    parser.add_argument("--layer", default="1/0", help="GDS layer/datatype (default: 1/0)")
    parser.add_argument("--cell", default=DEFAULT_CELL_NAME, help=f"GDS cell name (default: {DEFAULT_CELL_NAME})")
    args = parser.parse_args(argv)

    if args.export_preset:
        try:
            export_preset_cli(args.export_preset, args.cell, args.layer)
        except GeometryError as exc:
            parser.error(str(exc))
        return

    root = tk.Tk()
    LumericalGeometryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
