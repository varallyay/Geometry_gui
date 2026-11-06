#!/usr/bin/env python3
"""
3D Geometry Generation — interactive 4-view 3D geometry builder for
Ansys Lumerical (.lsf), GDSII and gdsfactory
===================================================================
Opens a window divided into four quadrants:
  - Top-Left     : Top view (x-y)
  - Bottom-Left  : Side view / Cross section (x-z)
  - Top-Right    : 3D view (x-y-z, interactive rotation & zoom)
  - Bottom-Right : Front view / Cross section (y-z)

Features:
  - Draw connected polygon points with the mouse pointer in each 2D view
    (Top x-y, Side x-z, Front y-z). The top view gives the width along x, the
    side view the thickness along x, and the front view the cross-section
    shape, which is stretched at every x to the local width and thickness.
  - Modify points at any time after drawing:
      * Drag any vertex with the left mouse button to move it.
      * Click and drag on any polygon edge midpoint (or Shift+click an edge)
        to insert a new vertex between connected points.
      * Right-click (or Ctrl+click / Delete key) on a vertex to remove it.
      * Edit exact numeric coordinates of the selected vertex in the toolbar.
      * Undo / redo every change of the drawn points (Cmd/Ctrl+Z, Cmd+Shift+Z / Ctrl+Y).
      * Mirror any view about its horizontal or vertical axis (⇅ / ⇆ in the view header).
  - The grid, tick labels and snap step follow the zoom level.
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
  - Models (the points of the three views + export settings) are saved and
    opened as .3dgg files (JSON), via the File menu.
  - Help > Check for Updates compares with https://github.com/varallyay/Geometry_gui
    and can install a newer version.
  - On first start (macOS / Linux) it offers a per-user installation and
    creates its configuration in ~/.3dgg/config.json.

Usage:
    python geometry_gui.py [model.3dgg]
    python geometry_gui.py --export-preset out.lsf out.gds out.py   (headless)
    python geometry_gui.py --install | --uninstall | --check-updates | --version
"""

__version__ = "0.3.2"
APP_NAME = "3D Geometry Generation"
APP_SHORT_NAME = "3D Geometry Gen"  # macOS menu bar (CFBundleName, max. ~15 characters)

import argparse
import datetime
import json
import keyword
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
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
    section: list = None  # Front-view cross-section, normalized to [0, 1] x [0, 1] (counter-clockwise)

    @property
    def section_is_rect(self):
        return self.section is None or section_is_rectangle(self.section)

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


def _segments_cross(p1, p2, p3, p4):
    """True if the segments p1-p2 and p3-p4 properly cross (touching ends do not count)."""

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0


def normalized_cross_section(yz_pts):
    """
    Turns the Front view (y-z) polygon into the cross-section shape used along x:
    its points scaled to the unit square [0, 1] x [0, 1] of its bounding box,
    counter-clockwise in the (y, z) plane, without duplicate or collinear points.
    Raises GeometryError for a degenerate or self-crossing outline.
    """
    ys = [p[0] for p in yz_pts]
    zs = [p[1] for p in yz_pts]
    y0, z0 = min(ys), min(zs)
    wy, wz = max(ys) - y0, max(zs) - z0
    if wy < 1e-9 or wz < 1e-9:
        raise GeometryError("The Front view (y-z) polygon has zero area (all points on one line).")
    pts = []
    for y, z in yz_pts:
        p = ((y - y0) / wy, (z - z0) / wz)
        if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-9:
            pts.append(p)
    if len(pts) > 1 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 1e-9:
        pts.pop()
    changed = True
    while changed and len(pts) > 3:  # drop points that lie on a straight edge
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-12:
                del pts[i]
                changed = True
                break
    n = len(pts)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # neighbouring edges
            if _segments_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                raise GeometryError(
                    "The Front view (y-z) outline crosses itself. Please move the points so that "
                    "the cross-section is a simple polygon."
                )
    if abs(polygon_area(pts)) < 1e-9:
        raise GeometryError("The Front view (y-z) polygon has zero area (all points on one line).")
    if polygon_area(pts) < 0:
        pts.reverse()
    return [(round(s, 12), round(t, 12)) for s, t in pts]


def section_is_rectangle(section):
    """True when a normalized cross-section is the full unit square (plain rectangular waveguide)."""
    return len(section) == 4 and {(round(s, 9), round(t, 9)) for s, t in section} == {
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    }


def section_points_3d(section, x, y_bot, y_top, z_bot, z_top):
    """Places the normalized cross-section at x, stretched to the local width and height."""
    return [(x, y_bot + s * (y_top - y_bot), z_bot + t * (z_top - z_bot)) for s, t in section]


# ----------------------------------------------------------------------
# 2D view helpers (GUI independent): axis ticks and mirroring
# ----------------------------------------------------------------------
def nice_tick_step(span, pixels, min_px):
    """Smallest 1/2/5 x 10^n step that keeps grid lines at least min_px pixels apart."""
    raw = abs(span) * min_px / max(float(pixels), 1.0)
    if not raw > 0 or math.isinf(raw):
        return 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw * (1 - 1e-12):
            return m * mag
    return 10 * mag


def minor_tick_step(step):
    """Minor grid / snap step: a quarter of a 2-step, a fifth of a 1- or 5-step."""
    lead = round(step / 10.0 ** math.floor(math.log10(step) + 1e-12))
    return step / (4 if lead == 2 else 5)


def format_tick(value, step):
    """Tick label with just enough decimals for the step (no float noise such as 0.30000000000000004)."""
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    text = f"{round(value / step) * step:.{decimals}f}"
    return "0" if float(text) == 0 else text


def mirror_outline(points, closed=True, axis="v", eps=1e-9):
    """
    Makes a 2D outline symmetric about u = 0 (axis="u") or v = 0 (axis="v").
    The half holding more points off the axis is kept (the positive half on a
    tie) and mirrored onto the other side. Works for closed polygons crossing
    or touching the axis, and for open profiles or closed outlines drawn on one
    side only (e.g. the upper edge of a taper).
    Returns (points, kept_sign) with kept_sign +1 / -1.
    """
    k = 1 if axis == "v" else 0
    pts = [[float(p[0]), float(p[1])] for p in points]
    for p in pts:
        if abs(p[k]) <= eps:
            p[k] = 0.0
    if len(pts) < 2:
        raise GeometryError("Draw at least 2 points before mirroring.")
    pos = sum(1 for p in pts if p[k] > 0)
    neg = sum(1 for p in pts if p[k] < 0)
    if pos == 0 and neg == 0:
        raise GeometryError("All points lie on the mirror axis.")
    sign = 1 if pos >= neg else -1

    def side(p):
        return p[k] * sign  # > 0 on the kept side

    def flip(p):
        q = list(p)
        q[k] = -q[k]
        return q

    # Clip to the kept half-plane (Sutherland-Hodgman); crossings land exactly on the axis
    n = len(pts)
    clipped = []
    for i in range(n if closed else n - 1):
        cur, nxt = pts[i], pts[(i + 1) % n]
        if side(cur) >= 0:
            clipped.append(cur)
        if side(cur) * side(nxt) < 0:
            t = cur[k] / (cur[k] - nxt[k])
            q = [cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])]
            q[k] = 0.0
            clipped.append(q)
    if not closed and side(pts[-1]) >= 0:
        clipped.append(pts[-1])
    chain = []
    for p in clipped:
        if not chain or max(abs(p[0] - chain[-1][0]), abs(p[1] - chain[-1][1])) > eps:
            chain.append(p)
    if closed and len(chain) > 1 and max(abs(chain[0][0] - chain[-1][0]), abs(chain[0][1] - chain[-1][1])) <= eps:
        chain.pop()

    # A closed outline that lies on the axis along one edge is opened at that edge
    if closed:
        m = len(chain)
        for i in range(m):
            if chain[i - 1][k] == 0.0 and chain[i][k] == 0.0:
                chain = chain[i:] + chain[:i]
                break

    result = chain + [flip(p) for p in reversed(chain) if p[k] != 0.0]

    # Drop axis points that only split a straight edge (e.g. at a vertical end face)
    changed = True
    while changed and len(result) > 3:
        changed = False
        for i in range(len(result)):
            a, b, c = result[i - 1], result[i], result[(i + 1) % len(result)]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            scale = math.hypot(c[0] - a[0], c[1] - a[1]) ** 2 or 1.0
            if b[k] == 0.0 and abs(cross) <= 1e-12 * scale:
                del result[i]
                changed = True
                break
    if len(result) < 3:
        raise GeometryError("Mirroring needs points away from the mirror axis.")
    return [[round(u, 9), round(v, 9)] for u, v in result], sign


def reconstruct_geometry(xy_pts, xz_pts, yz_pts):
    """
    Combines the three views into a 3D solid:
      * Top view (x-y):   width y_bot(x) .. y_top(x) along x (absolute),
      * Side view (x-z):  thickness z_bot(x) .. z_top(x) along x (its x-range is
                          stretched onto the x-range of the top view),
      * Front view (y-z): the cross-section shape, stretched at every x to the
                          local width and thickness from the other two views.
    The solid is split into longitudinal segments at every x-vertex of the top
    and side views; within a segment all bounds vary linearly.
    Raises GeometryError for incomplete/degenerate input.
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
    if abs(polygon_area(xz_pts)) < 1e-12:
        raise GeometryError("The Side view (x-z) polygon has zero area (all points on one line).")
    section = normalized_cross_section(yz_pts)

    # 1. x stations: every x-vertex of the top view. Only (near-)coincident
    #    x-values are merged so that short features are not distorted.
    x_stations = []
    for x_val in sorted(p[0] for p in xy_pts):
        if not x_stations or x_val - x_stations[-1] > X_MERGE_TOL_UM:
            x_stations.append(x_val)
    if len(x_stations) < 2:
        raise GeometryError("The Top view (x-y) polygon has no extent along x.")
    xy_xmin, xy_xmax = x_stations[0], x_stations[-1]

    # 2. The side view's x-range is mapped onto the top view; its vertices add stations
    xz_xmin = min(p[0] for p in xz_pts)
    xz_xmax = max(p[0] for p in xz_pts)
    xz_has_x = xz_xmax - xz_xmin > 1e-6
    if xz_has_x:
        to_xy = (xy_xmax - xy_xmin) / (xz_xmax - xz_xmin)
        for u in {p[0] for p in xz_pts}:
            x_val = xy_xmin + (u - xz_xmin) * to_xy
            if all(abs(x_val - xs) > X_MERGE_TOL_UM for xs in x_stations):
                x_stations.append(x_val)
        x_stations.sort()
    xz_zs = [p[1] for p in xz_pts]

    def z_bounds(x0, x1):
        if not xz_has_x:
            return (min(xz_zs), max(xz_zs)), (min(xz_zs), max(xz_zs))
        u0 = xz_xmin + (x0 - xy_xmin) / to_xy
        u1 = xz_xmin + (x1 - xy_xmin) / to_xy
        return interval_slice_bounds(xz_pts, u0, u1)

    segments = []
    for x0, x1 in zip(x_stations[:-1], x_stations[1:]):
        (y0_bot, y0_top), (y1_bot, y1_top) = interval_slice_bounds(xy_pts, x0, x1)
        (z0_bot, z0_top), (z1_bot, z1_top) = z_bounds(x0, x1)
        segments.append(
            {
                k: round(v, 6)
                for k, v in (
                    ("x0", x0), ("x1", x1),
                    ("y0_bot", y0_bot), ("y0_top", y0_top),
                    ("y1_bot", y1_bot), ("y1_top", y1_top),
                    ("z0_bot", z0_bot), ("z0_top", z0_top),
                    ("z1_bot", z1_bot), ("z1_top", z1_top),
                )
            }
        )

    # 3. Preview faces: one quad per cross-section edge and segment, plus end
    #    caps where the cross-section starts, ends or jumps.
    faces_3d = []
    n = len(section)
    prev_end = None
    for idx, seg in enumerate(segments):
        start = section_points_3d(section, seg["x0"], seg["y0_bot"], seg["y0_top"], seg["z0_bot"], seg["z0_top"])
        end = section_points_3d(section, seg["x1"], seg["y1_bot"], seg["y1_top"], seg["z1_bot"], seg["z1_top"])
        for i in range(n):
            j = (i + 1) % n
            ds, dt = section[j][0] - section[i][0], section[j][1] - section[i][1]
            # outward normal of a counter-clockwise edge is (dt, -ds)
            kind = "top" if -ds > abs(dt) else "bottom" if ds > abs(dt) else "side"
            faces_3d.append(([start[i], start[j], end[j], end[i]], kind))
        if prev_end is None or any(
            max(abs(a - b) for a, b in zip(p, q)) > 1e-9 for p, q in zip(prev_end, start)
        ):
            faces_3d.append((start[::-1], "end"))
            if prev_end is not None:
                faces_3d.append((prev_end, "end"))
        if idx == len(segments) - 1:
            faces_3d.append((end, "end"))
        prev_end = end

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
        section=section,
    )


# ----------------------------------------------------------------------
# Ansys Lumerical (.lsf)
# ----------------------------------------------------------------------
_LSF_GROUP_HEADER = [
    "# -------------------------------------------------------------------------",
    "# 2. Build 3D Geometry inside 'InP_Structure' Group",
    "# -------------------------------------------------------------------------",
    "addstructuregroup;",
    'set("name", "InP_Structure");',
    'set("x", 0);',
    'set("y", 0);',
    'set("z", 0);',
    "",
]


def _lsf_planar_solids(geom):
    """
    Script lines for a non-rectangular Front-view cross-section: one Lumerical
    planar solid (closed triangle/polygon mesh) per longitudinal section.
    Facet vertices are ordered counter-clockwise seen from outside.
    """
    f = fmt_num
    sec = geom.section
    n = len(sec)
    lines = [
        f"# Cross-section from the Front view (y-z): {n}-point outline, normalized to its",
        "# bounding box and stretched to the local width (Top view) and thickness (Side view):",
        "#   " + "  ".join(f"({f(s_)}, {f(t_)})" for s_, t_ in sec),
        "",
    ] + _LSF_GROUP_HEADER

    facets = [list(range(n, 0, -1)), list(range(n + 1, 2 * n + 1))]  # start cap (-x), end cap (+x)
    for i in range(n):
        j = (i + 1) % n
        facets.append([i + 1, j + 1, n + j + 1])
        facets.append([i + 1, n + j + 1, n + i + 1])

    for idx, seg in enumerate(geom.segments, start=1):
        start = section_points_3d(sec, seg["x0"], seg["y0_bot"], seg["y0_top"], seg["z0_bot"], seg["z0_top"])
        end = section_points_3d(sec, seg["x1"], seg["y1_bot"], seg["y1_top"], seg["z1_bot"], seg["z1_top"])
        verts = start + end
        len_tag = f"{round(seg['x1'] - seg['x0'], 3):g}".replace(".", "p")
        lines.append(
            f"# Section ({idx}): x=[{f(seg['x0'])}, {f(seg['x1'])}] um, "
            f"width {f(seg['y0_top'] - seg['y0_bot'])} -> {f(seg['y1_top'] - seg['y1_bot'])} um, "
            f"thickness {f(seg['z0_top'] - seg['z0_bot'])} -> {f(seg['z1_top'] - seg['z1_bot'])} um"
        )
        lines.append(f"V_sec_{idx} = [")
        for k, (vx, vy, vz) in enumerate(verts):
            sep = ";" if k < len(verts) - 1 else ""
            lines.append(f"    {f(vx)}, {f(vy)}, {f(vz)}{sep}")
        lines.append("] * um;")
        lines.append(f"F_sec_{idx} = cell({len(facets)});")
        for k, fac in enumerate(facets, start=1):
            lines.append(f"F_sec_{idx}{{{k}}} = cell(1);  F_sec_{idx}{{{k}}}{{1}} = [{', '.join(map(str, fac))}];")
        lines.extend(
            [
                f"addplanarsolid(V_sec_{idx}, F_sec_{idx});",
                f'set("name", "section_{idx}_{len_tag}um");',
                'set("material", mat_name);',
                'addtogroup("InP_Structure");',
                "",
            ]
        )

    x_total = geom.segments[-1]["x1"] - geom.segments[0]["x0"]
    z_min, z_max = geom.z_span
    lines.extend(
        [
            f'?"3D InP structure successfully generated ({len(geom.segments)} planar solids, '
            f'total x-length: {f(x_total)} um, max y-width: {f(geom.bbox[3] - geom.bbox[2])} um, '
            f'z-thickness: {f((z_max - z_min) * 1000.0)} nm).";',
            "",
        ]
    )
    return lines


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
    ]
    if not geom.section_is_rect:
        lines.extend(_lsf_planar_solids(geom))
        return "\n".join(lines)
    lines.extend(
        [
            "# Toggle: 0 = Individual longitudinal sections, 1 = Unified extruded polygon",
            "use_single_polygon = 0;",
            "",
        ]
        + _LSF_GROUP_HEADER
        + [
            "if (use_single_polygon == 1) {",
            "    V_all = [",
        ]
    )
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
    if not geom.section_is_rect:
        z_note += (
            "# NOTE: the Front view (y-z) defines a non-rectangular cross-section. GDS and the\n"
            "#       LayerStack only describe the top-view outline and the thickness; the exact\n"
            "#       3D shape is exported in the Lumerical script (.lsf, planar solids).\n"
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
# Model files (.3dgg): JSON with the points of the three 2D views
# ======================================================================
MODEL_EXT = ".3dgg"
MODEL_FORMAT = "3dgg"
MODEL_FORMAT_VERSION = 1
VIEW_KEYS = ("xy", "xz", "yz")


def model_to_dict(views, export=None, snap=True):
    """
    Builds the .3dgg document.

    views: {"xy": (points, closed), "xz": ..., "yz": ...} with points as [[u, v], ...] in micrometers.
    """
    return {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "generator": f"{APP_NAME} {__version__}",
        "saved": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "units": "um",
        "material": DEFAULT_MATERIAL,
        "views": {
            key: {"points": [[float(u), float(v)] for u, v in views[key][0]], "closed": bool(views[key][1])}
            for key in VIEW_KEYS
        },
        "export": dict(export or {}),
        "editor": {"snap": bool(snap)},
    }


def _write_text_atomic(path, text):
    """Writes text via a temporary file so an interrupted save never leaves a truncated file."""
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_model(path, data):
    _write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_model(path):
    """Reads and validates a .3dgg file. Raises GeometryError with a readable message on problems."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise GeometryError(f"Cannot open {os.path.basename(path)}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise GeometryError(f"{os.path.basename(path)} is not a valid {MODEL_EXT} (JSON) file: {exc}") from exc

    if not isinstance(data, dict) or data.get("format") != MODEL_FORMAT:
        raise GeometryError(f"{os.path.basename(path)} is not a {APP_NAME} model file.")
    file_version = data.get("format_version", 1)
    if not isinstance(file_version, int) or file_version > MODEL_FORMAT_VERSION:
        raise GeometryError(
            f"{os.path.basename(path)} was saved by a newer version ({data.get('generator', 'unknown')}). "
            "Please update the application."
        )

    views = data.get("views")
    if not isinstance(views, dict):
        raise GeometryError("The model file has no 'views' section.")
    clean = {}
    for key in VIEW_KEYS:
        view = views.get(key, {"points": [], "closed": False})
        pts = view.get("points", []) if isinstance(view, dict) else None
        if not isinstance(pts, list):
            raise GeometryError(f"View '{key}' in the model file is malformed.")
        out = []
        for p in pts:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                raise GeometryError(f"View '{key}' contains an invalid point: {p!r}")
            try:
                u, v = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                raise GeometryError(f"View '{key}' contains a non-numeric point: {p!r}") from None
            if not (math.isfinite(u) and math.isfinite(v)):
                raise GeometryError(f"View '{key}' contains a non-finite point: {p!r}")
            out.append([u, v])
        clean[key] = {"points": out, "closed": bool(view.get("closed", len(out) >= 3))}

    export = data.get("export") if isinstance(data.get("export"), dict) else {}
    editor = data.get("editor") if isinstance(data.get("editor"), dict) else {}
    return {**data, "views": clean, "export": export, "editor": editor}


# ======================================================================
# Per-user configuration (~/.3dgg/config.json)
# ======================================================================
CONFIG_DIRNAME = ".3dgg"
MAX_RECENT_FILES = 10


def user_home():
    """Home folder; GG3D_HOME overrides it (used by the tests and for portable setups)."""
    return os.environ.get("GG3D_HOME") or os.path.expanduser("~")


def config_dir(home=None):
    return os.path.join(home or user_home(), CONFIG_DIRNAME)


def config_path(home=None):
    return os.path.join(config_dir(home), "config.json")


def default_config():
    return {
        "config_version": 1,
        "app_version": __version__,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "check_updates_on_start": True,
        "last_update_check": None,
        "skipped_version": None,
        "install": {"prompt": True, "installed": False, "version": None, "date": None, "app_dir": None, "paths": []},
        "recent_files": [],
        "last_model_dir": None,
        "last_export_dir": None,
    }


def load_config(home=None):
    """Returns (config, existed). A corrupt file is kept as config.json.broken and replaced by defaults."""
    cfg = default_config()
    path = config_path(home)
    if not os.path.exists(path):
        return cfg, False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
        if not isinstance(stored, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError):
        try:
            os.replace(path, path + ".broken")
        except OSError:
            pass
        return cfg, False
    for key, value in stored.items():
        if isinstance(cfg.get(key), dict) and isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg, True


def save_config(cfg, home=None):
    os.makedirs(config_dir(home), exist_ok=True)
    cfg["app_version"] = __version__
    _write_text_atomic(config_path(home), json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")


def add_recent_file(cfg, path):
    path = os.path.abspath(path)
    recent = [p for p in cfg.get("recent_files", []) if p != path]
    cfg["recent_files"] = [path] + recent[: MAX_RECENT_FILES - 1]


# ======================================================================
# Update check (GitHub)
# ======================================================================
GITHUB_REPO = "varallyay/Geometry_gui"
PROJECT_URL = f"https://github.com/{GITHUB_REPO}"
UPDATE_BRANCH = "main"
UPDATE_FILE = "geometry_gui.py"
UPDATE_CHECK_INTERVAL_S = 24 * 3600


def parse_version(text):
    """'v0.3.1' -> (0, 3, 1). Missing parts count as 0."""
    nums = [int(n) for n in re.findall(r"\d+", str(text or ""))[:3]]
    return tuple(nums + [0] * (3 - len(nums)))


def is_newer_version(candidate, current=__version__):
    return parse_version(candidate) > parse_version(current)


def read_version_from_source(text):
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    return match.group(1) if match else None


def _http_get(url, timeout=10.0):
    """GET url and return the body bytes. Falls back to certifi's CA bundle if the system one is missing."""
    import ssl
    import urllib.error
    import urllib.request

    headers = {"User-Agent": f"3DGG-update-check/{__version__}"}
    if "api.github.com" in url:
        headers["Accept"] = "application/vnd.github+json"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        if not isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            raise
        try:
            import certifi
        except ImportError:
            raise exc from None
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
            return resp.read()


def fetch_latest_version(timeout=10.0):
    """
    Asks GitHub for the newest version. Uses the latest release if the repository
    has one, otherwise the __version__ of geometry_gui.py on the main branch.
    Returns {"version", "ref", "page", "notes", "source"}.
    """
    import urllib.error

    try:
        data = json.loads(_http_get(f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest", timeout))
        tag = data["tag_name"]
        return {
            "version": tag.lstrip("vV"),
            "ref": tag,
            "page": data.get("html_url") or f"{PROJECT_URL}/releases",
            "notes": (data.get("body") or "").strip(),
            "source": "release",
        }
    except urllib.error.HTTPError as exc:
        if exc.code != 404:  # 404 = no releases published yet
            raise
    source = _http_get(f"https://raw.githubusercontent.com/{GITHUB_REPO}/{UPDATE_BRANCH}/{UPDATE_FILE}", timeout)
    version = read_version_from_source(source.decode("utf-8", "replace"))
    if not version:
        raise GeometryError(f"Could not read the version number of {UPDATE_FILE} on GitHub.")
    return {"version": version, "ref": UPDATE_BRANCH, "page": PROJECT_URL, "notes": "", "source": "branch"}


def download_update(ref, expected_version=None, timeout=30.0):
    """Downloads geometry_gui.py at the given git ref and checks it is a valid, newer program."""
    text = _http_get(f"https://raw.githubusercontent.com/{GITHUB_REPO}/{ref}/{UPDATE_FILE}", timeout).decode("utf-8")
    version = read_version_from_source(text)
    if not version:
        raise GeometryError("The downloaded file has no version number — update aborted.")
    if expected_version and parse_version(version) != parse_version(expected_version):
        raise GeometryError(f"Expected version {expected_version} but downloaded {version} — update aborted.")
    compile(text, UPDATE_FILE, "exec")  # raises SyntaxError for a broken download
    return text, version


def is_git_checkout(path):
    folder = os.path.dirname(os.path.abspath(path))
    return os.path.exists(os.path.join(folder, ".git"))


def apply_update(text, target, home=None):
    """Replaces target with the new program text; the old file is kept in ~/.3dgg/backups."""
    backups = os.path.join(config_dir(home), "backups")
    os.makedirs(backups, exist_ok=True)
    backup = os.path.join(backups, f"geometry_gui-{__version__}.py")
    shutil.copy2(target, backup)
    _write_text_atomic(target, text)
    os.chmod(target, 0o755)
    return backup


# ======================================================================
# Per-user installation (macOS: ~/Applications/<app>.app, Linux: XDG desktop entry)
# ======================================================================
BUNDLE_ID = "io.github.varallyay.3dgg"
MODEL_UTI = BUNDLE_ID + ".model"
MODEL_MIME = "application/x-3dgg"
ICON_FILES = ("3dgg_icon.png", "3dgg_icon_256.png", "3dgg.icns")


def source_dir():
    return os.path.dirname(os.path.abspath(__file__))


def asset_path(name, base=None):
    path = os.path.join(base or source_dir(), "assets", name)
    return path if os.path.exists(path) else None


def installed_app_dir(home=None):
    return os.path.join(config_dir(home), "app")


def is_running_installed_copy(home=None):
    return os.path.abspath(source_dir()) == os.path.abspath(installed_app_dir(home))


def _write_executable(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, 0o755)


def _launcher_script(python, script):
    return f'#!/bin/sh\nexec {shlex.quote(python)} {shlex.quote(script)} "$@"\n'


def _copy_program(app_dir, src_dir):
    """Copies the program and its icons into app_dir (~/.3dgg/app)."""
    os.makedirs(os.path.join(app_dir, "assets"), exist_ok=True)
    if os.path.abspath(src_dir) != os.path.abspath(app_dir):
        shutil.copy2(os.path.join(src_dir, UPDATE_FILE), os.path.join(app_dir, UPDATE_FILE))
        for name in ICON_FILES:
            src = asset_path(name, src_dir)
            if src:
                shutil.copy2(src, os.path.join(app_dir, "assets", name))
    return os.path.join(app_dir, UPDATE_FILE)


def _install_macos(home, python, script, app_dir, register):
    import plistlib

    bundle = os.path.join(home, "Applications", f"{APP_NAME}.app")
    if os.path.exists(bundle):
        info = os.path.join(bundle, "Contents", "Info.plist")
        try:
            with open(info, "rb") as fh:
                ours = plistlib.load(fh).get("CFBundleIdentifier") == BUNDLE_ID
        except (OSError, plistlib.InvalidFileException):
            ours = False
        if not ours:
            raise GeometryError(f"{bundle} already exists and was not created by this program.")
        shutil.rmtree(bundle)

    macos_dir = os.path.join(bundle, "Contents", "MacOS")
    res_dir = os.path.join(bundle, "Contents", "Resources")
    os.makedirs(macos_dir)
    os.makedirs(res_dir)
    _write_executable(os.path.join(macos_dir, "3dgg"), _launcher_script(python, script))
    icns = os.path.join(app_dir, "assets", "3dgg.icns")
    if os.path.exists(icns):
        shutil.copy2(icns, os.path.join(res_dir, "3dgg.icns"))

    doc_name = f"{APP_NAME} model"
    plist = {
        "CFBundleName": APP_SHORT_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": __version__,
        "CFBundleShortVersionString": __version__,
        "CFBundleExecutable": "3dgg",
        "CFBundleIconFile": "3dgg.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "10.13",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": doc_name,
                "CFBundleTypeRole": "Editor",
                "LSHandlerRank": "Owner",
                "LSItemContentTypes": [MODEL_UTI],
                "CFBundleTypeIconFile": "3dgg.icns",
            }
        ],
        "UTExportedTypeDeclarations": [
            {
                "UTTypeIdentifier": MODEL_UTI,
                "UTTypeDescription": doc_name,
                "UTTypeConformsTo": ["public.json", "public.data"],
                "UTTypeIconFile": "3dgg.icns",
                "UTTypeTagSpecification": {
                    "public.filename-extension": [MODEL_EXT.lstrip(".")],
                    "public.mime-type": MODEL_MIME,
                },
            }
        ],
    }
    with open(os.path.join(bundle, "Contents", "Info.plist"), "wb") as fh:
        plistlib.dump(plist, fh)

    lsregister = (
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
        "LaunchServices.framework/Support/lsregister"
    )
    if register and os.path.exists(lsregister):
        subprocess.run([lsregister, "-f", bundle], check=False, capture_output=True)
    return [bundle], bundle


def _install_linux(home, python, script, app_dir, register):
    share = os.path.join(home, ".local", "share")
    launcher = os.path.join(home, ".local", "bin", "3dgg")
    desktop = os.path.join(share, "applications", "3dgg.desktop")
    mime_xml = os.path.join(share, "mime", "packages", "3dgg.xml")
    icon_app = os.path.join(share, "icons", "hicolor", "256x256", "apps", "3dgg.png")
    icon_mime = os.path.join(share, "icons", "hicolor", "256x256", "mimetypes", "application-x-3dgg.png")
    for path in (launcher, desktop, mime_xml, icon_app, icon_mime):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    _write_executable(launcher, _launcher_script(python, script))
    icon_src = os.path.join(app_dir, "assets", "3dgg_icon_256.png")
    created = [launcher, desktop, mime_xml]
    if os.path.exists(icon_src):
        shutil.copy2(icon_src, icon_app)
        shutil.copy2(icon_src, icon_mime)
        created += [icon_app, icon_mime]

    with open(desktop, "w", encoding="utf-8") as fh:
        fh.write(
            "[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={APP_NAME}\n"
            "Comment=Draw 2D views and generate 3D geometry for Lumerical, GDSII and gdsfactory\n"
            f"Exec={shlex.quote(launcher)} %f\n"
            "Icon=3dgg\n"
            "Terminal=false\n"
            "Categories=Science;Engineering;Graphics;\n"
            f"MimeType={MODEL_MIME};\n"
            "StartupWMClass=3dgg\n"
        )
    with open(mime_xml, "w", encoding="utf-8") as fh:
        fh.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">\n'
            f'  <mime-type type="{MODEL_MIME}">\n'
            f"    <comment>{APP_NAME} model</comment>\n"
            '    <sub-class-of type="application/json"/>\n'
            f'    <glob pattern="*{MODEL_EXT}"/>\n'
            "    <icon name=\"application-x-3dgg\"/>\n"
            "  </mime-type>\n"
            "</mime-info>\n"
        )
    if register:
        for cmd in (
            ["update-mime-database", os.path.join(share, "mime")],
            ["update-desktop-database", os.path.join(share, "applications")],
            ["gtk-update-icon-cache", "-q", "-t", os.path.join(share, "icons", "hicolor")],
        ):
            if shutil.which(cmd[0]):
                subprocess.run(cmd, check=False, capture_output=True)
    return created, launcher


def install_app(home=None, platform=None, python=None, src_dir=None, register=True):
    """
    Installs the application for the current user.
      * copies the program and icons to ~/.3dgg/app
      * macOS: creates ~/Applications/3D Geometry Generation.app (opens .3dgg files)
      * Linux: ~/.local/bin/3dgg, a desktop entry, icon and the .3dgg MIME type
    Returns {"app_dir", "paths", "launcher"}.
    """
    home = home or user_home()
    platform = platform or sys.platform
    python = python or sys.executable
    if platform == "darwin":
        installer = _install_macos
    elif platform.startswith("linux"):
        installer = _install_linux
    else:
        raise GeometryError(f"Installation is supported on macOS and Linux only (this system: {platform}).")

    app_dir = installed_app_dir(home)
    script = _copy_program(app_dir, src_dir or source_dir())
    paths, launcher = installer(home, python, script, app_dir, register)
    return {"app_dir": app_dir, "paths": [app_dir] + paths, "launcher": launcher}


def uninstall_app(paths, home=None):
    """Removes the paths created by install_app (only ones inside the home folder). Returns removed paths."""
    home = os.path.abspath(home or user_home())
    removed = []
    for path in paths:
        full = os.path.abspath(path)
        if not full.startswith(home + os.sep) or full == home or not os.path.lexists(full):
            continue
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        else:
            os.remove(full)
        removed.append(full)
    return removed


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
    "grid_minor": "#181c22",
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


class ChoiceDialog(tk.Toplevel):
    """
    Dark modal dialog with a title, message and a row of buttons.
    buttons: list of (key, label, variant); returns the chosen key (or cancel_key on Esc / close).
    """

    def __init__(self, parent, title, message, buttons, default=None, cancel_key=None, detail=None, width=460):
        super().__init__(parent)
        self.withdraw()
        self.title(title)
        self.configure(bg=THEME["bar_bg"], padx=22, pady=18)
        self.resizable(False, False)
        self.transient(parent)
        self.result = cancel_key

        tk.Label(
            self, text=title, font=(FONT_UI, 14, "bold"), bg=THEME["bar_bg"], fg=THEME["text"], anchor="w",
            justify=tk.LEFT,
        ).pack(fill=tk.X)
        tk.Label(
            self, text=message, font=(FONT_UI, 11), bg=THEME["bar_bg"], fg=THEME["text"], anchor="w",
            justify=tk.LEFT, wraplength=width,
        ).pack(fill=tk.X, pady=(8, 0))
        if detail:
            tk.Label(
                self, text=detail, font=(FONT_UI, 10), bg=THEME["bar_bg"], fg=THEME["text_muted"], anchor="w",
                justify=tk.LEFT, wraplength=width,
            ).pack(fill=tk.X, pady=(8, 0))

        row = tk.Frame(self, bg=THEME["bar_bg"])
        row.pack(fill=tk.X, pady=(18, 0))
        # First button on the right is the default (macOS / GNOME convention)
        for key, label, variant in buttons:
            FlatButton(row, label, lambda k=key: self._choose(k), variant=variant, size=11, padx=14, pady=6).pack(
                side=tk.RIGHT, padx=(8, 0)
            )

        self.bind("<Escape>", lambda e: self._choose(cancel_key))
        if default is not None:
            self.bind("<Return>", lambda e: self._choose(default))
        self.protocol("WM_DELETE_WINDOW", lambda: self._choose(cancel_key))

        # Centre over the parent window
        self.update_idletasks()
        px, py = parent.winfo_rootx(), parent.winfo_rooty()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        self.geometry(f"+{px + max(0, (pw - w) // 2)}+{py + max(0, (ph - h) // 3)}")
        self.deiconify()
        self.lift()
        self.focus_force()
        self.grab_set()

    def _choose(self, key):
        self.result = key
        self.grab_release()
        self.destroy()

    @classmethod
    def ask(cls, parent, *args, **kwargs):
        dialog = cls(parent, *args, **kwargs)
        parent.wait_window(dialog)
        return dialog.result


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
        on_point_selected=None,
        on_geometry_changed=None,
        on_edit_finished=None,
        on_mirror=None,
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
        self.on_point_selected = on_point_selected
        self.on_geometry_changed = on_geometry_changed
        self.on_edit_finished = on_edit_finished  # end of one undoable edit (e.g. mouse released)
        self.on_mirror = on_mirror

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
            anchor="w",
            font=(FONT_MONO, 10),
            bg=THEME["panel_bg"],
            fg=THEME["text_muted"],
        )

        for text, cmd, tip in (
            ("Clear", self.clear_points, "Remove all points from this view"),
            ("Close shape", self.close_polygon, "Close the polygon and switch to edit mode\n(drag vertices, click an edge to insert a point)"),
            ("Fit", self.reset_view, "Reset zoom / pan of this view"),
            (
                "⇆ Mirror",
                lambda: self._mirror("u"),
                f"Make the outline symmetric about {u_label} = 0:\nthe half with more points is mirrored onto the other side",
            ),
            (
                "⇅ Mirror",
                lambda: self._mirror("v"),
                f"Make the outline symmetric about {v_label} = 0:\nthe half with more points is mirrored onto the other side",
            ),
        ):
            FlatButton(header, text, cmd, variant="ghost", size=10, padx=7, pady=3, tooltip=tip).pack(
                side=tk.RIGHT, padx=(0, 3) if text != "Clear" else (0, 8), pady=4
            )
        # packed last: the cursor readout gives way first when the window is narrow
        self.coord_lbl.pack(side=tk.LEFT, padx=4)

        # Drawing Canvas
        self.canvas = tk.Canvas(
            self,
            bg=THEME["plot_bg"],
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Margins for axis labels
        self.margin_left = 64
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
            return round(u, 6), round(v, 6)

        u_span = self.u_range[1] - self.u_range[0]
        v_span = self.v_range[1] - self.v_range[0]

        # Snap to the nearest key guide coordinate if within 2.5% (u) / 3% (v) of span
        def nearest_guide(val, guides, tol):
            best = min(guides, key=lambda g: abs(val - g), default=None)
            return best if best is not None and abs(val - best) <= tol else None

        # Guides catch the pointer within 12 px; otherwise snap to the minor grid of the current zoom
        x0, y0, x1, y1 = self._plot_rect()
        (_, fine_u), (_, fine_v) = self.tick_steps()
        snapped_u = nearest_guide(u, self.u_guides, 12.0 * u_span / max(1.0, x1 - x0))
        if snapped_u is None:
            snapped_u = round(u / fine_u) * fine_u

        snapped_v = nearest_guide(v, self.v_guides, 12.0 * v_span / max(1.0, y1 - y0))
        if snapped_v is None:
            snapped_v = round(v / fine_v) * fine_v

        return round(snapped_u, 9), round(snapped_v, 9)

    def tick_steps(self):
        """((major, minor) along u, (major, minor) along v) for the current zoom and canvas size."""
        x0, y0, x1, y1 = self._plot_rect()
        su = nice_tick_step(self.u_range[1] - self.u_range[0], x1 - x0, 80)
        sv = nice_tick_step(self.v_range[1] - self.v_range[0], y1 - y0, 40)
        return (su, minor_tick_step(su)), (sv, minor_tick_step(sv))

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
        self.canvas.focus_set()  # so Delete / Backspace act on the points, not on a text field
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
        was_dragging = self.dragging_idx is not None
        self.dragging_idx = None
        self.redraw()
        if was_dragging:
            self._edit_finished()

    def _edit_finished(self):
        if self.on_edit_finished:
            self.on_edit_finished()

    def _on_double_click(self, event):
        if len(self.points) >= 3:
            self.closed = True
            self.redraw()
            self._edit_finished()

    def _mirror(self, axis):
        if self.on_mirror:
            self.on_mirror(self, axis)

    def mirror(self, axis):
        """Makes the outline symmetric about u = 0 (axis 'u') or v = 0 (axis 'v'). Returns the kept sign."""
        pts, sign = mirror_outline(self.points, self.closed, axis)
        self.set_points(pts, closed=True)
        return sign

    def _on_right_click(self, event):
        self.canvas.focus_set()
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
        self._edit_finished()

    def _on_mousewheel(self, event):
        # Zoom around cursor position
        u_c, v_c = self.px_to_uv(event.x, event.y)
        factor = 0.85 if event.delta > 0 else 1.18
        u_min, u_max = self.u_range
        v_min, v_max = self.v_range
        # Limits: down to 1 nm across the view, out to 1000x the default view
        spans = ((u_max - u_min, self.default_u_range), (v_max - v_min, self.default_v_range))
        if any(sp * factor < 1e-3 or sp * factor > 1000 * (d[1] - d[0]) for sp, d in spans):
            return
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

        # 1. Grid adapted to the zoom: faint minor lines (= snap grid), major lines with
        #    tick labels (1/2/5 x 10^n, kept ~80 px apart along u and ~40 px along v),
        #    zero axis emphasised
        (step_u, minor_u), (step_v, minor_v) = self.tick_steps()
        for minor, step, lo, hi, vertical in (
            (minor_u, step_u, u_min, u_max, True),
            (minor_v, step_v, v_min, v_max, False),
        ):
            per_major = round(step / minor)
            for i in range(math.ceil(lo / minor - 1e-9), math.floor(hi / minor + 1e-9) + 1):
                val = i * minor
                is_major = i % per_major == 0
                if vertical:
                    px, _ = self.uv_to_px(val, 0)
                    line = (px, y0, px, y1)
                else:
                    _, py = self.uv_to_px(0, val)
                    line = (x0, py, x1, py)
                color = T["grid_axis"] if i == 0 else T["grid"] if is_major else T["grid_minor"]
                c.create_line(*line, fill=color, width=1)
                if is_major:
                    label = format_tick(val, step)
                    if vertical:
                        c.create_text(px, y1 + 12, text=label, font=tick_font, fill=T["text_muted"])
                    else:
                        c.create_text(x0 - 6, py, text=label, font=tick_font, fill=T["text_muted"], anchor="e")

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

    def __init__(self, root, config=None, startup_tasks=False):
        """
        config: the per-user config dict (default: loaded from ~/.3dgg/config.json).
        startup_tasks: run the first-start install offer and the automatic update check
                       (enabled by main(); off for tests / embedding).
        """
        self.root = root
        self.config = config if config is not None else load_config()[0]
        self.model_path = None  # .3dgg file of the current model
        self.dirty = False  # unsaved changes
        self._last_shortcut = {}
        self._update_result = None
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
        self._undo_stack = []  # earlier states of the three views (see _view_state)
        self._redo_stack = []
        self._commit_pending = False

        self._build_ui()
        self._history_state = self._view_state()
        self._build_menu()
        self._set_window_icon()
        self._update_title()
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        if sys.platform == "darwin":
            # Finder "Open With" / double-click on a .3dgg file, About and Quit from the app menu
            self.root.createcommand("::tk::mac::OpenDocument", self._mac_open_document)
            self.root.createcommand("tkAboutDialog", self.show_about)
            self.root.createcommand("::tk::mac::Quit", self.quit_app)
        if startup_tasks:
            self.root.after(700, self._run_startup_tasks)

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
            title_box, text=APP_NAME, font=(FONT_UI, 15, "bold"), bg=T["bar_bg"], fg=T["text"]
        ).pack(side=tk.LEFT)
        tk.Label(
            title_box, text=f"  v{__version__}", font=(FONT_UI, 11), bg=T["bar_bg"], fg=T["text_dim"]
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
            tooltip=f"Step 2 — rebuild from the current drawing and save\nin the selected export format  ({mod}+E)",
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
        self.snap_btn = ToggleButton(
            tools, "Snap to guides / grid", self.snap_var, command=self._toggle_snap,
            tooltip="Snap new and dragged points to guide lines and the grid\n(hold Shift to disable temporarily)",
        )
        self.snap_btn.pack(side=tk.LEFT, padx=(0, 6))
        mod = "⌘" if sys.platform == "darwin" else "Ctrl+"
        FlatButton(
            tools, "↶ Undo", self.undo, size=11, padx=10, pady=4,
            tooltip=f"Undo the last change of the drawn points  ({mod}Z)",
        ).pack(side=tk.LEFT, padx=(0, 4))
        FlatButton(
            tools, "↷ Redo", self.redo, size=11, padx=10, pady=4,
            tooltip=f"Redo the change that was undone  ({mod}{'⇧Z' if sys.platform == 'darwin' else 'Y'})",
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
            on_point_selected=self._on_point_selected,
            on_geometry_changed=self._on_geometry_changed,
            on_edit_finished=self._commit_history,
            on_mirror=self.mirror_view,
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
            on_point_selected=self._on_point_selected,
            on_geometry_changed=self._on_geometry_changed,
            on_edit_finished=self._commit_history,
            on_mirror=self.mirror_view,
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
            on_point_selected=self._on_point_selected,
            on_geometry_changed=self._on_geometry_changed,
            on_edit_finished=self._commit_history,
            on_mirror=self.mirror_view,
        )
        self.view_yz.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)

        # Keyboard shortcuts
        self.root.bind("<BackSpace>", lambda e: self._handle_delete_key())
        self.root.bind("<Delete>", lambda e: self._handle_delete_key())
        accel = "Command" if sys.platform == "darwin" else "Control"
        self._accel = accel

    # ==================================================================
    # Menu bar & keyboard shortcuts
    # ==================================================================
    def _shortcut(self, name, func):
        """
        Runs a menu / shortcut action once. On macOS a menu key equivalent and the
        matching key binding can both fire for one key press; the second call
        (arriving right after the first finished) is ignored.
        """
        now = time.monotonic()
        if now - self._last_shortcut.get(name, -1.0) < 0.35:
            return "break"
        try:
            func()
        finally:
            self._last_shortcut[name] = time.monotonic()
        return "break"

    def _build_menu(self):
        mac = sys.platform == "darwin"
        accel_label = "Command-" if mac else "Ctrl+"
        menubar = tk.Menu(self.root)

        def add(menu, label, name, func, key=None, shift=False):
            accel = None
            if key:
                accel = f"{accel_label}{'Shift-' if mac and shift else ('Shift+' if shift else '')}{key.upper()}"
                keysym = key.upper() if shift else key.lower()
                self.root.bind(f"<{self._accel}-{keysym}>", lambda e: self._shortcut(name, func))
            menu.add_command(label=label, accelerator=accel, command=lambda: self._shortcut(name, func))

        if mac:
            app_menu = tk.Menu(menubar, name="apple", tearoff=False)
            app_menu.add_command(label=f"About {APP_NAME}", command=self.show_about)
            app_menu.add_command(label="Check for Updates…", command=lambda: self.check_for_updates(manual=True))
            app_menu.add_separator()
            menubar.add_cascade(menu=app_menu)

        file_menu = tk.Menu(menubar, tearoff=False)
        add(file_menu, "New", "new", self.new_model, "n")
        add(file_menu, "Open…", "open", self.open_model, "o")
        self.recent_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Open Recent", menu=self.recent_menu)
        file_menu.add_separator()
        add(file_menu, "Save", "save", self.save_model, "s")
        add(file_menu, "Save As…", "save_as", self.save_model_as, "s", shift=True)
        file_menu.add_separator()
        add(file_menu, "Export (Generate file)…", "export", self.generate_file, "e")
        if not mac:
            file_menu.add_separator()
            add(file_menu, "Quit", "quit", self.quit_app, "q")
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        add(edit_menu, "Undo", "undo", self.undo, "z")
        add(edit_menu, "Redo", "redo", self.redo, "z", shift=True)
        if not mac:
            self.root.bind("<Control-y>", lambda e: self._shortcut("redo", self.redo))
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete Selected Point", accelerator="Delete", command=self.delete_selected_point)
        mirror_menu = tk.Menu(edit_menu, tearoff=False)
        for view in self._views().values():
            name = view.title.split(" (")[0]
            for axis, label in (("v", view.v_label), ("u", view.u_label)):
                mirror_menu.add_command(
                    label=f"{name}: symmetric about {label} = 0",
                    command=lambda v=view, a=axis: self.mirror_view(v, a),
                )
        edit_menu.add_cascade(label="Mirror", menu=mirror_menu)
        edit_menu.add_separator()
        edit_menu.add_command(label="Load Slide 1 Preset", command=self.load_slide1_preset)
        edit_menu.add_command(label="Clear All Views", command=self.clear_all)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        geom_menu = tk.Menu(menubar, tearoff=False)
        add(geom_menu, "Generate Geometry", "generate", self.generate_geometry, "g")
        geom_menu.add_command(label="Reset All Views", command=self.reset_all_views)
        menubar.add_cascade(label="Geometry", menu=geom_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Check for Updates…", command=lambda: self.check_for_updates(manual=True))
        self.check_updates_var = tk.BooleanVar(value=bool(self.config.get("check_updates_on_start", True)))
        help_menu.add_checkbutton(
            label="Check for Updates at Startup", variable=self.check_updates_var, command=self._toggle_update_check
        )
        help_menu.add_separator()
        help_menu.add_command(label="Install for This User…", command=self.offer_install)
        help_menu.add_command(label="Uninstall…", command=self.uninstall_application)
        self._uninstall_index = help_menu.index("end")
        help_menu.add_separator()
        help_menu.add_command(label="Open Configuration Folder", command=self.open_config_folder)
        help_menu.add_command(label="Project Page on GitHub", command=self.open_project_page)
        if not mac:
            help_menu.add_separator()
            help_menu.add_command(label=f"About {APP_NAME}", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.help_menu = help_menu

        self.root.config(menu=menubar)
        self._rebuild_recent_menu()
        self._refresh_install_menu()

        # Export settings belong to the model: editing them marks it as changed
        self.format_var.trace_add("write", lambda *a: self._set_dirty(True))
        mark = self.root.register(lambda: self._set_dirty(True) or True)
        for entry in (self.ent_layer, self.ent_cell):
            entry.config(validate="key", validatecommand=(mark,))

    def _rebuild_recent_menu(self):
        self.recent_menu.delete(0, "end")
        recent = self.config.get("recent_files", [])
        for path in recent:
            self.recent_menu.add_command(
                label=os.path.basename(path), command=lambda p=path: self.open_model_path(p)
            )
        if recent:
            self.recent_menu.add_separator()
            self.recent_menu.add_command(label="Clear Menu", command=self._clear_recent)
        else:
            self.recent_menu.add_command(label="No recent models", state=tk.DISABLED)

    def _clear_recent(self):
        self.config["recent_files"] = []
        self._save_config()
        self._rebuild_recent_menu()

    def _refresh_install_menu(self):
        installed = bool(self.config.get("install", {}).get("installed"))
        self.help_menu.entryconfig(self._uninstall_index, state=tk.NORMAL if installed else tk.DISABLED)

    def _set_window_icon(self):
        path = asset_path("3dgg_icon_256.png")
        if not path:
            return
        try:
            self._icon_img = tk.PhotoImage(file=path)
            self.root.iconphoto(True, self._icon_img)
        except tk.TclError:
            pass

    def _save_config(self):
        try:
            save_config(self.config)
        except OSError:
            pass  # the config is a convenience; never block the user on it

    def reset_all_views(self):
        for view in self._views().values():
            view.reset_view()
        self.view_3d.reset_camera()

    # ==================================================================
    # Model files (.3dgg)
    # ==================================================================
    def _views(self):
        return {"xy": self.view_xy, "xz": self.view_xz, "yz": self.view_yz}

    def _set_dirty(self, dirty):
        if dirty != self.dirty:
            self.dirty = dirty
            self._update_title()

    def _update_title(self):
        name = os.path.basename(self.model_path) if self.model_path else "Untitled"
        self.root.title(f"{name}{' — edited' if self.dirty else ''}  ·  {APP_NAME} {__version__}")
        if sys.platform == "darwin":
            try:
                self.root.wm_attributes("-modified", self.dirty)
            except tk.TclError:
                pass

    def _model_dir(self):
        if self.model_path:
            return os.path.dirname(self.model_path)
        last = self.config.get("last_model_dir")
        return last if last and os.path.isdir(last) else os.path.expanduser("~")

    def model_data(self):
        """The current model as a .3dgg document (dict)."""
        views = {key: (view.points, view.closed) for key, view in self._views().items()}
        export = {
            "format": self.format_var.get(),
            "gds_layer": self.ent_layer.get().strip(),
            "cell_name": self.ent_cell.get().strip(),
        }
        return model_to_dict(views, export, self.snap_var.get())

    def apply_model(self, data):
        """Loads a validated .3dgg document (see read_model) into the views and export settings."""
        for key, view in self._views().items():
            view.set_points(data["views"][key]["points"], closed=data["views"][key]["closed"])
            view.reset_view()
        export = data.get("export", {})
        if export.get("format") in EXPORT_FORMATS:
            self.format_var.set(export["format"])
        for entry, key in ((self.ent_layer, "gds_layer"), (self.ent_cell, "cell_name")):
            if isinstance(export.get(key), str) and export[key].strip():
                entry.delete(0, tk.END)
                entry.insert(0, export[key].strip())
        if "snap" in data.get("editor", {}):
            self.snap_var.set(bool(data["editor"]["snap"]))
            self.snap_btn._refresh()
            self._toggle_snap()

        self.view_3d.clear_3d()
        self.view_3d.set_state("empty")
        self.geometry = None
        self.geometry_outdated = False
        if all(len(v.points) >= 3 for v in self._views().values()):
            self.generate_geometry(quiet=True)

    def confirm_discard(self):
        """Asks to save unsaved changes. Returns False if the user cancelled."""
        if not self.dirty:
            return True
        name = os.path.basename(self.model_path) if self.model_path else "Untitled"
        choice = ChoiceDialog.ask(
            self.root,
            "Save changes?",
            f"Do you want to save the changes made to “{name}”?",
            [("save", "Save", "primary"), ("discard", "Don't save", "secondary"), ("cancel", "Cancel", "secondary")],
            default="save",
            cancel_key="cancel",
            detail="Your changes will be lost if you don't save them.",
        )
        if choice == "save":
            return self.save_model()
        return choice == "discard"

    def new_model(self):
        if not self.confirm_discard():
            return
        self.clear_all()
        self.format_var.set(next(iter(EXPORT_FORMATS)))
        for entry, value in ((self.ent_layer, f"{DEFAULT_GDS_LAYER[0]}/{DEFAULT_GDS_LAYER[1]}"), (self.ent_cell, DEFAULT_CELL_NAME)):
            entry.delete(0, tk.END)
            entry.insert(0, value)
        self.reset_all_views()
        self.reset_history()
        self.model_path = None
        self._set_dirty(False)
        self._update_title()
        self.set_status("New model. Draw connected points in the Top (x-y), Side (x-z) and Front (y-z) views.")

    def open_model(self):
        if not self.confirm_discard():
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title=f"Open {APP_NAME} Model",
            initialdir=self._model_dir(),
            filetypes=[(f"{APP_NAME} model", f"*{MODEL_EXT}"), ("All files", "*")],
        )
        if path:
            self.open_model_path(path, confirm=False)

    def open_model_path(self, path, confirm=True):
        """Opens a .3dgg file. Returns True on success."""
        if confirm and not self.confirm_discard():
            return False
        try:
            data = read_model(path)
        except GeometryError as exc:
            if not os.path.exists(path) and path in self.config.get("recent_files", []):
                self.config["recent_files"].remove(path)
                self._save_config()
                self._rebuild_recent_menu()
            self.set_status(str(exc), "error")
            messagebox.showerror("Cannot Open Model", str(exc), parent=self.root)
            return False

        self.apply_model(data)
        self.reset_history()
        self.model_path = os.path.abspath(path)
        self._set_dirty(False)
        self._update_title()
        self._remember_model_path()
        self.set_status(f"Opened {os.path.basename(path)}.", "ok")
        return True

    def save_model(self):
        """Saves to the current file (asks for a name the first time). Returns True if saved."""
        if not self.model_path:
            return self.save_model_as()
        return self.save_model_to(self.model_path)

    def save_model_as(self):
        initial = os.path.basename(self.model_path) if self.model_path else f"untitled{MODEL_EXT}"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Model As",
            initialdir=self._model_dir(),
            initialfile=initial,
            defaultextension=MODEL_EXT,
            filetypes=[(f"{APP_NAME} model", f"*{MODEL_EXT}")],
        )
        if not path:
            return False
        if not path.lower().endswith(MODEL_EXT):
            path += MODEL_EXT
        return self.save_model_to(path)

    def save_model_to(self, path):
        try:
            write_model(path, self.model_data())
        except OSError as exc:
            self.set_status(f"Could not save {os.path.basename(path)}: {exc}", "error")
            messagebox.showerror("Save Failed", f"Could not save {os.path.basename(path)}:\n\n{exc}", parent=self.root)
            return False
        self.model_path = os.path.abspath(path)
        self._set_dirty(False)
        self._update_title()
        self._remember_model_path()
        self.set_status(f"Saved model {os.path.basename(path)} in {os.path.dirname(self.model_path)}", "ok")
        return True

    def _remember_model_path(self):
        add_recent_file(self.config, self.model_path)
        self.config["last_model_dir"] = os.path.dirname(self.model_path)
        self._save_config()
        self._rebuild_recent_menu()

    def _mac_open_document(self, *paths):
        for path in paths:
            if path.lower().endswith(MODEL_EXT):
                self.open_model_path(path)
                break

    def quit_app(self):
        if not self.confirm_discard():
            return
        self._save_config()
        self.root.destroy()

    # ==================================================================
    # About / project page / config folder
    # ==================================================================
    def show_about(self):
        inst = self.config.get("install", {})
        installed = f"yes — version {inst.get('version')}" if inst.get("installed") else "no (running from this folder)"
        choice = ChoiceDialog.ask(
            self.root,
            APP_NAME,
            f"Version {__version__}\n\nDraw the top, side and front views of a structure and export it as an "
            "Ansys Lumerical script (.lsf), GDSII layout (.gds) or gdsfactory script (.py).",
            [("ok", "OK", "primary"), ("updates", "Check for Updates", "secondary"), ("web", "Project page", "secondary")],
            default="ok",
            cancel_key="ok",
            detail=f"Project:  {PROJECT_URL}\nProgram:  {os.path.abspath(__file__)}\n"
            f"Settings:  {config_path()}\nInstalled for this user:  {installed}",
        )
        if choice == "updates":
            self.check_for_updates(manual=True)
        elif choice == "web":
            self.open_project_page()

    def open_project_page(self, url=PROJECT_URL):
        import webbrowser

        webbrowser.open(url)

    def open_config_folder(self):
        folder = config_dir()
        os.makedirs(folder, exist_ok=True)
        if not os.path.exists(config_path()):
            self._save_config()
        opener = "open" if sys.platform == "darwin" else ("explorer" if sys.platform.startswith("win") else "xdg-open")
        try:
            subprocess.Popen([opener, folder])
        except OSError:
            messagebox.showinfo("Configuration Folder", folder, parent=self.root)

    # ==================================================================
    # Updates
    # ==================================================================
    def _toggle_update_check(self):
        self.config["check_updates_on_start"] = bool(self.check_updates_var.get())
        self._save_config()

    def check_for_updates(self, manual=False):
        """Asks GitHub for the latest version in a background thread; offers an update if newer."""
        if getattr(self, "_update_checking", False):
            return
        self._update_checking = True
        self._update_result = None

        def worker():
            try:
                self._update_result = ("ok", fetch_latest_version())
            except Exception as exc:  # network / parse problems are reported, never fatal
                self._update_result = ("error", exc)

        threading.Thread(target=worker, daemon=True).start()
        if manual:
            self.set_status("Checking for updates…")
        self.root.after(200, lambda: self._poll_update_check(manual))

    def _poll_update_check(self, manual):
        if self._update_result is None:
            self.root.after(200, lambda: self._poll_update_check(manual))
            return
        kind, value = self._update_result
        self._update_checking = False
        self._update_result = None
        self.config["last_update_check"] = time.time()
        self._save_config()

        if kind == "error":
            if manual:
                self.set_status("Update check failed.", "warn")
                messagebox.showwarning(
                    "Update Check Failed", f"Could not get the latest version from GitHub:\n\n{value}", parent=self.root
                )
            return
        latest = value["version"]
        if is_newer_version(latest):
            if manual or self.config.get("skipped_version") != latest:
                self.set_status(f"Version {latest} is available (you have {__version__}).", "warn")
                self._offer_update(value)
        elif manual:
            self.set_status(f"{APP_NAME} {__version__} is up to date.", "ok")
            messagebox.showinfo(
                "No Update Available", f"You have the latest version of {APP_NAME} ({__version__}).", parent=self.root
            )

    def _update_targets(self):
        """Program files an update should replace: this copy (unless it is a git checkout) and the installed copy."""
        targets = []
        here = os.path.abspath(__file__)
        if not is_git_checkout(here):
            targets.append(here)
        installed = os.path.join(installed_app_dir(), UPDATE_FILE)
        if self.config.get("install", {}).get("installed") and os.path.exists(installed):
            if os.path.abspath(installed) not in targets:
                targets.append(os.path.abspath(installed))
        return targets

    def _offer_update(self, info):
        targets = self._update_targets()
        notes = info.get("notes") or ""
        if len(notes) > 700:
            notes = notes[:700].rstrip() + " …"
        detail = (notes + "\n\n") if notes else ""
        if targets:
            detail += "The update replaces:\n" + "\n".join(targets)
        if is_git_checkout(__file__):
            detail += ("\n\n" if targets else "") + "This copy is a git working tree — update it with  git pull."
        buttons = [("install", "Install update", "primary")] if targets else []
        buttons += [("web", "View on GitHub", "secondary"), ("skip", "Skip this version", "secondary"), ("later", "Later", "secondary")]
        choice = ChoiceDialog.ask(
            self.root,
            "Update available",
            f"{APP_NAME} {info['version']} is available — you have {__version__}.",
            buttons,
            default=buttons[0][0],
            cancel_key="later",
            detail=detail,
            width=520,
        )
        if choice == "install":
            self._install_update(info, targets)
        elif choice == "web":
            self.open_project_page(info.get("page") or PROJECT_URL)
        elif choice == "skip":
            self.config["skipped_version"] = info["version"]
            self._save_config()

    def _install_update(self, info, targets):
        self.set_status(f"Downloading {APP_NAME} {info['version']}…")
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            text, version = download_update(info["ref"], info["version"])
            backups = [apply_update(text, target) for target in targets]
        except Exception as exc:
            self.set_status("Update failed.", "error")
            messagebox.showerror("Update Failed", f"The update could not be installed:\n\n{exc}", parent=self.root)
            return
        finally:
            self.root.config(cursor="")
        if self.config.get("install", {}).get("installed"):
            self.config["install"]["version"] = version
        self.config["skipped_version"] = None
        self._save_config()
        self.set_status(f"Updated to {version}. Restart to use the new version.", "ok")
        restart = messagebox.askyesno(
            "Update Installed",
            f"{APP_NAME} {version} was installed.\n\nPrevious version saved as:\n{backups[0]}\n\nRestart now?",
            parent=self.root,
        )
        if restart and self.confirm_discard():
            self._save_config()
            self.root.destroy()
            args = [sys.executable, targets[0]] + ([self.model_path] if self.model_path else [])
            os.execv(sys.executable, args)

    # ==================================================================
    # First start: per-user installation
    # ==================================================================
    def _run_startup_tasks(self):
        inst = self.config.get("install", {})
        if (
            inst.get("prompt", True)
            and not inst.get("installed")
            and (sys.platform == "darwin" or sys.platform.startswith("linux"))
            and not is_running_installed_copy()
        ):
            self.offer_install(first_start=True)
        if self.config.get("check_updates_on_start", True):
            last = self.config.get("last_update_check") or 0
            if time.time() - float(last) > UPDATE_CHECK_INTERVAL_S:
                self.check_for_updates(manual=False)

    def offer_install(self, first_start=False):
        if sys.platform == "darwin":
            where = f"~/Applications/{APP_NAME}.app (Launchpad, Spotlight, Dock)"
        elif sys.platform.startswith("linux"):
            where = "an entry in your applications menu and the '3dgg' command in ~/.local/bin"
        else:
            messagebox.showinfo("Install", "Installation is available on macOS and Linux only.", parent=self.root)
            return
        reinstall = bool(self.config.get("install", {}).get("installed"))
        buttons = [("install", "Reinstall" if reinstall else "Install", "primary"), ("later", "Not now", "secondary")]
        if first_start:
            buttons.append(("never", "Don't ask again", "secondary"))
        choice = ChoiceDialog.ask(
            self.root,
            f"Welcome to {APP_NAME} {__version__}" if first_start else f"Install {APP_NAME}",
            f"{'Reinstall' if reinstall else 'Install'} {APP_NAME} for your user account?",
            buttons,
            default="install",
            cancel_key="later",
            detail=f"Copies the program to ~/{CONFIG_DIRNAME}/app and adds {where}, with its icon and the "
            f"{MODEL_EXT} file type. No administrator rights are needed.\n\nSettings: {config_path()}",
            width=500,
        )
        if choice == "install":
            self.install_application()
        elif choice == "never":
            self.config["install"]["prompt"] = False
            self._save_config()

    def install_application(self):
        try:
            info = install_app()
        except (OSError, GeometryError) as exc:
            self.set_status("Installation failed.", "error")
            messagebox.showerror("Installation Failed", str(exc), parent=self.root)
            return
        self.config["install"].update(
            installed=True,
            prompt=False,
            version=__version__,
            date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            app_dir=info["app_dir"],
            paths=info["paths"],
        )
        self._save_config()
        self._refresh_install_menu()
        self.set_status(f"Installed {APP_NAME} {__version__} — {info['launcher']}", "ok")
        how = (
            f"Start it from Launchpad / Spotlight or\n{info['launcher']}"
            if sys.platform == "darwin"
            else f"Start it from the applications menu or with\n{info['launcher']}"
        )
        if sys.platform == "darwin" and not is_running_installed_copy():
            if messagebox.askyesno(
                "Installed",
                f"{APP_NAME} {__version__} is installed.\n\n{how}\n\n"
                "This window still runs from the Python interpreter. "
                "Switch to the installed app now?",
                parent=self.root,
            ):
                self.switch_to_installed_app(info["launcher"])
            return
        messagebox.showinfo("Installed", f"{APP_NAME} {__version__} is installed.\n\n{how}", parent=self.root)

    def switch_to_installed_app(self, bundle):
        """Opens the installed macOS app (with the current model) and closes this window."""
        if not self.confirm_discard():
            return
        cmd = ["open", "-n", "-a", bundle]
        if self.model_path and os.path.exists(self.model_path):
            cmd.append(self.model_path)
        try:
            subprocess.Popen(cmd)
        except OSError as exc:
            messagebox.showerror("Could Not Start", str(exc), parent=self.root)
            return
        self._save_config()
        self.root.destroy()

    def uninstall_application(self):
        inst = self.config.get("install", {})
        if not inst.get("installed"):
            messagebox.showinfo("Uninstall", f"{APP_NAME} is not installed for this user.", parent=self.root)
            return
        if not messagebox.askyesno(
            "Uninstall",
            f"Remove the installed {APP_NAME}?\n\n" + "\n".join(inst.get("paths", []))
            + f"\n\nYour models and {config_path()} are kept.",
            parent=self.root,
        ):
            return
        try:
            uninstall_app(inst.get("paths", []))
        except OSError as exc:
            messagebox.showerror("Uninstall Failed", str(exc), parent=self.root)
            return
        inst.update(installed=False, version=None, date=None, app_dir=None, paths=[])
        self._save_config()
        self._refresh_install_menu()
        self.set_status(f"{APP_NAME} was uninstalled for this user.", "ok")

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

    # ==================================================================
    # Undo / redo (snapshots of the points of the three 2D views)
    # ==================================================================
    HISTORY_LIMIT = 300

    def _view_state(self):
        return tuple(
            (tuple(tuple(p) for p in view.points), bool(view.closed)) for view in self._views().values()
        )

    def _schedule_commit(self):
        """Records the change once the current action is complete (several views may change in one action)."""
        if not self._commit_pending:
            self._commit_pending = True
            self.root.after_idle(self._commit_history)

    def _commit_history(self):
        """Pushes the previous state onto the undo stack if the views changed since the last commit."""
        self._commit_pending = False
        if any(view.dragging_idx is not None for view in self._views().values()):
            return  # committed when the mouse button is released
        state = self._view_state()
        if state != self._history_state:
            self._undo_stack.append(self._history_state)
            del self._undo_stack[: -self.HISTORY_LIMIT]
            self._redo_stack.clear()
            self._history_state = state

    def reset_history(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._history_state = self._view_state()

    def _restore_state(self, state):
        self._history_state = state
        for view, (points, closed) in zip(self._views().values(), state):
            if (tuple(tuple(p) for p in view.points), bool(view.closed)) != (points, closed):
                view.set_points([list(p) for p in points], closed=closed)

    def undo(self):
        self._commit_history()  # include a change that is still waiting to be recorded
        if not self._undo_stack:
            self.set_status("Nothing to undo.")
            return
        self._redo_stack.append(self._history_state)
        self._restore_state(self._undo_stack.pop())
        self.set_status(f"Undone. ({len(self._undo_stack)} more step(s) to undo)")

    def redo(self):
        self._commit_history()
        if not self._redo_stack:
            self.set_status("Nothing to redo.")
            return
        self._undo_stack.append(self._history_state)
        self._restore_state(self._redo_stack.pop())
        self.set_status(f"Redone. ({len(self._redo_stack)} more step(s) to redo)")

    def _on_geometry_changed(self):
        """Called by the 2D views whenever points are added, moved, deleted or replaced."""
        self._set_dirty(True)
        self._schedule_commit()
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

    def mirror_view(self, view, axis):
        """Makes the outline of one 2D view symmetric about u = 0 or v = 0 (undoable)."""
        label = view.u_label if axis == "u" else view.v_label
        try:
            sign = view.mirror(axis)
        except GeometryError as exc:
            self.set_status(f"Mirror {view.title}: {exc}", "warn")
            messagebox.showinfo(f"Mirror {view.title}", str(exc), parent=self.root)
            return
        half = f"{label} {'>' if sign > 0 else '<'} 0"
        self.set_status(f"Mirrored the {half} half of the {view.title} onto the other side of {label} = 0.")
        self._commit_history()

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
    def generate_geometry(self, quiet=False):
        """
        Reconstructs the 3D solid from the current 2D views and renders it in the
        3D view. Returns the Geometry3D, or None if the views are incomplete
        (quiet=True: report only in the status bar, no dialog).
        """
        try:
            geom = reconstruct_geometry(self.view_xy.points, self.view_xz.points, self.view_yz.points)
        except GeometryError as exc:
            self.set_status(f"Cannot generate geometry: {exc}", "error")
            if not quiet:
                messagebox.showwarning("Cannot Generate Geometry", str(exc))
            return None

        # Mark all 3 views as closed polygons so subsequent clicks edit nicely
        self.view_xy.close_polygon()
        self.view_xz.close_polygon()
        self.view_yz.close_polygon()

        self.geometry = geom
        self.geometry_outdated = False

        xmin, xmax, ymin, ymax, zmin, zmax = geom.bbox
        section = "rectangle" if geom.section_is_rect else f"{len(geom.section)}-point outline"
        summary = (
            f"Material  InP     Sections  {len(geom.segments)}\n"
            f"Cross-section  {section} (Front view)\n"
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
        # Default: next to the model (named like it), else the last export folder
        last_export = self.config.get("last_export_dir")
        if self.model_path:
            default_dir = os.path.dirname(self.model_path)
            default_name = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            default_dir = last_export if last_export and os.path.isdir(last_export) else self._model_dir()
            default_name = "generate_structure"
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Save {fmt_label}" + (" — base name for all files" if len(exts) > 1 else ""),
            initialdir=default_dir,
            initialfile=default_name + exts[0],
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
        self.config["last_export_dir"] = folder
        self._save_config()
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


def set_macos_app_name(name=APP_SHORT_NAME):
    """
    Makes the macOS menu bar show `name` instead of "Python" when the program is
    started with the python interpreter. Must run before tk.Tk() is created.
    Uses the Objective-C runtime through ctypes (no pyobjc needed); silently does
    nothing if anything is unavailable.
    """
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        import ctypes.util

        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send0 = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        send_str = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
            ("objc_msgSend", objc)
        )
        send2 = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        sel = objc.sel_registerName

        def nsstring(text):
            return send_str(objc.objc_getClass(b"NSString"), sel(b"stringWithUTF8String:"), text.encode("utf-8"))

        bundle = send0(objc.objc_getClass(b"NSBundle"), sel(b"mainBundle"))
        changed = False
        for getter in (b"localizedInfoDictionary", b"infoDictionary"):
            info = send0(bundle, sel(getter))
            if not info:
                continue
            mutable_cls = objc.objc_getClass(b"NSMutableDictionary")
            is_kind = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
                ("objc_msgSend", objc)
            )
            if not is_kind(info, sel(b"isKindOfClass:"), mutable_cls):
                continue
            send2(info, sel(b"setObject:forKey:"), nsstring(name), nsstring("CFBundleName"))
            changed = True
        return changed
    except Exception:  # never block the start because of a cosmetic detail
        return False


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = [a for a in argv if not a.startswith("-psn_")]  # old macOS Finder launch argument

    parser = argparse.ArgumentParser(prog="geometry_gui.py", description=__doc__.split("\n")[1].strip())
    parser.add_argument("model", nargs="?", help=f"{MODEL_EXT} model file to open")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    parser.add_argument(
        "--export-preset",
        nargs="+",
        metavar="FILE",
        help="export the Slide 1 preset to FILE(s) (.lsf, .gds or .py) without opening the GUI",
    )
    parser.add_argument("--layer", default="1/0", help="GDS layer/datatype (default: 1/0)")
    parser.add_argument("--cell", default=DEFAULT_CELL_NAME, help=f"GDS cell name (default: {DEFAULT_CELL_NAME})")
    parser.add_argument("--install", action="store_true", help="install for the current user (macOS / Linux) and exit")
    parser.add_argument("--uninstall", action="store_true", help="remove the per-user installation and exit")
    parser.add_argument("--check-updates", action="store_true", help="print whether a newer version exists and exit")
    args = parser.parse_args(argv)

    if args.export_preset:
        try:
            export_preset_cli(args.export_preset, args.cell, args.layer)
        except GeometryError as exc:
            parser.error(str(exc))
        return

    cfg, existed = load_config()
    if args.install or args.uninstall:
        try:
            if args.install:
                info = install_app()
                cfg["install"].update(
                    installed=True,
                    prompt=False,
                    version=__version__,
                    date=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                    app_dir=info["app_dir"],
                    paths=info["paths"],
                )
                print(f"Installed {APP_NAME} {__version__}:")
                print("\n".join("  " + p for p in info["paths"]))
            else:
                removed = uninstall_app(cfg["install"].get("paths", []))
                cfg["install"].update(installed=False, version=None, date=None, app_dir=None, paths=[])
                print("Removed:\n" + "\n".join("  " + p for p in removed) if removed else "Nothing to remove.")
        except GeometryError as exc:
            parser.error(str(exc))
        save_config(cfg)
        return
    if args.check_updates:
        try:
            info = fetch_latest_version()
        except Exception as exc:  # network or GitHub problem
            parser.exit(1, f"Update check failed: {exc}\n")
        newer = is_newer_version(info["version"])
        print(f"Installed: {__version__}   Latest on GitHub: {info['version']}" + ("   -> update available" if newer else ""))
        return

    if not existed:
        save_config(cfg)  # first start: create ~/.3dgg/config.json

    set_macos_app_name()
    root = tk.Tk(className="3dgg")
    app = LumericalGeometryApp(root, config=cfg, startup_tasks=True)
    if args.model:
        root.after(100, lambda: app.open_model_path(args.model, confirm=False))
    root.mainloop()


if __name__ == "__main__":
    main()
