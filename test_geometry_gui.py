"""Tests for geometry_gui.py — run with:  python -m pytest -q test_geometry_gui.py"""

import ast
import importlib.util
import os
import subprocess
import sys

import pytest

import geometry_gui as gg

HAS_KLAYOUT = importlib.util.find_spec("klayout") is not None
HAS_GDSFACTORY = importlib.util.find_spec("gdsfactory") is not None

RECT_XZ = [[0, -0.1], [20, -0.1], [20, 0.1], [0, 0.1]]
RECT_YZ = [[0, -0.1], [1.2, -0.1], [1.2, 0.1], [0, 0.1]]


@pytest.fixture
def preset():
    return gg.reconstruct_geometry(gg.DEFAULT_OBJECT["xy"], gg.DEFAULT_OBJECT["xz"], gg.DEFAULT_OBJECT["yz"])


def read_gds_polygons(path):
    """Returns (dbu, cell names, {(layer, datatype): [sorted point lists]}) using KLayout."""
    import klayout.db as kdb

    layout = kdb.Layout()
    layout.read(str(path))
    shapes = {}
    for li in layout.layer_indexes():
        info = layout.get_info(li)
        for cell in layout.each_cell():
            for shape in cell.shapes(li).each():
                pts = sorted((p.x, p.y) for p in shape.polygon.each_point_hull())
                shapes.setdefault((info.layer, info.datatype), []).append(pts)
    return layout.dbu, [c.name for c in layout.each_cell()], shapes


# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def test_preset_reconstruction(preset):
    xs = [(s["x0"], s["x1"]) for s in preset.segments]
    assert xs == [(-300, -110), (-110, -100), (-100, 100), (100, 110), (110, 300)]
    widths = [w for s in preset.segments for w in (s["y0_top"] - s["y0_bot"], s["y1_top"] - s["y1_bot"])]
    assert widths == pytest.approx([0.1, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.1], abs=1e-9)
    assert preset.bbox == pytest.approx((-300, 300, -0.75, 0.75, -0.125, 0.125))
    assert preset.z_is_uniform


def test_close_x_vertices_are_not_merged():
    # Old code merged x=10 and x=10.2 into a single station at 10.1
    xy = [[0, 0], [10, 0], [10.2, 1], [20, 1], [20, 1.2], [0, 1.2]]
    geom = gg.reconstruct_geometry(xy, RECT_XZ, RECT_YZ)
    assert [(s["x0"], s["x1"]) for s in geom.segments] == [(0, 10), (10, 10.2), (10.2, 20)]



def test_steep_step_widths_are_exact():
    # 1 nm long step from y=0 to y=1: the old sampling offset gave ~10 nm width errors here
    xy = [[0, 0], [10, 0], [10.001, 1], [20, 1], [20, 2], [0, 2]]
    geom = gg.reconstruct_geometry(xy, RECT_XZ, RECT_YZ)
    step = geom.segments[1]
    assert (step["x0"], step["x1"]) == (10, 10.001)
    assert (step["y0_bot"], step["y1_bot"], step["y0_top"], step["y1_top"]) == (0, 1, 2, 2)

@pytest.mark.parametrize(
    "xy",
    [
        [[5, 0], [5, 1], [5, 2]],  # no x extent (used to crash with ValueError: min() arg is empty)
        [[0, 0], [1, 1], [2, 2]],  # zero area
        [[0, 0], [1, 1]],  # too few points
    ],
)
def test_degenerate_input_raises_geometry_error(xy):
    with pytest.raises(gg.GeometryError):
        gg.reconstruct_geometry(xy, RECT_XZ, RECT_YZ)


def test_varying_thickness_is_detected():
    xz = [[0, -0.1], [20, -0.1], [20, 0.3], [0, 0.1]]
    geom = gg.reconstruct_geometry([[0, 0], [20, 0], [20, 1], [0, 1]], xz, RECT_YZ)
    assert not geom.z_is_uniform


def test_front_view_shape_changes_the_solid():
    xy = [[0, 0], [20, 0], [20, 1], [0, 1]]
    hexagon = [[0, 0], [-0.05, 0.02], [-0.05, -0.02], [0, -0.04], [0.05, -0.02], [0.05, 0.02]]
    geom = gg.reconstruct_geometry(xy, RECT_XZ, hexagon)
    assert not geom.section_is_rect
    assert len(geom.section) == 6
    moved = [list(p) for p in hexagon]
    moved[0] = [0.0, -0.03]
    assert gg.reconstruct_geometry(xy, RECT_XZ, moved).faces != geom.faces
    # the cross-section is stretched to the local width (top view) and thickness (side view)
    assert geom.bbox == pytest.approx((0, 20, 0, 1, -0.1, 0.1))


def test_rectangular_front_view_gives_box_sections():
    # extra collinear point on an edge is still a rectangle
    yz = [[0, -0.1], [0.6, -0.1], [1.2, -0.1], [1.2, 0.1], [0, 0.1]]
    geom = gg.reconstruct_geometry([[0, 0], [20, 0], [20, 1], [0, 1]], RECT_XZ, yz)
    assert geom.section_is_rect
    assert "addplanarsolid" not in gg.build_lsf(geom)


def test_self_crossing_front_view_raises():
    bow_tie = [[0, 0], [1, 1], [1, 0], [0, 1]]
    with pytest.raises(gg.GeometryError, match="crosses itself"):
        gg.reconstruct_geometry([[0, 0], [20, 0], [20, 1], [0, 1]], RECT_XZ, bow_tie)


def test_side_view_vertices_become_stations():
    xz = [[0, -0.1], [5, -0.2], [15, -0.2], [20, -0.1], [20, 0.1], [15, 0.2], [5, 0.2], [0, 0.1]]
    geom = gg.reconstruct_geometry([[0, 0], [20, 0], [20, 1], [0, 1]], xz, RECT_YZ)
    assert [(s["x0"], s["x1"]) for s in geom.segments] == [(0, 5), (5, 15), (15, 20)]
    assert (geom.segments[1]["z0_bot"], geom.segments[1]["z0_top"]) == (-0.2, 0.2)


def test_lsf_planar_solids_are_closed_and_outward():
    xy = [[0, 0], [10, 0.2], [20, 0.2], [20, 0.8], [10, 0.8], [0, 1]]
    hexagon = [[0, 0], [-0.05, 0.02], [-0.05, -0.02], [0, -0.04], [0.05, -0.02], [0.05, 0.02]]
    geom = gg.reconstruct_geometry(xy, RECT_XZ, hexagon)
    lsf = gg.build_lsf(geom)
    assert lsf.count("addplanarsolid(") == len(geom.segments)
    assert lsf.count("{") == lsf.count("}") and lsf.count("[") == lsf.count("]")
    facets = [[int(i) for i in m.split(",")] for m in __import__("re").findall(r"F_sec_1\{\d+\}\{1\} = \[([^]]+)\]", lsf)]
    edges = {}
    for fc in facets:
        for a, b in zip(fc, fc[1:] + fc[:1]):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    assert all(c == 1 and edges.get((b, a)) == 1 for (a, b), c in edges.items())  # closed, consistently oriented


# ----------------------------------------------------------------------
# Lumerical
# ----------------------------------------------------------------------
def test_fmt_num_keeps_precision():
    assert gg.fmt_num(1234.5678) == "1234.5678"
    assert gg.fmt_num(0.1 + 0.2) == "0.3"
    assert gg.fmt_num(-0.0) == "0"
    assert gg.fmt_num(250.0) == "250"


def test_lsf_preset_content(preset):
    lsf = gg.build_lsf(preset)
    assert "t_z      = 250 * nm;" in lsf
    assert lsf.count("addpoly;") == 5  # 4 tapers + full polygon
    assert lsf.count("addrect;") == 1  # central straight section
    assert 'set("x span", 200 * um);' in lsf
    assert 'set("y span", 1.5 * um);' in lsf
    assert "        -300 * um, -0.05 * um;" in lsf
    assert lsf.count("{") == lsf.count("}")
    assert lsf.count("[") == lsf.count("]")


def test_lsf_full_precision():
    xy = [[0, 0], [1234.5678, 0], [1234.5678, 0.5], [0, 0.5]]
    xz = [[0, -0.1], [1234.5678, -0.1], [1234.5678, 0.1], [0, 0.1]]
    lsf = gg.build_lsf(gg.reconstruct_geometry(xy, xz, RECT_YZ))
    assert "1234.5678 * um" in lsf
    assert "1234.57" not in lsf


# ----------------------------------------------------------------------
# GDS
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [("1/0", (1, 0)), ("34, 2", (34, 2)), ("7", (7, 0)), (" 3:1 ", (3, 1))])
def test_parse_layer_spec(text, expected):
    assert gg.parse_layer_spec(text) == expected


@pytest.mark.parametrize("text", ["", "a/b", "1/2/3", "-1/0", "70000/0"])
def test_parse_layer_spec_invalid(text):
    with pytest.raises(gg.GeometryError):
        gg.parse_layer_spec(text)


def test_sanitize_cell_name():
    assert gg.sanitize_cell_name("my wg-1") == "my_wg_1"
    assert gg.sanitize_cell_name("1abc") == "cell_1abc"
    assert gg.sanitize_cell_name("class") == "class_"
    assert gg.sanitize_cell_name("  ") == gg.DEFAULT_CELL_NAME


def test_gds_real8_known_values():
    assert gg._gds_real8(1e-3).hex() == "3e4189374bc6a7f0"
    assert gg._gds_real8(1e-9).hex() == "3944b82fa09b5a54"


@pytest.mark.skipif(not HAS_KLAYOUT, reason="klayout not installed")
@pytest.mark.parametrize("backend", ["builtin", "klayout"])
def test_gds_roundtrip(preset, tmp_path, backend):
    path = tmp_path / f"{backend}.gds"
    used = gg.write_gds(preset, path, cell_name="my wg", layer=(34, 2), backend=backend)
    assert used == backend
    dbu, cells, shapes = read_gds_polygons(path)
    assert dbu == pytest.approx(0.001)
    assert cells == ["my_wg"]
    expected = sorted((round(x * 1000), round(y * 1000)) for x, y in gg.DEFAULT_OBJECT["xy"])
    assert shapes == {(34, 2): [expected]}


def test_builtin_gds_without_klayout(preset, tmp_path):
    path = tmp_path / "b.gds"
    gg.write_gds(preset, path, backend="builtin")
    data = path.read_bytes()
    assert data[:6] == bytes.fromhex("000600020258")  # HEADER, version 600
    assert data[-4:] == bytes.fromhex("00040400")  # ENDLIB


def test_gds_off_grid_points_are_snapped_and_deduplicated():
    pts = gg.polygon_to_dbu([[0, 0], [0.0004, 0], [1.0006, 0], [1, 1], [0, 1], [0, 0]])
    assert pts == [(0, 0), (1001, 0), (1000, 1000), (0, 1000)]


# ----------------------------------------------------------------------
# gdsfactory script
# ----------------------------------------------------------------------
def test_gdsfactory_script_is_valid_python(preset):
    src = gg.build_gdsfactory_script(preset, cell_name="InP wg", layer=(5, 1))
    tree = ast.parse(src)
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"InP_wg", "InP_wg_sections", "get_layer_stack"} <= funcs
    assert '"layer": (5, 1),' in src
    assert '"thickness_um": 0.25,' in src
    assert '"zmin_um": -0.125,' in src


def test_export_rejects_shadowing_script_name(preset, tmp_path):
    with pytest.raises(gg.GeometryError):
        gg.export_geometry(preset, str(tmp_path / "gdsfactory.py"))


@pytest.mark.skipif(not (HAS_GDSFACTORY and HAS_KLAYOUT), reason="gdsfactory not installed")
def test_gdsfactory_script_generates_matching_gds(preset, tmp_path):
    script = tmp_path / "structure.py"
    gg.export_geometry(preset, str(script), cell_name="InP_waveguide", layer=(1, 0))
    direct = tmp_path / "direct.gds"
    gg.export_geometry(preset, str(direct), cell_name="InP_waveguide", layer=(1, 0))

    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    generated = script.with_suffix(".gds")
    assert generated.exists()

    _, cells, gf_shapes = read_gds_polygons(generated)
    _, _, direct_shapes = read_gds_polygons(direct)
    assert cells == ["InP_waveguide"]  # exact cell name, no parameter suffix
    assert gf_shapes == direct_shapes

    # The section-split cell covers the same x/y extent with one polygon per section
    sec = subprocess.run(
        [sys.executable, "-c", "import structure; c = structure.InP_waveguide_sections(); "
         "print(c.name, len(list(c.shapes(c.kcl.layer(1, 0)).each())))"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert sec.returncode == 0, sec.stderr
    assert sec.stdout.split()[-2:] == ["InP_waveguide_sections", "5"]

    # LayerStack builds with the installed gdsfactory version
    check = subprocess.run(
        [sys.executable, "-c", "import structure; s = structure.get_layer_stack(); print(s.layers['core'].thickness)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert check.returncode == 0, check.stderr
    assert check.stdout.strip().endswith("0.25")


def test_cli_export(tmp_path):
    files = [str(tmp_path / n) for n in ("a.lsf", "a.gds", "a.py")]
    run = subprocess.run(
        [sys.executable, gg.__file__, "--export-preset", *files, "--layer", "2/0"],
        capture_output=True, text=True, timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert all(os.path.getsize(f) > 0 for f in files)


# ----------------------------------------------------------------------
# Model files (.3dgg)
# ----------------------------------------------------------------------
def preset_views():
    return {key: (gg.DEFAULT_OBJECT[key], True) for key in gg.VIEW_KEYS}


def test_model_round_trip(tmp_path):
    path = tmp_path / "taper.3dgg"
    export = {"format": "GDSII layout (.gds)", "gds_layer": "5/1", "cell_name": "taper"}
    gg.write_model(str(path), gg.model_to_dict(preset_views(), export, snap=False))
    data = gg.read_model(str(path))
    assert data["format"] == "3dgg" and data["generator"] == f"{gg.APP_NAME} {gg.__version__}"
    assert [layer["gds_layer"] for layer in data["layers"]] == ["5/1"]  # single views dict -> one layer
    for key in gg.VIEW_KEYS:
        view = data["layers"][0]["views"][key]
        assert view["points"] == [[float(u), float(v)] for u, v in gg.DEFAULT_OBJECT[key]]
        assert view["closed"] is True
    assert data["export"] == {"format": "GDSII layout (.gds)", "cell_name": "taper"}
    assert data["editor"] == {"snap": False}


def test_model_reads_version_1_files(tmp_path):
    # file written by 0.3.x: one set of views, GDS layer in the export settings
    path = tmp_path / "old.3dgg"
    old = {
        "format": "3dgg", "format_version": 1,
        "views": {k: {"points": gg.DEFAULT_OBJECT[k], "closed": True} for k in gg.VIEW_KEYS},
        "export": {"format": "GDSII layout (.gds)", "gds_layer": "7/2", "cell_name": "old"},
    }
    path.write_text(__import__("json").dumps(old))
    data = gg.read_model(str(path))
    assert [layer["gds_layer"] for layer in data["layers"]] == ["7/2"]
    assert data["layers"][0]["views"]["xy"]["points"][0] == [-300.0, -0.05]
    assert data["active_layer"] == 0


def two_layers():
    rib = {"xy": [[-50, -1], [50, -1], [50, 1], [-50, 1]], "xz": [[-50, 0.125], [50, 0.125], [50, 0.4], [-50, 0.4]],
           "yz": [[-1, 0.125], [1, 0.125], [1, 0.4], [-1, 0.4]]}
    return [((1, 0), dict(gg.DEFAULT_OBJECT)), ((2, 0), rib)]


def test_model_round_trip_several_layers(tmp_path):
    layers = [{"gds_layer": gg.format_layer(spec), "views": {k: (pts, True) for k, pts in views.items()}}
              for spec, views in two_layers()]
    path = tmp_path / "stack.3dgg"
    gg.write_model(str(path), gg.model_to_dict(layers, {"cell_name": "stack"}, active_layer=1))
    data = gg.read_model(str(path))
    assert data["format_version"] == 2 and data["active_layer"] == 1
    assert [layer["gds_layer"] for layer in data["layers"]] == ["1/0", "2/0"]
    assert data["layers"][1]["views"]["xz"]["points"][2] == [50.0, 0.4]


def test_read_model_rejects_duplicate_layers(tmp_path):
    views = {k: {"points": [], "closed": False} for k in gg.VIEW_KEYS}
    doc = {"format": "3dgg", "format_version": 2, "layers": [{"gds_layer": "1/0", "views": views}] * 2}
    path = tmp_path / "dup.3dgg"
    path.write_text(__import__("json").dumps(doc))
    with pytest.raises(gg.GeometryError, match="twice"):
        gg.read_model(str(path))


def test_reconstruct_layers_skips_empty_and_names_incomplete_layer():
    layers = two_layers() + [((3, 0), {"xy": [], "xz": [], "yz": []})]
    geoms = gg.reconstruct_layers(layers)
    assert [lay for lay, _ in geoms] == [(1, 0), (2, 0)]
    assert geoms[1][1].z_span == pytest.approx((0.125, 0.4))
    broken = two_layers() + [((3, 0), {"xy": [[0, 0], [1, 0], [1, 1]], "xz": [], "yz": []})]
    with pytest.raises(gg.GeometryError, match="Layer 3/0"):
        gg.reconstruct_layers(broken)
    with pytest.raises(gg.GeometryError):
        gg.reconstruct_layers([((1, 0), {"xy": [], "xz": [], "yz": []})])


def test_next_free_layer():
    assert gg.next_free_layer([(1, 0)]) == (2, 0)
    assert gg.next_free_layer([(1, 0), (5, 2), (3, 0)]) == (6, 0)


def test_lsf_several_layers():
    lsf = gg.build_lsf(gg.reconstruct_layers(two_layers()))
    assert 'set("name", "Layer_1_0");' in lsf and 'set("name", "Layer_2_0");' in lsf
    assert 'addtogroup("Layer_2_0");' in lsf
    assert "t_z_L2_0 = 275 * nm;" in lsf
    assert "V_L1_0_all" in lsf and "V_L2_0_all" in lsf
    assert lsf.count("{") == lsf.count("}") and lsf.count("[") == lsf.count("]")


@pytest.mark.skipif(not HAS_KLAYOUT, reason="klayout not installed")
@pytest.mark.parametrize("backend", ["builtin", "klayout"])
def test_gds_several_layers(tmp_path, backend):
    path = tmp_path / "stack.gds"
    gg.write_gds(gg.reconstruct_layers(two_layers()), path, backend=backend)
    _, _, shapes = read_gds_polygons(path)
    assert set(shapes) == {(1, 0), (2, 0)}
    assert shapes[(2, 0)] == [sorted([(-50000, -1000), (50000, -1000), (50000, 1000), (-50000, 1000)])]


@pytest.mark.skipif(not (HAS_GDSFACTORY and HAS_KLAYOUT), reason="gdsfactory not installed")
def test_gdsfactory_script_several_layers(tmp_path):
    script = tmp_path / "stack.py"
    gg.export_geometry(gg.reconstruct_layers(two_layers()), str(script), cell_name="stack")
    check = subprocess.run(
        [sys.executable, "-c", "import stack; c = stack.stack(); c.write_gds('s.gds'); "
         "s = stack.get_layer_stack(); print(sorted(s.layers), s.layers['layer_2_0'].zmin, "
         "len(list(c.shapes(c.kcl.layer(2, 0)).each())))"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert check.returncode == 0, check.stderr
    assert check.stdout.split() == ["['layer_1_0',", "'layer_2_0']", "0.125", "1"]


@pytest.mark.parametrize(
    "content, message",
    [
        ("not json", "not a valid"),
        ('{"format": "other"}', "not a 3D Geometry Generation model"),
        ('{"format": "3dgg", "format_version": 99, "views": {}}', "newer version"),
        ('{"format": "3dgg", "views": {"xy": {"points": [[1, "a"]]}}}', "non-numeric"),
        ('{"format": "3dgg", "views": {"xy": {"points": [[1, 2, 3]]}}}', "invalid point"),
    ],
)
def test_read_model_rejects_bad_files(tmp_path, content, message):
    path = tmp_path / "bad.3dgg"
    path.write_text(content)
    with pytest.raises(gg.GeometryError, match=message):
        gg.read_model(str(path))


# ----------------------------------------------------------------------
# Version / update check (network mocked)
# ----------------------------------------------------------------------
def test_version_helpers():
    assert gg.parse_version("v0.3.1") == (0, 3, 1)
    assert gg.parse_version("1.2") == (1, 2, 0)
    assert gg.is_newer_version("0.3.1", "0.3.0")
    assert gg.is_newer_version("0.10.0", "0.9.9")
    assert not gg.is_newer_version("v0.3.0", "0.3.0")
    with open(gg.__file__, encoding="utf-8") as fh:
        assert gg.read_version_from_source(fh.read()) == gg.__version__


def test_fetch_latest_version_falls_back_to_main_branch(monkeypatch):
    import urllib.error

    def fake_get(url, timeout=10.0):
        if "releases/latest" in url:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        assert url.endswith("/main/geometry_gui.py")
        return b'"""doc"""\n__version__ = "0.4.2"\n'

    monkeypatch.setattr(gg, "_http_get", fake_get)
    info = gg.fetch_latest_version()
    assert info["version"] == "0.4.2" and info["ref"] == "main"


def test_fetch_latest_version_prefers_release(monkeypatch):
    payload = b'{"tag_name": "v0.5.0", "html_url": "https://example/rel", "body": "notes"}'
    monkeypatch.setattr(gg, "_http_get", lambda url, timeout=10.0: payload)
    info = gg.fetch_latest_version()
    assert (info["version"], info["ref"], info["notes"]) == ("0.5.0", "v0.5.0", "notes")


def test_download_and_apply_update(monkeypatch, tmp_path):
    new_source = '__version__ = "9.9.9"\nprint("hello")\n'
    monkeypatch.setattr(gg, "_http_get", lambda url, timeout=30.0: new_source.encode())
    text, version = gg.download_update("v9.9.9", "9.9.9")
    target = tmp_path / "geometry_gui.py"
    target.write_text("old")
    backup = gg.apply_update(text, str(target), home=str(tmp_path / "home"))
    assert target.read_text() == new_source and open(backup).read() == "old"

    monkeypatch.setattr(gg, "_http_get", lambda url, timeout=30.0: b'__version__ = "1.0"\nif:\n')
    with pytest.raises(SyntaxError):
        gg.download_update("main")


# ----------------------------------------------------------------------
# Config & per-user installation (into a temporary home)
# ----------------------------------------------------------------------
def test_config_round_trip_and_corrupt_file(tmp_path):
    home = str(tmp_path)
    cfg, existed = gg.load_config(home)
    assert not existed and cfg["check_updates_on_start"] is True
    gg.add_recent_file(cfg, str(tmp_path / "a.3dgg"))
    gg.save_config(cfg, home)
    again, existed = gg.load_config(home)
    assert existed and again["recent_files"] == [str(tmp_path / "a.3dgg")]
    assert os.path.exists(os.path.join(home, ".3dgg", "config.json"))

    with open(gg.config_path(home), "w") as fh:
        fh.write("{broken")
    cfg, existed = gg.load_config(home)
    assert not existed and os.path.exists(gg.config_path(home) + ".broken")


def test_install_linux(tmp_path):
    home = str(tmp_path)
    info = gg.install_app(home=home, platform="linux", python="/usr/bin/python3", register=False)
    script = os.path.join(home, ".3dgg", "app", "geometry_gui.py")
    assert os.path.exists(script)
    launcher = os.path.join(home, ".local", "bin", "3dgg")
    assert os.access(launcher, os.X_OK) and script in open(launcher).read()
    desktop = open(os.path.join(home, ".local", "share", "applications", "3dgg.desktop")).read()
    assert "Name=3D Geometry Generation" in desktop and "MimeType=application/x-3dgg;" in desktop
    assert os.path.exists(os.path.join(home, ".local", "share", "mime", "packages", "3dgg.xml"))
    removed = gg.uninstall_app(info["paths"], home=home)
    assert removed and not os.path.exists(launcher) and not os.path.exists(script)


def test_install_macos_bundle(tmp_path):
    import plistlib

    home = str(tmp_path)
    info = gg.install_app(home=home, platform="darwin", python="/usr/bin/python3", register=False)
    bundle = os.path.join(home, "Applications", "3D Geometry Generation.app")
    with open(os.path.join(bundle, "Contents", "Info.plist"), "rb") as fh:
        plist = plistlib.load(fh)
    assert plist["CFBundleShortVersionString"] == gg.__version__
    assert plist["UTExportedTypeDeclarations"][0]["UTTypeTagSpecification"]["public.filename-extension"] == ["3dgg"]
    assert os.access(os.path.join(bundle, "Contents", "MacOS", "3dgg"), os.X_OK)
    gg.install_app(home=home, platform="darwin", python="/usr/bin/python3", register=False)  # reinstall works
    gg.uninstall_app(info["paths"], home=home)
    assert not os.path.exists(bundle)


def test_uninstall_ignores_paths_outside_home(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    assert gg.uninstall_app([str(outside)], home=str(tmp_path / "home")) == []
    assert outside.exists()


# ----------------------------------------------------------------------
# GUI smoke tests (skipped without a display)
# ----------------------------------------------------------------------
@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("GG3D_HOME", str(tmp_path / "home"))  # never touch the real ~/.3dgg
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    warnings = []
    monkeypatch.setattr(gg.messagebox, "showwarning", lambda *a, **k: warnings.append(a))
    monkeypatch.setattr(gg.messagebox, "showinfo", lambda *a, **k: None)
    application = gg.LumericalGeometryApp(root)
    application.warnings = warnings
    yield application
    root.destroy()


def test_gui_export_uses_current_drawing(app):
    # Old code exported the cached geometry, ignoring edits made after "Generate geometry"
    app.load_default_object()
    first = app.build_lsf_content()
    app.view_xy.selected_idx = 0
    app.view_xy.update_selected_point(-300.0, -0.5)
    assert app.geometry_outdated
    second = app.build_lsf_content()
    assert second != first
    assert "-300 * um, -0.5 * um" in second


def test_gui_incomplete_views_warn_instead_of_crash(app):
    app.view_xy.set_points([[5, 0], [5, 1], [5, 2]])
    app.view_xz.set_points(RECT_XZ)
    app.view_yz.set_points(RECT_YZ)
    assert app.generate_geometry() is None
    assert app.warnings


def test_gui_generate_file_all_formats(app, tmp_path, monkeypatch):
    app.load_default_object()
    app.format_var.set("All (.lsf + .gds + .py)")
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    monkeypatch.setattr(gg.filedialog, "asksaveasfilename", lambda **k: str(out_dir / "out.lsf"))
    app.generate_file()
    assert sorted(p.name for p in out_dir.iterdir()) == ["out.gds", "out.lsf", "out.py"]
    assert app.config["last_export_dir"] == str(out_dir)


def test_gui_save_and_open_model(app, tmp_path):
    app.load_default_object()
    app.ent_cell.delete(0, "end")
    app.ent_cell.insert(0, "my_cell")
    assert app.dirty
    path = tmp_path / "model.3dgg"
    assert app.save_model_to(str(path))
    assert not app.dirty and app.model_path == str(path)

    app.dirty = False
    app.clear_all()
    app.dirty = False
    assert app.open_model_path(str(path))
    assert app.view_xy.points == [[float(u), float(v)] for u, v in gg.DEFAULT_OBJECT["xy"]]
    assert app.ent_cell.get() == "my_cell"
    assert app.geometry is not None and not app.dirty
    assert app.config["recent_files"][0] == str(path)


# ----------------------------------------------------------------------
# 2D view helpers: zoom-dependent ticks, mirroring
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "span,pixels,expected",
    [(680, 600, 100), (68, 600, 10), (0.5, 600, 0.1), (2.1, 300, 1), (0.002, 600, 0.0005), (6800, 600, 1000)],
)
def test_nice_tick_step_follows_zoom(span, pixels, expected):
    step = gg.nice_tick_step(span, pixels, 80)
    assert step == pytest.approx(expected)
    assert pixels * step / span >= 80 * (1 - 1e-9)  # never denser than the minimum spacing


def test_minor_step_and_labels():
    assert gg.minor_tick_step(100) == pytest.approx(20)
    assert gg.minor_tick_step(0.2) == pytest.approx(0.05)
    assert gg.minor_tick_step(0.5) == pytest.approx(0.1)
    assert gg.format_tick(3 * 0.1, 0.1) == "0.3"
    assert gg.format_tick(-1e-17, 0.05) == "0"
    assert gg.format_tick(0.0015, 0.0005) == "0.0015"
    assert gg.format_tick(300, 100) == "300"


UPPER_EDGE = [[-300, 0.05], [-110, 0.25], [-100, 0.75], [100, 0.75], [110, 0.25], [300, 0.05]]


@pytest.mark.parametrize("closed", [False, True])
def test_mirror_upper_profile(closed):
    pts, sign = gg.mirror_outline(UPPER_EDGE, closed=closed, axis="v")
    assert sign == 1
    assert sorted(map(tuple, pts)) == sorted(map(tuple, gg.DEFAULT_OBJECT["xy"]))


def test_mirror_keeps_shape_when_already_symmetric():
    pts, _ = gg.mirror_outline(gg.DEFAULT_OBJECT["xy"], closed=True, axis="v")
    assert sorted(map(tuple, pts)) == sorted(map(tuple, gg.DEFAULT_OBJECT["xy"]))
    assert abs(gg.polygon_area(pts)) == pytest.approx(abs(gg.polygon_area(gg.DEFAULT_OBJECT["xy"])))


def test_mirror_front_view_left_right():
    right_half = [[0, -0.04], [0.05, -0.02], [0.05, 0.02], [0, 0]]
    pts, _ = gg.mirror_outline(right_half, closed=True, axis="u")
    assert sorted(map(tuple, pts)) == sorted(
        [(0, -0.04), (0.05, -0.02), (0.05, 0.02), (0, 0), (-0.05, 0.02), (-0.05, -0.02)]
    )
    gg.normalized_cross_section(pts)  # a valid, simple outline


def test_mirror_side_view_keeps_the_larger_half():
    # lower half (z < 0) has more points: it is mirrored upwards
    xz = [[-300, 0.02], [-300, -0.02], [-100, -0.125], [100, -0.125], [300, -0.02], [300, 0.02]]
    pts, sign = gg.mirror_outline(xz, closed=True, axis="v")
    assert sign == -1
    assert sorted(map(tuple, pts)) == sorted(
        [(-300, -0.02), (-100, -0.125), (100, -0.125), (300, -0.02), (300, 0.02), (100, 0.125), (-100, 0.125), (-300, 0.02)]
    )


def test_mirror_rejects_points_on_the_axis():
    with pytest.raises(gg.GeometryError):
        gg.mirror_outline([[0, 0], [1, 0], [2, 0]], closed=True, axis="v")


def test_gui_undo_redo(app):
    view = app.view_xy
    view.set_points([[0, 0], [10, 0], [10, 1]], closed=True)
    app._commit_history()
    view.set_points([[0, 0], [10, 0], [10, 1], [0, 1]], closed=True)  # e.g. an accidental extra point
    app._commit_history()
    app.undo()
    assert view.points == [[0, 0], [10, 0], [10, 1]]
    app.undo()
    assert view.points == []
    app.undo()  # nothing more: no error
    app.redo()
    app.redo()
    assert view.points == [[0, 0], [10, 0], [10, 1], [0, 1]]
    # a new edit clears the redo stack
    app.undo()
    view.set_points([[5, 5], [6, 5], [6, 6]], closed=True)
    app._commit_history()
    app.redo()
    assert view.points == [[5, 5], [6, 5], [6, 6]]


def test_gui_undo_clear_all_and_mirror(app):
    app.load_default_object()
    app._commit_history()
    app.clear_all()
    app._commit_history()
    app.undo()  # one step restores all three views
    assert all(len(v.points) >= 3 for v in app._views().values())
    app.view_yz.set_points([[0, -0.04], [0.05, -0.02], [0.05, 0.02], [0, 0]], closed=True)
    app._commit_history()
    app.mirror_view(app.view_yz, "u")
    assert len(app.view_yz.points) == 6
    app.undo()
    assert len(app.view_yz.points) == 4


def test_gui_open_model_resets_history(app, tmp_path):
    app.load_default_object()
    path = tmp_path / "m.3dgg"
    app.save_model_to(str(path))
    app.open_model_path(str(path), confirm=False)
    app.undo()
    assert len(app.view_xy.points) == 12  # nothing before the opened model


def test_gui_snap_follows_zoom(app):
    view = app.view_xy
    app.root.update()
    view.u_range, view.v_range = [-340.0, 340.0], [-1.05, 1.05]
    coarse = view.tick_steps()[0][1]
    view.u_range, view.v_range = [-0.34, 0.34], [-0.00105, 0.00105]
    fine = view.tick_steps()[0][1]
    assert fine == pytest.approx(coarse / 1000)
    view.u_guides = []  # test the grid snap only
    u, _ = view.snap_uv(0.01234, 0.0)
    assert abs(u - 0.01234) <= fine / 2 + 1e-12
    assert u / fine == pytest.approx(round(u / fine))


# ----------------------------------------------------------------------
# GUI: layers
# ----------------------------------------------------------------------
def test_gui_layers_switch_and_keep_drawings(app):
    app.load_default_object()
    first_xy = [list(p) for p in app.view_xy.points]
    app.add_layer()
    assert app.active_layer == 1 and app.layer_var.get() == "2/0"
    assert app.view_xy.points == []  # new layer starts empty
    assert app.view_xy.fill_color == gg.layer_colors(1)[0]
    assert [bg[0] for bg in app.view_xy.background] == [first_xy]  # layer 1 shown as outline
    app.view_xy.set_points([[-50, -1], [50, -1], [50, 1], [-50, 1]])
    app.select_layer(0)  # back to the first layer: its drawing is unchanged
    assert app.view_xy.points == first_xy
    app.select_layer(1)
    assert app.view_xy.points == [[-50, -1], [50, -1], [50, 1], [-50, 1]]


def test_gui_generate_and_save_several_layers(app, tmp_path):
    app.load_default_object()
    app.add_layer()
    for (key, view), pts in zip(app._views().items(), two_layers()[1][1].values()):
        view.set_points(pts)
    geoms = app.generate_geometry(quiet=True)
    assert [lay for lay, _ in geoms] == [(1, 0), (2, 0)]
    colours = {face[2] for face in app.view_3d.faces_3d}
    assert colours == {gg.layer_colors(0)[0], gg.layer_colors(1)[0]}

    path = tmp_path / "stack.3dgg"
    assert app.save_model_to(str(path))
    app.new_model()
    assert len(app.layers) == 1 and app.view_xy.points == []
    assert app.open_model_path(str(path), confirm=False)
    assert [gg.format_layer(layer["spec"]) for layer in app.layers] == ["1/0", "2/0"]
    assert app.active_layer == 1 and app.view_xy.points == [[-50, -1], [50, -1], [50, 1], [-50, 1]]
    assert len(app.geometry) == 2


def test_gui_undo_add_and_delete_layer(app):
    app.load_default_object()
    app._commit_history()
    app.add_layer()
    app.view_xy.set_points([[0, 0], [1, 0], [1, 1]])
    app._commit_history()
    app.delete_layer(confirm=False)
    assert len(app.layers) == 1 and app.active_layer == 0
    app.undo()  # layer 2/0 comes back with its drawing, and is shown
    assert len(app.layers) == 2 and app.active_layer == 1
    assert app.view_xy.points == [[0, 0], [1, 0], [1, 1]]
    app.undo()
    app.undo()  # before the layer was added
    assert len(app.layers) == 1 and app.active_layer == 0
    assert len(app.view_xy.points) == 12


def test_gui_change_layer_number(app, monkeypatch):
    app.add_layer()
    monkeypatch.setattr(gg.simpledialog, "askstring", lambda *a, **k: "1/0")
    errors = []
    monkeypatch.setattr(gg.messagebox, "showerror", lambda *a, **k: errors.append(a))
    app.change_layer_number()  # 1/0 is taken
    assert errors and app.layer_var.get() == "2/0"
    monkeypatch.setattr(gg.simpledialog, "askstring", lambda *a, **k: "10/3")
    app.change_layer_number()
    assert app.layer_var.get() == "10/3" and app.layers[1]["spec"] == (10, 3)


# ----------------------------------------------------------------------
# 3D view: hidden-surface removal (BSP tree)
# ----------------------------------------------------------------------
def _box_faces(x0, x1, y0, y1, z0, z1, tag):
    geom = gg.reconstruct_geometry(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        [[x0, z0], [x1, z0], [x1, z1], [x0, z1]],
        [[y0, z0], [y1, z0], [y1, z1], [y0, z1]],
    )
    return [(verts, tag) for verts, _ in geom.faces]


def _ray_hit(eye, direction, verts):
    """Distance along the ray to the planar polygon, or None if it misses."""
    (nx, ny, nz), d = gg._newell_plane(verts)
    denom = nx * direction[0] + ny * direction[1] + nz * direction[2]
    if abs(denom) < 1e-12:
        return None
    t = (d - (nx * eye[0] + ny * eye[1] + nz * eye[2])) / denom
    if t <= 0:
        return None
    p = [eye[k] + t * direction[k] for k in range(3)]
    drop = max(range(3), key=lambda k: abs((nx, ny, nz)[k]))  # project onto the dominant plane
    a, b = [k for k in range(3) if k != drop]
    inside = False
    for i in range(len(verts)):
        p1, p2 = verts[i], verts[i - 1]
        if (p1[b] > p[b]) != (p2[b] > p[b]):
            if p[a] < p1[a] + (p[b] - p1[b]) * (p2[a] - p1[a]) / (p2[b] - p1[b]):
                inside = not inside
    return t if inside else None


@pytest.mark.parametrize("eye", [(6, 2.5, 3), (-4, 5, 1.5), (0.5, -7, -2), (-6, -3, 4), (2, 1, -8)])
def test_bsp_order_hides_a_layer_inside_another(eye):
    # Old depth sort: faces of a layer inside another one showed through
    faces = _box_faces(-1, 1, -1, 1, -0.5, 0.5, "outer") + _box_faces(-0.6, 0.4, -0.3, 0.5, -0.2, 0.4, "inner")
    faces += _box_faces(0.5, 1.6, -0.2, 0.2, -0.8, 0.0, "crossing")  # pierces the outer box
    order = gg.bsp_back_to_front(gg.build_bsp(gg.bsp_fragments(faces)), eye)
    for i in range(-6, 7):
        for j in range(-6, 7):
            target = (0.18 * i, 0.18 * j, 0.09 * (i - j))
            direction = [target[k] - eye[k] for k in range(3)]
            hits = [(t, idx) for idx, frag in enumerate(order) if (t := _ray_hit(eye, direction, frag[0]))]
            if hits:
                nearest = min(hits)[0]
                painted = max(idx for _, idx in hits)  # the fragment drawn last wins
                assert _ray_hit(eye, direction, order[painted][0]) == pytest.approx(nearest, abs=1e-9)


def test_bsp_outlines_only_original_edges():
    faces = _box_faces(-1, 1, -1, 1, -1, 1, "a") + _box_faces(0, 2, -0.5, 0.5, -0.5, 0.5, "b")
    frags = gg.bsp_fragments(faces)
    split = gg.bsp_back_to_front(gg.build_bsp(frags), (5, 4, 3), cull=False)
    assert len(split) > len(frags)  # faces were cut where the boxes intersect
    assert any(not all(flags) for _, flags, _, _ in split)  # cuts are not outlined


def test_bsp_non_planar_quad_becomes_triangles():
    quad = [(0, 0, 0), (1, 0, 0), (1, 1, 0.3), (0, 1, 0)]
    frags = gg.bsp_fragments([(quad, "q")])
    assert len(frags) == 2 and all(len(f[0]) == 3 for f in frags)
    assert [f[1] for f in frags] == [[True, True, False], [False, True, True]]


# ----------------------------------------------------------------------
# GUI: default object, cross-view guides
# ----------------------------------------------------------------------
def test_gui_default_object_goes_to_a_new_layer(app):
    app.load_default_object()  # first layer is empty: used directly
    assert len(app.layers) == 1 and app.view_xy.points == [[float(u), float(v)] for u, v in gg.DEFAULT_OBJECT["xy"]]
    app.add_layer()
    app.view_xy.set_points([[-50, -1], [50, -1], [50, 1], [-50, 1]])
    app._commit_history()
    app.load_default_object()  # two drawn layers: the object becomes layer 3/0
    assert [gg.format_layer(layer["spec"]) for layer in app.layers] == ["1/0", "2/0", "3/0"]
    assert app.active_layer == 2 and len(app.view_xy.points) == 12
    app._sync_active_layer()
    assert app.layers[1]["views"]["xy"][0] == [[-50, -1], [50, -1], [50, 1], [-50, 1]]  # untouched
    app.undo()  # one step removes it again
    assert len(app.layers) == 2 and app.view_xy.points == [[-50, -1], [50, -1], [50, 1], [-50, 1]]


def test_gui_guides_come_from_the_other_views(app):
    app.load_default_object()
    assert app.view_yz.u_guides == sorted({p[1] for p in gg.DEFAULT_OBJECT["xy"]})  # widths of the top view
    assert app.view_yz.v_guides == [-0.125, 0.125]  # thickness of the side view
    assert app.view_xz.u_guides == sorted({p[0] for p in gg.DEFAULT_OBJECT["xy"]})
    app.view_xy.set_points([[-10, -2], [10, -2], [10, 2], [-10, 2]])
    assert app.view_yz.u_guides == [-2, 2] and app.view_xz.u_guides == [-10, 10]
