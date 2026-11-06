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
    return gg.reconstruct_geometry(gg.SLIDE1_PRESET["xy"], gg.SLIDE1_PRESET["xz"], gg.SLIDE1_PRESET["yz"])


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
    expected = sorted((round(x * 1000), round(y * 1000)) for x, y in gg.SLIDE1_PRESET["xy"])
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
    assert "LAYER = (5, 1)" in src
    assert "THICKNESS_UM = 0.25" in src
    assert "ZMIN_UM = -0.125" in src


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
    return {key: (gg.SLIDE1_PRESET[key], True) for key in gg.VIEW_KEYS}


def test_model_round_trip(tmp_path):
    path = tmp_path / "taper.3dgg"
    export = {"format": "GDSII layout (.gds)", "gds_layer": "5/1", "cell_name": "taper"}
    gg.write_model(str(path), gg.model_to_dict(preset_views(), export, snap=False))
    data = gg.read_model(str(path))
    assert data["format"] == "3dgg" and data["generator"] == f"{gg.APP_NAME} {gg.__version__}"
    for key in gg.VIEW_KEYS:
        assert data["views"][key]["points"] == [[float(u), float(v)] for u, v in gg.SLIDE1_PRESET[key]]
        assert data["views"][key]["closed"] is True
    assert data["export"] == export and data["editor"] == {"snap": False}


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
    app.load_slide1_preset()
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
    app.load_slide1_preset()
    app.format_var.set("All (.lsf + .gds + .py)")
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    monkeypatch.setattr(gg.filedialog, "asksaveasfilename", lambda **k: str(out_dir / "out.lsf"))
    app.generate_file()
    assert sorted(p.name for p in out_dir.iterdir()) == ["out.gds", "out.lsf", "out.py"]
    assert app.config["last_export_dir"] == str(out_dir)


def test_gui_save_and_open_model(app, tmp_path):
    app.load_slide1_preset()
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
    assert app.view_xy.points == [[float(u), float(v)] for u, v in gg.SLIDE1_PRESET["xy"]]
    assert app.ent_cell.get() == "my_cell"
    assert app.geometry is not None and not app.dirty
    assert app.config["recent_files"][0] == str(path)
