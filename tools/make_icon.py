#!/usr/bin/env python3
"""
Generates the 3D Geometry Generation application / document icon.

The icon shows the two-stage tapered InP waveguide of generate_structure.lsf
(Slide 1 preset) as an extruded 3D solid, its long axis at 45 degrees to the
horizontal, on a white rounded-square background.

Outputs (in ../assets):
    3dgg_icon.png      1024 x 1024 master
    3dgg_icon_256.png  256 x 256 (window icon, Linux launcher)
    3dgg.icns          macOS icon (only when 'iconutil' and 'sips' are available)

Requires Pillow (pip install pillow). Only needed to regenerate the icons.
"""
import math
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")

SS = 4  # supersampling factor for anti-aliasing
SIZE = 1024 * SS

# Top view (x-y) of generate_structure.lsf, in micrometers
OUTLINE_UM = [
    (-300.0, -0.05), (-110.0, -0.25), (-100.0, -0.75), (100.0, -0.75), (110.0, -0.25), (300.0, -0.05),
    (300.0, 0.05), (110.0, 0.25), (100.0, 0.75), (-100.0, 0.75), (-110.0, 0.25), (-300.0, 0.05),
]

TOP = (255, 205, 64)
TOP_EDGE = (176, 116, 14)
SIDE_LIGHT = (226, 150, 22)
SIDE_DARK = (178, 105, 10)


def schematic_outline():
    """Exaggerates the width so the 600 um x 1.5 um taper reads as a shape (same idea as the 3D view)."""
    half_len = 0.41 * SIZE  # half length of the waveguide along its axis
    half_w = 0.115 * SIZE  # half of the 1.5 um core width
    return [(x / 300.0 * half_len, y / 0.75 * half_w) for x, y in OUTLINE_UM]


def to_image(pts, angle_deg=45.0, drop=0.0):
    """Rotates (x along axis, y lateral) by angle (counter-clockwise) and maps to image pixels."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = SIZE / 2, SIZE / 2 - 0.035 * SIZE  # slightly above centre: room for the visible thickness
    return [(cx + x * c - y * s, cy - (x * s + y * c) + drop) for x, y in pts]


def make_master():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # White rounded-square tile (macOS icon grid: 824/1024 tile) with a soft drop shadow
    tile = 824 * SS
    off = (SIZE - tile) // 2
    radius = 185 * SS
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (off, off + 12 * SS, off + tile, off + tile + 12 * SS), radius, fill=(0, 0, 0, 70)
    )
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14 * SS)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((off, off, off + tile, off + tile), radius, fill=(255, 255, 255, 255))
    d.rounded_rectangle((off, off, off + tile, off + tile), radius, outline=(222, 226, 232, 255), width=3 * SS)

    outline = schematic_outline()
    thickness = 0.055 * SIZE
    top = to_image(outline)
    bottom = to_image(outline, drop=thickness)

    # Soft contact shadow under the solid
    sh = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(sh).polygon(to_image(outline, drop=thickness * 1.6), fill=(40, 50, 70, 45))
    img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(22 * SS)))

    # Side walls: an edge is visible when its outward normal points downwards on screen
    n = len(top)
    area2 = sum(top[i][0] * top[(i + 1) % n][1] - top[(i + 1) % n][0] * top[i][1] for i in range(n))
    orient = 1 if area2 > 0 else -1
    walls = []
    for i in range(n):
        j = (i + 1) % n
        ex, ey = top[j][0] - top[i][0], top[j][1] - top[i][1]
        nx, ny = ey * orient, -ex * orient  # outward normal in image coordinates
        length = math.hypot(nx, ny) or 1.0
        nx, ny = nx / length, ny / length
        if ny > 0:  # faces the viewer (downwards)
            shade = SIDE_LIGHT if nx < 0 else SIDE_DARK
            walls.append((top[i][1] + top[j][1], [top[i], top[j], bottom[j], bottom[i]], shade))
    for _, quad, shade in sorted(walls, key=lambda w: w[0]):
        d.polygon(quad, fill=shade, outline=TOP_EDGE)
        d.line(quad + [quad[0]], fill=TOP_EDGE, width=2 * SS, joint="curve")

    # Top face
    d.polygon(top, fill=TOP)
    d.line(top + [top[0]], fill=TOP_EDGE, width=4 * SS, joint="curve")

    return img.resize((1024, 1024), Image.LANCZOS)


def make_icns(master, path):
    if not (shutil.which("iconutil") and sys.platform == "darwin"):
        print("iconutil not available — skipping .icns")
        return
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "3dgg.iconset")
        os.mkdir(iconset)
        for base in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = base * scale
                name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
                master.resize((px, px), Image.LANCZOS).save(os.path.join(iconset, name))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", path], check=True)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    master = make_master()
    master.save(os.path.join(ASSETS, "3dgg_icon.png"))
    master.resize((256, 256), Image.LANCZOS).save(os.path.join(ASSETS, "3dgg_icon_256.png"))
    make_icns(master, os.path.join(ASSETS, "3dgg.icns"))
    print("Icons written to", ASSETS)


if __name__ == "__main__":
    main()
