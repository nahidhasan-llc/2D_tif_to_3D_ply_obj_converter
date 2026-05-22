#!/usr/bin/env python3
"""
cyberware_to_obj.py
────────────────────────────────────────────────────────────────
Convert a Cyberware 3030/RGB range file + paired color TIF
into a colored OBJ file (vertex colors, no texture file needed).

Usage
-----
    python cyberware_to_obj.py <range_file> <color_tif> [output.obj]

Dependencies
------------
    pip install numpy Pillow

Example
-------
    python cyberware_to_obj.py pat1day0C pat1day0C.tif
    python cyberware_to_obj.py pat1day0C pat1day0C.tif output/pat1day0C.obj
"""

import sys, math, os
import numpy as np
from PIL import Image

INVALID_SENTINEL = 0x8000


def parse_header(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"Cyberware"):
        raise ValueError(f"'{filepath}' is not a Cyberware range file.")
    idx = raw.find(b"DATA=\n")
    if idx == -1:
        raise ValueError("Could not find DATA= marker.")
    header_end = idx + len(b"DATA=\n")
    params = {}
    for line in raw[:header_end].decode("ascii", errors="replace").split("\n"):
        if "=" in line and not line.startswith("DATA"):
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params, header_end, raw


def cyberware_to_obj(range_path, color_path, output_path):
    print(f"\n{'─'*55}")
    print(f"  Cyberware 3030/RGB → OBJ")
    print(f"  Range : {range_path}")
    print(f"  Color : {color_path}")
    print(f"  Out   : {output_path}")
    print(f"{'─'*55}\n")

    # ── 1. Parse header ───────────────────────────────────────────────────────
    params, header_end, raw = parse_header(range_path)
    NLG    = int(params["NLG"])
    NLT    = int(params["NLT"])
    RSHIFT = int(params["RSHIFT"])
    LGINCR = int(params["LGINCR"])
    LTINCR = int(params["LTINCR"])

    N_THETA    = NLG
    N_Z        = NLT
    r_scale_mm = LGINCR / 32768.0
    z_scale_mm = LTINCR / 1000.0
    theta_step = (2.0 * math.pi) / N_THETA

    print(f"  Angular steps : {N_THETA}  ({math.degrees(theta_step):.4f}°/step)")
    print(f"  Height steps  : {N_Z}  ({z_scale_mm:.4f} mm/step → {N_Z*z_scale_mm:.1f} mm total)")
    print(f"  Radius scale  : (raw >> {RSHIFT}) × {r_scale_mm:.6f} mm/unit\n")

    # ── 2. Read range data ────────────────────────────────────────────────────
    data = (np.frombuffer(raw[header_end:header_end + NLG*NLT*2], dtype=">u2")
              .reshape(NLG, NLT)
              .astype(np.float32))

    valid_mask = (data != INVALID_SENTINEL) & (data > 0)
    radius_mm  = np.where(valid_mask, (data / (2**RSHIFT)) * r_scale_mm, np.nan)
    valid_mask = ~np.isnan(radius_mm) & (radius_mm > 0)

    n_valid = int(valid_mask.sum())
    print(f"  Valid points  : {n_valid:,} / {N_THETA*N_Z:,}  ({100*n_valid/(N_THETA*N_Z):.1f}%)")

    if n_valid == 0:
        raise RuntimeError("No valid range points found.")

    # ── 3. Cylindrical → Cartesian ────────────────────────────────────────────
    Z_grid, THETA = np.meshgrid(
        np.arange(N_Z)     * z_scale_mm,
        np.arange(N_THETA) * theta_step
    )
    X = radius_mm * np.cos(THETA)
    Y = radius_mm * np.sin(THETA)
    Z = Z_grid

    print(f"  X : {np.nanmin(X):.1f} – {np.nanmax(X):.1f} mm")
    print(f"  Y : {np.nanmin(Y):.1f} – {np.nanmax(Y):.1f} mm")
    print(f"  Z : {np.nanmin(Z[valid_mask]):.1f} – {np.nanmax(Z[valid_mask]):.1f} mm\n")

    # ── 4. Color ──────────────────────────────────────────────────────────────
    color  = np.array(Image.open(color_path).convert("RGB"))
    ch, cw = color.shape[:2]

    # ── 5. Build colored point cloud ──────────────────────────────────────────
    rows, cols = np.where(valid_mask)
    pts        = np.column_stack([X[valid_mask], Y[valid_mask], Z[valid_mask]])
    colors     = color[
        (rows * ch / N_THETA).astype(int).clip(0, ch-1),
        (cols * cw / N_Z).astype(int).clip(0, cw-1)
    ].astype(np.uint8)

    # ── 6. Write OBJ with vertex colors ──────────────────────────────────────
    # Vertex colors in OBJ: "v x y z r g b" (r/g/b normalized 0-1)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Cyberware 3030/RGB scan\n")
        f.write(f"# {n_valid:,} points\n\n")
        for i in range(len(pts)):
            r, g, b = colors[i] / 255.0
            f.write(f"v {pts[i,0]:.4f} {pts[i,1]:.4f} {pts[i,2]:.4f} {r:.4f} {g:.4f} {b:.4f}\n")

    print(f"  ✓ Saved : {output_path}")
    print(f"    {len(pts):,} points  |  {os.path.getsize(output_path)/1e6:.2f} MB")
    print(f"{'─'*55}\n")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    range_path  = sys.argv[1]
    color_path  = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) >= 4 else os.path.splitext(range_path)[0] + ".obj"

    for path in [range_path, color_path]:
        if not os.path.isfile(path):
            print(f"Error: file not found: '{path}'")
            sys.exit(1)

    try:
        cyberware_to_obj(range_path, color_path, output_path)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()