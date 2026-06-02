#!/usr/bin/env python3
"""
Restore the per-link two-tone (dark joints / light panels) finish that the
MimicKit teaser shows. The shipped g1.usd collapsed every mesh into a single
white DefaultMaterial; we re-derive the per-mesh colors from g1.xml (MJCF),
build one Material per unique color, and rebind each mesh prim accordingly.

This is the principled fix — diffuse + metallic come from the MJCF rgba
attribute on the visual <geom> for each link, which matches what the source
of truth was before USD conversion lost the binding.

Usage:
  python scripts/g1_usd_recolor.py
  # → dependencies/MimicKit/data/assets/g1/g1_textured.usd
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdShade

REPO = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO / "dependencies/MimicKit/data/assets/g1"
MJCF = ASSET_DIR / "g1.xml"
SRC = ASSET_DIR / "g1.usd"
DST = ASSET_DIR / "g1_textured.usd"

# PBR finish — same metallic for everything, the contrast comes from diffuse.
METALLIC = 0.7
ROUGHNESS = 0.32


def parse_mjcf_colors() -> dict[str, tuple[float, float, float]]:
    """Pull per-mesh rgba from the MJCF visual <geom> entries.

    Visual geoms in MimicKit's g1.xml carry contype="0" conaffinity="0"
    group="1" — collision geoms (capsules/spheres) reuse 0.7 0.7 0.7 placeholder.
    We index by mesh name and return the visual mesh color.
    """
    root = ET.parse(MJCF).getroot()
    colors: dict[str, tuple[float, float, float]] = {}
    for geom in root.iter("geom"):
        mesh_name = geom.attrib.get("mesh")
        if not mesh_name:
            continue
        # only visual geoms — contype=0 conaffinity=0 group=1
        if geom.attrib.get("contype") != "0":
            continue
        rgba = geom.attrib.get("rgba")
        if not rgba:
            continue
        parts = rgba.strip().split()
        if len(parts) < 3:
            continue
        colors[mesh_name] = tuple(float(p) for p in parts[:3])
    return colors


def ensure_material(stage: Usd.Stage, path: str, diffuse: tuple[float, float, float]) -> Sdf.Path:
    """Create (or update) a UsdPreviewSurface material at path with this diffuse color."""
    mat_prim = stage.GetPrimAtPath(path)
    if not mat_prim:
        mat = UsdShade.Material.Define(stage, path)
        mat_prim = mat.GetPrim()
    else:
        mat = UsdShade.Material(mat_prim)

    shader_path = f"{path}/Shader"
    shader_prim = stage.GetPrimAtPath(shader_path)
    if not shader_prim:
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader_prim = shader.GetPrim()
    else:
        shader = UsdShade.Shader(shader_prim)

    # Set shader id directly via attribute (avoids stale-prim issues with UsdShade API).
    id_attr = shader_prim.GetAttribute("info:id")
    if not id_attr:
        id_attr = shader_prim.CreateAttribute("info:id", Sdf.ValueTypeNames.Token)
    id_attr.Set("UsdPreviewSurface")

    def write_input(name: str, sdf_type, value):
        attr = shader_prim.GetAttribute(f"inputs:{name}")
        if not attr:
            attr = shader_prim.CreateAttribute(f"inputs:{name}", sdf_type)
        attr.Set(value)

    write_input("diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*diffuse))
    write_input("metallic", Sdf.ValueTypeNames.Float, METALLIC)
    write_input("roughness", Sdf.ValueTypeNames.Float, ROUGHNESS)
    write_input("emissiveColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.0, 0.0, 0.0))
    write_input("opacity", Sdf.ValueTypeNames.Float, 1.0)
    write_input("ior", Sdf.ValueTypeNames.Float, 1.5)
    write_input("useSpecularWorkflow", Sdf.ValueTypeNames.Int, 0)

    # Wire shader.outputs:surface to material.outputs:surface.
    shader_out_path = Sdf.Path(f"{shader_path}.outputs:surface")
    shader_out_attr = shader_prim.GetAttribute("outputs:surface")
    if not shader_out_attr:
        shader_out_attr = shader_prim.CreateAttribute("outputs:surface", Sdf.ValueTypeNames.Token)
    mat_out_attr = mat_prim.GetAttribute("outputs:surface")
    if not mat_out_attr:
        mat_out_attr = mat_prim.CreateAttribute("outputs:surface", Sdf.ValueTypeNames.Token)
    if shader_out_path not in mat_out_attr.GetConnections():
        mat_out_attr.AddConnection(shader_out_path)

    return Sdf.Path(path)


def main():
    assert SRC.exists(), f"source USD missing: {SRC}"
    assert MJCF.exists(), f"MJCF missing: {MJCF}"

    mjcf_colors = parse_mjcf_colors()
    unique = sorted(set(mjcf_colors.values()))
    print(f"parsed {len(mjcf_colors)} mesh→color from MJCF, {len(unique)} unique colors")
    for c in unique:
        meshes = [m for m, col in mjcf_colors.items() if col == c]
        print(f"  {c} → {len(meshes)} meshes")

    stage = Usd.Stage.Open(str(SRC))

    # Build one material per unique color under /g1/Looks/.
    color2matpath: dict[tuple[float, float, float], Sdf.Path] = {}
    for i, color in enumerate(unique):
        name = f"Mat_{int(color[0]*100):02d}_{int(color[1]*100):02d}_{int(color[2]*100):02d}"
        path = f"/g1/Looks/{name}"
        color2matpath[color] = ensure_material(stage, path, color)

    # Re-bind every mesh prim. Mesh prim paths look like:
    #   /Flattened_Prototype_N/<mesh_name>/<mesh_name>
    # We use the basename to look up the MJCF color.
    rebound = 0
    skipped = []
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Mesh":
            continue
        # Mesh prim name == link name (e.g. 'torso_link') in the source USD.
        mesh_name = prim.GetName()
        color = mjcf_colors.get(mesh_name)
        if color is None:
            skipped.append(mesh_name)
            continue
        target_path = color2matpath[color]
        api = UsdShade.MaterialBindingAPI.Apply(prim)
        api.Bind(UsdShade.Material(stage.GetPrimAtPath(target_path)))
        rebound += 1

    print(f"rebound {rebound} meshes; skipped {len(skipped)}: {skipped[:5]}")

    stage.GetRootLayer().Export(str(DST))
    size_mb = DST.stat().st_size / 1024 / 1024
    print(f"✓ wrote {DST}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
