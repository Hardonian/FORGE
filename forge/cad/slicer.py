"""Production 3MF slicer package generator (Bambu Studio / PrusaSlicer / OrcaSlicer) with interlocking multi-part splitting."""
from __future__ import annotations
import io
import json
from typing import Any, Dict, List, Tuple
import zipfile
from forge.core.geometry import Triangle, Vec3
from forge.core.models import DuctSpec
from forge.cad.dfm import PRINTER_BED_LIMITS
from forge.cad.flanges import generate_flange_mesh
from forge.cad.mesh import build_duct_mesh, split_duct_for_printing


SLICER_MATERIAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "PETG": {
        "nozzle_temp_c": 240,
        "bed_temp_c": 70,
        "chamber_temp_c": 35,
        "wall_loops": 3,
        "sparse_infill_density": "20%",
        "sparse_infill_pattern": "gyroid",
        "max_volumetric_speed_mm3_s": 12.0,
        "fan_min_speed_pct": 40,
        "fan_max_speed_pct": 80,
        "nozzle_type": "stainless_steel_0.4mm",
    },
    "ABS": {
        "nozzle_temp_c": 260,
        "bed_temp_c": 100,
        "chamber_temp_c": 50,
        "wall_loops": 4,
        "sparse_infill_density": "25%",
        "sparse_infill_pattern": "gyroid",
        "max_volumetric_speed_mm3_s": 15.0,
        "fan_min_speed_pct": 10,
        "fan_max_speed_pct": 35,
        "nozzle_type": "hardened_steel_0.4mm",
    },
    "ASA": {
        "nozzle_temp_c": 260,
        "bed_temp_c": 100,
        "chamber_temp_c": 50,
        "wall_loops": 4,
        "sparse_infill_density": "25%",
        "sparse_infill_pattern": "gyroid",
        "max_volumetric_speed_mm3_s": 15.0,
        "fan_min_speed_pct": 10,
        "fan_max_speed_pct": 35,
        "nozzle_type": "hardened_steel_0.4mm",
    },
    "PC-CF": {
        "nozzle_temp_c": 290,
        "bed_temp_c": 110,
        "chamber_temp_c": 60,
        "wall_loops": 4,
        "sparse_infill_density": "30%",
        "sparse_infill_pattern": "gyroid",
        "max_volumetric_speed_mm3_s": 10.0,
        "fan_min_speed_pct": 0,
        "fan_max_speed_pct": 20,
        "nozzle_type": "hardened_steel_0.6mm",
    },
    "PA12-CF": {
        "nozzle_temp_c": 290,
        "bed_temp_c": 100,
        "chamber_temp_c": 55,
        "wall_loops": 4,
        "sparse_infill_density": "30%",
        "sparse_infill_pattern": "gyroid",
        "max_volumetric_speed_mm3_s": 10.0,
        "fan_min_speed_pct": 0,
        "fan_max_speed_pct": 15,
        "nozzle_type": "hardened_steel_0.6mm",
    },
    "PEEK": {
        "nozzle_temp_c": 400,
        "bed_temp_c": 140,
        "chamber_temp_c": 120,
        "wall_loops": 5,
        "sparse_infill_density": "40%",
        "sparse_infill_pattern": "gyroid",
        "max_volumetric_speed_mm3_s": 6.0,
        "fan_min_speed_pct": 0,
        "fan_max_speed_pct": 10,
        "nozzle_type": "hardened_steel_0.6mm",
    },
}


def split_duct_with_interlocking_flanges(
    spec: DuctSpec,
    split_ratio: float = 0.5,
) -> Tuple[List[Triangle], List[Triangle]]:
    """Split an oversized duct into two halves with aligned mating coupling flanges at the split plane."""
    part_a_spec, part_b_spec = split_duct_for_printing(spec, split_ratio=split_ratio)
    tris_a = build_duct_mesh(part_a_spec)
    tris_b = build_duct_mesh(part_b_spec)

    # Add mating union flange collar at the outlet of part_a (z = part_a_spec.length - 3.5mm)
    collar_t = 3.5
    mw = part_a_spec.outlet_width
    mh = part_a_spec.outlet_height
    collar_a = generate_flange_mesh(
        inner_width=mw,
        inner_height=mh,
        flange_width=mw + 14.0,
        flange_height=mh + 14.0,
        flange_thickness=collar_t,
        z_start=max(0.0, part_a_spec.length - collar_t),
    )
    # Translate collar_a to part_a's outlet XY offset
    shift_a = Vec3(part_a_spec.offset_x, part_a_spec.offset_y, 0.0)
    for tri in collar_a:
        tris_a.append(Triangle(tri.a + shift_a, tri.b + shift_a, tri.c + shift_a))

    # Add matching mating flange collar at the inlet of part_b (z = 0.0)
    collar_b = generate_flange_mesh(
        inner_width=mw,
        inner_height=mh,
        flange_width=mw + 14.0,
        flange_height=mh + 14.0,
        flange_thickness=collar_t,
        z_start=0.0,
    )
    tris_b.extend(collar_b)

    return (tris_a, tris_b)


def _mesh_to_3mf_object_xml(object_id: int, object_name: str, triangles: List[Triangle]) -> str:
    """Serialize a triangle list into a 3MF <object> XML block."""
    vert_map: Dict[Tuple[float, float, float], int] = {}
    verts: List[Vec3] = []
    tri_indices: List[Tuple[int, int, int]] = []

    for t in triangles:
        idxs = []
        for v in (t.a, t.b, t.c):
            key = (round(v.x, 5), round(v.y, 5), round(v.z, 5))
            if key not in vert_map:
                vert_map[key] = len(verts)
                verts.append(v)
            idxs.append(vert_map[key])
        tri_indices.append((idxs[0], idxs[1], idxs[2]))

    v_lines = [f'          <vertex x="{v.x:.5f}" y="{v.y:.5f}" z="{v.z:.5f}" />' for v in verts]
    t_lines = [f'          <triangle v1="{i1}" v2="{i2}" v3="{i3}" />' for (i1, i2, i3) in tri_indices]

    return (
        f'    <object id="{object_id}" name="{object_name}" type="model">\n'
        "      <mesh>\n"
        "        <vertices>\n" + "\n".join(v_lines) + "\n        </vertices>\n"
        "        <triangles>\n" + "\n".join(t_lines) + "\n        </triangles>\n"
        "      </mesh>\n"
        "    </object>"
    )


def export_production_slicer_3mf(
    spec: DuctSpec,
    triangles: List[Triangle] | None = None,
    printer_profile: str = "Bambu_Lab_X1C",
    auto_split_if_oversized: bool = True,
) -> bytes:
    """Export a production-ready 3MF container with embedded Bambu/Prusa slicer profiles and auto-splitting."""
    bed_x, bed_y, bed_z = PRINTER_BED_LIMITS.get(printer_profile, (256.0, 256.0, 256.0))
    needs_split = auto_split_if_oversized and (spec.length > bed_z)

    objects_xml: List[str] = []
    build_items_xml: List[str] = []

    if needs_split:
        tris_a, tris_b = split_duct_with_interlocking_flanges(spec, split_ratio=0.5)
        objects_xml.append(_mesh_to_3mf_object_xml(1, "FORGE_Duct_Lower_Section", tris_a))
        objects_xml.append(_mesh_to_3mf_object_xml(2, "FORGE_Duct_Upper_Section", tris_b))
        build_items_xml.append('    <item objectid="1" />')
        build_items_xml.append(f'    <item objectid="2" transform="1 0 0 0 1 0 0 0 1 {spec.inlet_width + 25.0:.2f} 0 0" />')
    else:
        tris = triangles if triangles is not None else build_duct_mesh(spec)
        objects_xml.append(_mesh_to_3mf_object_xml(1, "FORGE_Duct_Primary", tris))
        build_items_xml.append('    <item objectid="1" />')

    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        '  <metadata name="Application">FORGE Generative CAD Engine v1.0.0</metadata>\n'
        f'  <metadata name="Material">{spec.material.value}</metadata>\n'
        f'  <metadata name="TargetPrinter">{printer_profile}</metadata>\n'
        "  <resources>\n"
        + "\n".join(objects_xml)
        + "\n  </resources>\n"
        "  <build>\n"
        + "\n".join(build_items_xml)
        + "\n  </build>\n"
        "</model>\n"
    )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
        '  <Default Extension="config" ContentType="text/xml" />\n'
        "</Types>\n"
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
        "</Relationships>\n"
    )

    mat_cfg = SLICER_MATERIAL_PROFILES.get(spec.material.value, SLICER_MATERIAL_PROFILES["PETG"])
    project_settings = {
        "printer_model": printer_profile,
        "filament_type": spec.material.value,
        "bed_dimensions_mm": [bed_x, bed_y, bed_z],
        "auto_split_applied": needs_split,
        "slicer_parameters": mat_cfg,
        "flange_reinforcement": {
            "enabled": bool(spec.flange_thickness > 0.0 or needs_split),
            "flange_zone_infill_density": "100%",
            "flange_wall_loops": 6,
        },
        "overhang_optimization": {
            "slow_down_for_curled_perimeters": True,
            "bridge_speed_mm_s": 25.0,
        },
    }

    model_settings_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<config>\n"
        '  <object id="1">\n'
        f'    <metadata key="name" value="FORGE_{spec.material.value}_Duct"/>\n'
        f'    <metadata key="wall_loops" value="{mat_cfg["wall_loops"]}"/>\n'
        f'    <metadata key="sparse_infill_density" value="{mat_cfg["sparse_infill_density"]}"/>\n'
        "  </object>\n"
        "</config>\n"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("3D/3dmodel.model", model_xml)
        zf.writestr("Metadata/project_settings.config", json.dumps(project_settings, indent=2))
        zf.writestr("Metadata/model_settings.config", model_settings_xml)

    return buf.getvalue()
