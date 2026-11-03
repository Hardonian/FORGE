"""Design for Manufacturing (DFM) analysis for 3D printed thermal hardware."""
from __future__ import annotations
import math
from typing import List
from forge.core.geometry import Triangle, calculate_mesh_volume
from forge.core.models import DFMReport, DuctSpec, MATERIAL_PROPERTIES
from forge.cad.mesh import compute_normal_wall_compensation


PRINTER_BED_LIMITS = {
    "Bambu_Lab_X1C": (256.0, 256.0, 256.0),
    "Prusa_MK4": (250.0, 210.0, 220.0),
    "Voron_2.4_300": (300.0, 300.0, 300.0),
    "Industrial_Fortus": (406.0, 355.0, 406.0),
}


def analyze_dfm(spec: DuctSpec, triangles: List[Triangle], target_printer: str = "Bambu_Lab_X1C") -> DFMReport:
    """Analyze mesh for 3D printability, overhangs, material mass, normal wall thickness, and heat deflection resistance."""
    # 1. Bounding box calculation
    all_pts = [p for t in triangles for p in (t.a, t.b, t.c)]
    min_x = min(p.x for p in all_pts)
    max_x = max(p.x for p in all_pts)
    min_y = min(p.y for p in all_pts)
    max_y = max(p.y for p in all_pts)
    min_z = min(p.z for p in all_pts)
    max_z = max(p.z for p in all_pts)

    dx = max_x - min_x
    dy = max_y - min_y
    dz = max_z - min_z

    bed_x, bed_y, bed_z = PRINTER_BED_LIMITS.get(target_printer, (250.0, 250.0, 250.0))
    fits_bed = (dx <= bed_x) and (dy <= bed_y) and (dz <= bed_z)

    # 2. Solid wall volume and mass calculation
    solid_volume_mm3 = abs(calculate_mesh_volume(triangles))
    solid_volume_cm3 = solid_volume_mm3 / 1000.0

    mat_info = MATERIAL_PROPERTIES.get(spec.material.value, {"density": 1.25, "hdt_c": 75.0, "max_continuous_temp_c": 70.0})
    density = mat_info["density"]
    estimated_mass_g = solid_volume_cm3 * density

    # 3. Print time estimation (approx 18 cm^3/hr print speed with 0.4mm nozzle)
    estimated_hours = max(0.5, solid_volume_cm3 / 18.0)

    # 4. Normal wall thickness check
    _, _, min_normal_wall = compute_normal_wall_compensation(spec)

    # 5. Overhang angle analysis (against print bed Z-normal)
    max_overhang_deg = 0.0
    violates_overhang = False
    warnings: List[str] = []

    for tri in triangles:
        n = tri.normal
        # Ignore flat base plate sitting on print bed (z == min_z)
        if abs(tri.a.z - min_z) < 1e-4 and abs(tri.b.z - min_z) < 1e-4 and abs(tri.c.z - min_z) < 1e-4:
            continue
        if n.z < -0.01:
            cos_theta = max(-1.0, min(1.0, -n.z))
            angle_from_downward = math.degrees(math.acos(cos_theta))
            overhang_deg = 90.0 - angle_from_downward
            if overhang_deg > max_overhang_deg:
                max_overhang_deg = overhang_deg
            if overhang_deg > 50.0:
                violates_overhang = True

    if violates_overhang:
        warnings.append(f"Steep overhang of {max_overhang_deg:.1f} deg detected. Reorient on print bed or enable tree supports.")

    if not fits_bed:
        warnings.append(f"Dimensions ({dx:.1f}x{dy:.1f}x{dz:.1f}mm) exceed {target_printer} build volume ({bed_x}x{bed_y}x{bed_z}mm). Consider splitting into multi-part interlocking segments.")

    if min_normal_wall < 1.6:
        warnings.append(f"Effective normal wall thickness {min_normal_wall:.2f}mm is below recommended 1.6mm (4 perimeters with 0.4mm nozzle). Consider enabling preserve_normal_thickness.")

    if spec.material.value in ("PETG", "PLA") and mat_info["max_continuous_temp_c"] < 75.0:
        warnings.append(f"Material {spec.material.value} has continuous temperature rating of {mat_info['max_continuous_temp_c']} C. Risk of thermal creep in high-load AI server racks (>80 C exhaust). Recommend PC-CF or PA12-CF.")

    return DFMReport(
        material=spec.material.value,
        estimated_mass_grams=round(estimated_mass_g, 2),
        estimated_print_time_hours=round(estimated_hours, 2),
        overhang_compliance=not violates_overhang,
        max_overhang_angle_deg=round(max_overhang_deg, 2),
        bed_fit=fits_bed,
        min_normal_wall_mm=round(min_normal_wall, 3),
        warnings=warnings,
    )
