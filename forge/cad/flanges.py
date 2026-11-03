"""Server fan mounting flanges and hardware mounting features."""
from __future__ import annotations
from typing import Dict, List, NamedTuple
from forge.core.geometry import Vec3, Triangle


class FanFlangeDimensions(NamedTuple):
    frame_size_mm: float
    hole_spacing_mm: float
    hole_diameter_mm: float
    screw_type: str
    brass_insert_od_mm: float


# Industry-standard server and industrial fan specifications (San Ace / Delta / Noctua Industrial)
FAN_STANDARDS: Dict[str, FanFlangeDimensions] = {
    "40mm": FanFlangeDimensions(40.0, 32.0, 3.5, "M3", 4.6),
    "60mm": FanFlangeDimensions(60.0, 50.0, 4.3, "M4", 5.6),
    "80mm": FanFlangeDimensions(80.0, 71.5, 4.3, "M4", 5.6),
    "92mm": FanFlangeDimensions(92.0, 82.5, 4.3, "M4", 5.6),
    "120mm": FanFlangeDimensions(120.0, 105.0, 4.3, "M4", 5.6),
    "140mm": FanFlangeDimensions(140.0, 125.0, 4.5, "M4", 5.6),
}


def get_fan_hole_positions(frame_size: float, hole_spacing: float, z: float = 0.0) -> List[Vec3]:
    """Get the 4 mounting hole center coordinates for a standard square fan pattern."""
    h = hole_spacing / 2.0
    return [
        Vec3(-h, -h, z),
        Vec3(h, -h, z),
        Vec3(h, h, z),
        Vec3(-h, h, z),
    ]


def generate_flange_mesh(
    inner_width: float,
    inner_height: float,
    flange_width: float,
    flange_height: float,
    flange_thickness: float,
    z_start: float = 0.0,
) -> List[Triangle]:
    """Generate a watertight mounting collar/flange around a duct opening."""
    z0 = z_start
    z1 = z_start - flange_thickness if z_start <= 0 else z_start + flange_thickness

    # Outer rectangle points
    ow, oh = flange_width / 2.0, flange_height / 2.0
    outer_front = [Vec3(-ow, -oh, z0), Vec3(ow, -oh, z0), Vec3(ow, oh, z0), Vec3(-ow, oh, z0)]
    outer_back = [Vec3(-ow, -oh, z1), Vec3(ow, -oh, z1), Vec3(ow, oh, z1), Vec3(-ow, oh, z1)]

    # Inner opening points
    iw, ih = inner_width / 2.0, inner_height / 2.0
    inner_front = [Vec3(-iw, -ih, z0), Vec3(iw, -ih, z0), Vec3(iw, ih, z0), Vec3(-iw, ih, z0)]
    inner_back = [Vec3(-iw, -ih, z1), Vec3(iw, -ih, z1), Vec3(iw, ih, z1), Vec3(-iw, ih, z1)]

    tris: List[Triangle] = []

    def add_quad(p: Vec3, q: Vec3, r: Vec3, s: Vec3):
        tris.append(Triangle(p, q, r))
        tris.append(Triangle(p, r, s))

    # Flange faces
    for i in range(4):
        j = (i + 1) % 4
        # Outer rim
        add_quad(outer_front[i], outer_front[j], outer_back[j], outer_back[i])
        # Inner rim (bore)
        add_quad(inner_back[i], inner_back[j], inner_front[j], inner_front[i])
        # Front face (collar lip)
        add_quad(outer_front[j], outer_front[i], inner_front[i], inner_front[j])
        # Back face
        add_quad(outer_back[i], outer_back[j], inner_back[j], inner_back[i])

    return tris
