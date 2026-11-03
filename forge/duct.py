"""Generate a watertight rectangular transition duct as ASCII STL.

All dimensions are in mm. The duct is a hollow, straight loft; it does not
model mounting features, obstructions, temperature, pressure, or airflow.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Duct:
    inlet_width: float
    inlet_height: float
    outlet_width: float
    outlet_height: float
    length: float
    wall: float

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.wall * 2 >= min(self.inlet_width, self.inlet_height, self.outlet_width, self.outlet_height):
            raise ValueError("wall must leave a positive internal opening at both ends")

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _ring(width, height, z):
    return [(-width/2, -height/2, z), (width/2, -height/2, z),
            (width/2, height/2, z), (-width/2, height/2, z)]


def triangles(d):
    d.validate()
    rings = [_ring(d.inlet_width, d.inlet_height, 0),
             _ring(d.outlet_width, d.outlet_height, d.length),
             _ring(d.inlet_width-2*d.wall, d.inlet_height-2*d.wall, 0),
             _ring(d.outlet_width-2*d.wall, d.outlet_height-2*d.wall, d.length)]
    a, b, c, e = rings
    result = []
    def quad(p, q, r, s):
        result.extend([(p, q, r), (p, r, s)])
    for i in range(4):
        j = (i+1) % 4
        quad(a[i], a[j], b[j], b[i])       # outside
        quad(c[j], c[i], e[i], e[j])       # inside
        quad(a[j], a[i], c[i], c[j])       # inlet rim
        quad(b[i], b[j], e[j], e[i])       # outlet rim
    return result


def stl_text(d):
    lines = ["solid forge_duct"]
    for p, q, r in triangles(d):
        u = [q[i]-p[i] for i in range(3)]
        v = [r[i]-p[i] for i in range(3)]
        n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        norm = math.sqrt(sum(x*x for x in n))
        n = tuple(x/norm for x in n)
        lines.append("  facet normal %.9g %.9g %.9g" % n)
        lines.append("    outer loop")
        for vertex in (p, q, r):
            lines.append("      vertex %.9g %.9g %.9g" % vertex)
        lines.extend(["    endloop", "  endfacet"])
    return "\n".join(lines + ["endsolid forge_duct", ""])


def metrics(d):
    d.validate()
    inlet = (d.inlet_width-2*d.wall)*(d.inlet_height-2*d.wall)
    outlet = (d.outlet_width-2*d.wall)*(d.outlet_height-2*d.wall)
    return {"inlet_open_area_mm2": inlet, "outlet_open_area_mm2": outlet,
            "area_ratio_outlet_to_inlet": outlet/inlet,
            "max_side_angle_degrees": math.degrees(math.atan(max(abs(d.outlet_width-d.inlet_width), abs(d.outlet_height-d.inlet_height))/(2*d.length)))}


def write_candidate(d, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "duct.stl").write_text(stl_text(d), encoding="ascii")
    report = {"schema_version": 1, "geometry_sha256": d.fingerprint(), "dimensions_mm": asdict(d),
              "geometry_metrics": metrics(d), "evaluation_type": "geometry_only",
              "limitations": ["No airflow, thermal, pressure, strength, fit, or printability simulation performed.",
                              "Dimensions are outer dimensions; openings account for wall thickness.",
                              "Confirm physical clearances and safe materials before any hardware use."]}
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report
