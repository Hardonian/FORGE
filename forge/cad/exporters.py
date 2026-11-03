"""CAD and 3D printing mesh exporters: Binary STL, ASCII STL, 3MF, OBJ, and STEP."""
from __future__ import annotations
import io
import struct
import zipfile
from typing import List
from forge.core.geometry import Triangle


def export_ascii_stl(triangles: List[Triangle], solid_name: str = "forge_duct") -> str:
    """Generate ASCII STL representation."""
    lines = [f"solid {solid_name}"]
    for tri in triangles:
        n = tri.normal
        lines.append(f"  facet normal {n.x:.9g} {n.y:.9g} {n.z:.9g}")
        lines.append("    outer loop")
        for v in (tri.a, tri.b, tri.c):
            lines.append(f"      vertex {v.x:.9g} {v.y:.9g} {v.z:.9g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    return "\n".join(lines) + "\n"


def export_binary_stl(triangles: List[Triangle], header_text: str = "FORGE Binary STL - Parametric Inverse Design") -> bytes:
    """Generate compact standard Binary STL bytes (IEEE 754 little-endian)."""
    # 80-byte header
    header_bytes = header_text.encode("ascii", errors="replace")[:80].ljust(80, b"\0")
    num_triangles = len(triangles)

    buf = io.BytesIO()
    buf.write(header_bytes)
    buf.write(struct.pack("<I", num_triangles))

    for tri in triangles:
        n = tri.normal
        # Format: normal (3 floats), vertex a (3 floats), vertex b (3 floats), vertex c (3 floats), uint16 attribute
        data = struct.pack(
            "<12fH",
            n.x, n.y, n.z,
            tri.a.x, tri.a.y, tri.a.z,
            tri.b.x, tri.b.y, tri.b.z,
            tri.c.x, tri.c.y, tri.c.z,
            0,
        )
        buf.write(data)

    return buf.getvalue()


def export_3mf(triangles: List[Triangle], model_name: str = "FORGE_Transition_Duct") -> bytes:
    """Generate standard 3MF (3D Manufacturing Format) container package."""
    # Deduplicate vertices
    vertex_map = {}
    vertices = []
    tri_indices = []

    def get_vertex_index(v):
        key = (round(v.x, 5), round(v.y, 5), round(v.z, 5))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append(v)
        return vertex_map[key]

    for tri in triangles:
        ia = get_vertex_index(tri.a)
        ib = get_vertex_index(tri.b)
        ic = get_vertex_index(tri.c)
        tri_indices.append((ia, ib, ic))

    # Build 3D Model XML
    model_xml = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
        '  <metadata name="Title">FORGE Parametric Cooling Duct</metadata>',
        '  <metadata name="Designer">FORGE Inverse Design Engine</metadata>',
        '  <resources>',
        f'    <object id="1" type="model" name="{model_name}">',
        '      <mesh>',
        '        <vertices>',
    ]
    for v in vertices:
        model_xml.append(f'          <vertex x="{v.x:.4f}" y="{v.y:.4f}" z="{v.z:.4f}" />')
    model_xml.extend([
        '        </vertices>',
        '        <triangles>',
    ])
    for ia, ib, ic in tri_indices:
        model_xml.append(f'          <triangle v1="{ia}" v2="{ib}" v3="{ic}" />')
    model_xml.extend([
        '        </triangles>',
        '      </mesh>',
        '    </object>',
        '  </resources>',
        '  <build>',
        '    <item objectid="1" />',
        '  </build>',
        '</model>',
    ])
    model_content = "\n".join(model_xml).encode("utf-8")

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />\n'
        '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
        '</Types>'
    ).encode("utf-8")

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
        '</Relationships>'
    ).encode("utf-8")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("3D/3dmodel.model", model_content)

    return zip_buffer.getvalue()


def export_obj(triangles: List[Triangle]) -> str:
    """Generate Wavefront OBJ text representation."""
    lines = ["# FORGE Generative Cooling Duct OBJ", "o forge_duct"]
    vertex_map = {}
    vertices = []
    faces = []

    def get_v_idx(v):
        key = (round(v.x, 5), round(v.y, 5), round(v.z, 5))
        if key not in vertex_map:
            vertex_map[key] = len(vertices) + 1  # 1-based indexing in OBJ
            vertices.append(v)
        return vertex_map[key]

    for tri in triangles:
        ia = get_v_idx(tri.a)
        ib = get_v_idx(tri.b)
        ic = get_v_idx(tri.c)
        faces.append((ia, ib, ic))

    for v in vertices:
        lines.append(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}")
    for ia, ib, ic in faces:
        lines.append(f"f {ia} {ib} {ic}")

    return "\n".join(lines) + "\n"


def export_step(triangles: List[Triangle], name: str = "FORGE_DUCT") -> str:
    """Generate standard ISO-10303-21 STEP faceted boundary representation file."""
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('FORGE Faceted B-Rep Model'), '2;1');",
        f"FILE_NAME('{name}.stp', '2026-11-02T20:00:00', ('FORGE Inverse Design'), ('AI Thermal'), 'FORGE v0.2.0', 'FORGE Kernel', '');",
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));",
        "ENDSEC;",
        "DATA;",
        "#1 = CARTESIAN_POINT('', (0., 0., 0.));",
        "#2 = DIRECTION('', (0., 0., 1.));",
        "#3 = DIRECTION('', (1., 0., 0.));",
        "#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);",
    ]

    # Write points and triangles
    eid = 10
    point_ids = {}
    for tri in triangles:
        for v in (tri.a, tri.b, tri.c):
            key = (round(v.x, 5), round(v.y, 5), round(v.z, 5))
            if key not in point_ids:
                point_ids[key] = eid
                lines.append(f"#{eid} = CARTESIAN_POINT('', ({v.x:.6f}, {v.y:.6f}, {v.z:.6f}));")
                eid += 1

    face_ids = []
    for tri in triangles:
        p1 = point_ids[(round(tri.a.x, 5), round(tri.a.y, 5), round(tri.a.z, 5))]
        p2 = point_ids[(round(tri.b.x, 5), round(tri.b.y, 5), round(tri.b.z, 5))]
        p3 = point_ids[(round(tri.c.x, 5), round(tri.c.y, 5), round(tri.c.z, 5))]

        # Polyloop
        lines.append(f"#{eid} = POLY_LOOP('', (#{p1}, #{p2}, #{p3}));")
        loop_id = eid
        eid += 1
        lines.append(f"#{eid} = FACE_OUTER_BOUND('', #{loop_id}, .T.);")
        bound_id = eid
        eid += 1
        lines.append(f"#{eid} = FACETED_BREP_SHAPE_REPRESENTATION('', (#{bound_id}), #4);")
        face_ids.append(eid)
        eid += 1

    lines.extend([
        "ENDSEC;",
        "END-ISO-10303-21;",
    ])
    return "\n".join(lines) + "\n"
