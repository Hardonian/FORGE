"""CAD and 3D printing mesh exporters: Binary STL, ASCII STL, 3MF, OBJ, STEP, and glTF/GLB."""
from __future__ import annotations
import base64
import io
import json
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
    header_bytes = header_text.encode("ascii", errors="replace")[:80].ljust(80, b"\0")
    num_triangles = len(triangles)

    buf = io.BytesIO()
    buf.write(header_bytes)
    buf.write(struct.pack("<I", num_triangles))

    for tri in triangles:
        n = tri.normal
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
    """Generate Wavefront OBJ text representation with vertex normals."""
    lines = ["# FORGE Generative Cooling Duct OBJ", "o forge_duct"]
    vertex_map = {}
    vertices = []
    normals = []
    faces = []

    def get_v_idx(v):
        key = (round(v.x, 5), round(v.y, 5), round(v.z, 5))
        if key not in vertex_map:
            vertex_map[key] = len(vertices) + 1  # 1-based indexing in OBJ
            vertices.append(v)
        return vertex_map[key]

    for idx, tri in enumerate(triangles, 1):
        ia = get_v_idx(tri.a)
        ib = get_v_idx(tri.b)
        ic = get_v_idx(tri.c)
        normals.append(tri.normal)
        faces.append((ia, ib, ic, idx))

    for v in vertices:
        lines.append(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}")
    for n in normals:
        lines.append(f"vn {n.x:.6f} {n.y:.6f} {n.z:.6f}")
    for ia, ib, ic, inorm in faces:
        lines.append(f"f {ia}//{inorm} {ib}//{inorm} {ic}//{inorm}")

    return "\n".join(lines) + "\n"


def export_step(triangles: List[Triangle], name: str = "FORGE_DUCT") -> str:
    """Generate ISO-10303-21 STEP faceted boundary representation with CLOSED_SHELL & MANIFOLD_SOLID_BREP."""
    lines = [
        "ISO-10303-21;",
        "HEADER;",
        "FILE_DESCRIPTION(('FORGE Faceted B-Rep Solid Model'), '2;1');",
        f"FILE_NAME('{name}.stp', '2026-11-02T20:00:00', ('FORGE Inverse Design'), ('AI Thermal'), 'FORGE v0.2.0', 'FORGE Kernel', '');",
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));",
        "ENDSEC;",
        "DATA;",
        "#1 = CARTESIAN_POINT('', (0., 0., 0.));",
        "#2 = DIRECTION('', (0., 0., 1.));",
        "#3 = DIRECTION('', (1., 0., 0.));",
        "#4 = AXIS2_PLACEMENT_3D('', #1, #2, #3);",
    ]

    eid = 10
    vertex_ids = {}
    for tri in triangles:
        for v in (tri.a, tri.b, tri.c):
            key = (round(v.x, 5), round(v.y, 5), round(v.z, 5))
            if key not in vertex_ids:
                pt_id = eid
                lines.append(f"#{pt_id} = CARTESIAN_POINT('', ({v.x:.6f}, {v.y:.6f}, {v.z:.6f}));")
                eid += 1
                vtx_id = eid
                lines.append(f"#{vtx_id} = VERTEX_POINT('', #{pt_id});")
                vertex_ids[key] = vtx_id
                eid += 1

    face_ids = []
    for tri in triangles:
        p1 = vertex_ids[(round(tri.a.x, 5), round(tri.a.y, 5), round(tri.a.z, 5))]
        p2 = vertex_ids[(round(tri.b.x, 5), round(tri.b.y, 5), round(tri.b.z, 5))]
        p3 = vertex_ids[(round(tri.c.x, 5), round(tri.c.y, 5), round(tri.c.z, 5))]

        loop_id = eid
        lines.append(f"#{loop_id} = POLY_LOOP('', (#{p1}, #{p2}, #{p3}));")
        eid += 1
        bound_id = eid
        lines.append(f"#{bound_id} = FACE_OUTER_BOUND('', #{loop_id}, .T.);")
        eid += 1
        face_id = eid
        lines.append(f"#{face_id} = FACE_SURFACE('', (#{bound_id}), #4, .T.);")
        face_ids.append(f"#{face_id}")
        eid += 1

    shell_id = eid
    lines.append(f"#{shell_id} = CLOSED_SHELL('', ({', '.join(face_ids)}));")
    eid += 1
    brep_id = eid
    lines.append(f"#{brep_id} = MANIFOLD_SOLID_BREP('{name}', #{shell_id});")
    eid += 1
    lines.append(f"#{eid} = FACETED_BREP_SHAPE_REPRESENTATION('{name}', (#{brep_id}), #4);")

    lines.extend([
        "ENDSEC;",
        "END-ISO-10303-21;",
    ])
    return "\n".join(lines) + "\n"


def _build_gltf_binary_payload(triangles: List[Triangle]):
    """Build packed binary vertex position and normal buffers for glTF 2.0."""
    pos_bytes = bytearray()
    norm_bytes = bytearray()
    min_pt = [float("inf"), float("inf"), float("inf")]
    max_pt = [float("-inf"), float("-inf"), float("-inf")]

    vertex_count = len(triangles) * 3
    for tri in triangles:
        n = tri.normal
        for v in (tri.a, tri.b, tri.c):
            pos_bytes.extend(struct.pack("<3f", v.x, v.y, v.z))
            norm_bytes.extend(struct.pack("<3f", n.x, n.y, n.z))
            min_pt[0] = min(min_pt[0], v.x)
            min_pt[1] = min(min_pt[1], v.y)
            min_pt[2] = min(min_pt[2], v.z)
            max_pt[0] = max(max_pt[0], v.x)
            max_pt[1] = max(max_pt[1], v.y)
            max_pt[2] = max(max_pt[2], v.z)

    combined = bytes(pos_bytes + norm_bytes)
    return combined, len(pos_bytes), len(norm_bytes), vertex_count, min_pt, max_pt


def export_gltf(triangles: List[Triangle], name: str = "forge_duct") -> str:
    """Export standard glTF 2.0 JSON with embedded base64 data URI buffer."""
    combined, pos_len, norm_len, vertex_count, min_pt, max_pt = _build_gltf_binary_payload(triangles)
    b64_uri = "data:application/octet-stream;base64," + base64.b64encode(combined).decode("ascii")

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "FORGE CAD Engine v0.2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "mode": 4}],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": vertex_count,
                "type": "VEC3",
                "min": [round(x, 4) for x in min_pt],
                "max": [round(x, 4) for x in max_pt],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": vertex_count,
                "type": "VEC3",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": norm_len, "target": 34962},
        ],
        "buffers": [{"uri": b64_uri, "byteLength": len(combined)}],
    }
    return json.dumps(gltf_dict, indent=2) + "\n"


def export_glb(triangles: List[Triangle], name: str = "forge_duct") -> bytes:
    """Export binary glTF 2.0 (.glb) container."""
    combined, pos_len, norm_len, vertex_count, min_pt, max_pt = _build_gltf_binary_payload(triangles)

    gltf_dict = {
        "asset": {"version": "2.0", "generator": "FORGE CAD Engine v0.2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "mode": 4}],
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
                "min": [round(x, 4) for x in min_pt],
                "max": [round(x, 4) for x in max_pt],
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": norm_len, "target": 34962},
        ],
        "buffers": [{"byteLength": len(combined)}],
    }
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    # Pad JSON chunk to 4-byte boundary with spaces (0x20)
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad

    # Pad BIN chunk to 4-byte boundary with zeros (0x00)
    bin_pad = (4 - (len(combined) % 4)) % 4
    bin_bytes = combined + (b"\x00" * bin_pad)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    out = io.BytesIO()
    # 12-byte GLB header: magic 'glTF' (0x46546C67), version 2, total_length
    out.write(struct.pack("<III", 0x46546C67, 2, total_length))
    # JSON chunk: length, type 'JSON' (0x4E4F534A), data
    out.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    out.write(json_bytes)
    # BIN chunk: length, type 'BIN\0' (0x004E4942), data
    out.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
    out.write(bin_bytes)

    return out.getvalue()
