"""Cryptographic Hardware Bill of Materials (HBOM) & SLSA Supply-Chain Provenance.

Generates a tamper-evident Digital Manufacturing Passport (`HardwareProvenancePassport`)
containing:
- Complete mechanical & additive Hardware Bill of Materials (HBOM) with fasteners,
  brass heat-set inserts, filament spool mass, and desiccant drying schedule.
- SHA-256 cryptographic digests of the canonical DuctSpec, 3D binary STL mesh,
  and enterprise compliance qualification dossier.
- HMAC-SHA256 cryptographic attestation signature with deterministic verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac

from pydantic import BaseModel, Field

from forge.cad.dfm import analyze_dfm
from forge.cad.exporters import export_binary_stl
from forge.cad.flanges import FAN_STANDARDS
from forge.cad.mesh import build_duct_mesh
from forge.compliance.dossier import generate_compliance_dossier
from forge.core.hashes import canonical_json, sha256_hash
from forge.core.models import DuctSpec, PhysicsConditions


class HardwareBOMItem(BaseModel):
    """Single line item in the Hardware Bill of Materials (HBOM)."""

    part_number: str
    category: str
    description: str
    quantity: float
    unit: str
    material_or_standard: str
    unit_cost_usd: float
    total_cost_usd: float


class DryingAndProcessSpec(BaseModel):
    """Additive manufacturing filament drying and chamber conditioning protocol."""

    material: str
    drying_temp_c: int
    drying_duration_hours: int
    max_moisture_ppm: int
    nozzle_temp_c: int
    chamber_temp_c: int
    annealing_protocol: str


class HardwareProvenancePassport(BaseModel):
    """SLSA-style cryptographic Hardware Bill of Materials & Provenance Attestation."""

    schema_version: str = "https://forge.dev/schemas/v1.0.0/HardwareProvenancePassport.json"
    passport_id: str
    generated_at_utc: str
    spec_sha256: str
    mesh_stl_sha256: str
    compliance_dossier_sha256: str
    overall_qualification_status: str
    drying_and_process: DryingAndProcessSpec
    bom_items: list[HardwareBOMItem] = Field(default_factory=list)
    total_estimated_unit_cost_usd: float
    attestation_payload_sha256: str
    attestation_signature_hmac_sha256: str


DRYING_PROTOCOLS: dict[str, DryingAndProcessSpec] = {
    "PC-CF": DryingAndProcessSpec(
        material="PC-CF",
        drying_temp_c=85,
        drying_duration_hours=8,
        max_moisture_ppm=400,
        nozzle_temp_c=290,
        chamber_temp_c=60,
        annealing_protocol="Anneal at 110 C for 2 hours, ramp down at 15 C/hr",
    ),
    "PA12-CF": DryingAndProcessSpec(
        material="PA12-CF",
        drying_temp_c=80,
        drying_duration_hours=10,
        max_moisture_ppm=500,
        nozzle_temp_c=285,
        chamber_temp_c=55,
        annealing_protocol="Anneal at 100 C for 4 hours in dry convection oven",
    ),
    "PEEK": DryingAndProcessSpec(
        material="PEEK",
        drying_temp_c=150,
        drying_duration_hours=5,
        max_moisture_ppm=200,
        nozzle_temp_c=405,
        chamber_temp_c=120,
        annealing_protocol="Step-crystallize: 150 C (1h) -> 200 C (2h) -> slow cool",
    ),
    "ASA": DryingAndProcessSpec(
        material="ASA",
        drying_temp_c=70,
        drying_duration_hours=4,
        max_moisture_ppm=800,
        nozzle_temp_c=260,
        chamber_temp_c=50,
        annealing_protocol="No post-print annealing required",
    ),
    "ABS": DryingAndProcessSpec(
        material="ABS",
        drying_temp_c=70,
        drying_duration_hours=4,
        max_moisture_ppm=800,
        nozzle_temp_c=255,
        chamber_temp_c=50,
        annealing_protocol="No post-print annealing required",
    ),
    "PETG": DryingAndProcessSpec(
        material="PETG",
        drying_temp_c=65,
        drying_duration_hours=4,
        max_moisture_ppm=800,
        nozzle_temp_c=245,
        chamber_temp_c=35,
        annealing_protocol="No post-print annealing required",
    ),
}

FILAMENT_COST_PER_GRAM_USD: dict[str, float] = {
    "PC-CF": 0.095,
    "PA12-CF": 0.125,
    "PEEK": 0.650,
    "ASA": 0.035,
    "ABS": 0.032,
    "PETG": 0.028,
}


def generate_hardware_provenance_passport(
    spec: DuctSpec,
    conditions: PhysicsConditions | None = None,
    signing_key: str = "forge-enterprise-oem-root-key",
) -> HardwareProvenancePassport:
    """Generate a cryptographic HBOM and SLSA provenance passport for a duct design."""
    cond = conditions or PhysicsConditions()
    spec_hash = sha256_hash(spec.to_canonical_dict())
    tris = build_duct_mesh(spec)
    dfm = analyze_dfm(spec, tris)
    dossier = generate_compliance_dossier(spec, conditions=cond)

    stl_bytes = export_binary_stl(tris)
    mesh_sha256 = hashlib.sha256(stl_bytes).hexdigest()
    dossier_sha256 = sha256_hash(dossier.model_dump(mode="json"))

    mat_key = spec.material.value
    bom: list[HardwareBOMItem] = []

    # 1. Additive Filament with 12% purge/support/brim allowance
    gross_mass_g = round(dfm.estimated_mass_grams * 1.12, 2)
    cost_per_g = FILAMENT_COST_PER_GRAM_USD.get(mat_key, 0.085)
    filament_cost = round(gross_mass_g * cost_per_g, 2)
    bom.append(
        HardwareBOMItem(
            part_number=f"FRG-FIL-{mat_key}",
            category="Additive Raw Material",
            description=f"{mat_key} Engineering Filament (incl. 12% purge allowance)",
            quantity=gross_mass_g,
            unit="grams",
            material_or_standard=f"ISO 1043 {mat_key}",
            unit_cost_usd=cost_per_g,
            total_cost_usd=filament_cost,
        )
    )

    # 2. Machine Time (Industrial FFF Depreciation & Energy @ $3.50/hr)
    machine_rate = 3.50
    machine_cost = round(dfm.estimated_print_time_hours * machine_rate, 2)
    bom.append(
        HardwareBOMItem(
            part_number="FRG-MFG-FFF",
            category="Manufacturing Machine Time",
            description="High-Temp FFF Print Cycle (256x256x256 Build Volume)",
            quantity=dfm.estimated_print_time_hours,
            unit="hours",
            material_or_standard="ASME Y14.46 Additive Process",
            unit_cost_usd=machine_rate,
            total_cost_usd=machine_cost,
        )
    )

    # 3. Fasteners & Heat-Set Inserts when flange is present
    if spec.flange_thickness > 0 or spec.fan_standard is not None:
        fan_key = spec.fan_standard.value if spec.fan_standard else "120mm"
        f_std = FAN_STANDARDS.get(fan_key, FAN_STANDARDS["120mm"])
        thread_code = f_std.screw_type
        bolt_count = 4
        bom.append(
            HardwareBOMItem(
                part_number=f"ISO4762-{thread_code}x16-A2",
                category="Mechanical Fasteners",
                description=f"{thread_code}x16mm Socket Head Cap Screw, Stainless A2-70",
                quantity=float(bolt_count),
                unit="pcs",
                material_or_standard="ISO 4762 / DIN 912",
                unit_cost_usd=0.14,
                total_cost_usd=round(bolt_count * 0.14, 2),
            )
        )
        bom.append(
            HardwareBOMItem(
                part_number=f"FRG-INS-{thread_code}-BRASS",
                category="Threaded Inserts",
                description=f"{thread_code}x5.7mm Knurled Brass Heat-Set Insert (OD {f_std.brass_insert_od_mm}mm)",
                quantity=float(bolt_count),
                unit="pcs",
                material_or_standard="C36000 Free-Machining Brass",
                unit_cost_usd=0.22,
                total_cost_usd=round(bolt_count * 0.22, 2),
            )
        )

    total_cost = round(sum(item.total_cost_usd for item in bom), 2)
    drying_spec = DRYING_PROTOCOLS.get(mat_key, DRYING_PROTOCOLS["PC-CF"])

    core_payload = {
        "spec_sha256": spec_hash,
        "mesh_stl_sha256": mesh_sha256,
        "compliance_dossier_sha256": dossier_sha256,
        "overall_qualification_status": dossier.overall_qualification_status,
        "total_estimated_unit_cost_usd": total_cost,
        "bom_part_numbers": [b.part_number for b in bom],
    }
    payload_bytes = canonical_json(core_payload).encode("utf-8")
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    signature = hmac.new(
        signing_key.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()

    return HardwareProvenancePassport(
        passport_id=f"FRG-PASS-{spec_hash[:12].upper()}",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        spec_sha256=spec_hash,
        mesh_stl_sha256=mesh_sha256,
        compliance_dossier_sha256=dossier_sha256,
        overall_qualification_status=dossier.overall_qualification_status,
        drying_and_process=drying_spec,
        bom_items=bom,
        total_estimated_unit_cost_usd=total_cost,
        attestation_payload_sha256=payload_sha256,
        attestation_signature_hmac_sha256=signature,
    )


def verify_hardware_provenance(
    passport: HardwareProvenancePassport,
    signing_key: str = "forge-enterprise-oem-root-key",
) -> bool:
    """Cryptographically verify that a HardwareProvenancePassport has not been tampered with."""
    core_payload = {
        "spec_sha256": passport.spec_sha256,
        "mesh_stl_sha256": passport.mesh_stl_sha256,
        "compliance_dossier_sha256": passport.compliance_dossier_sha256,
        "overall_qualification_status": passport.overall_qualification_status,
        "total_estimated_unit_cost_usd": passport.total_estimated_unit_cost_usd,
        "bom_part_numbers": [b.part_number for b in passport.bom_items],
    }
    payload_bytes = canonical_json(core_payload).encode("utf-8")
    expected_payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    if not hmac.compare_digest(expected_payload_sha, passport.attestation_payload_sha256):
        return False

    expected_sig = hmac.new(
        signing_key.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected_sig, passport.attestation_signature_hmac_sha256)
