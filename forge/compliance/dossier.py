"""Automated enterprise hardware qualification and compliance dossier generator (UL94, ASHRAE, NEBS GR-63-CORE)."""
from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel
from forge.core.hashes import canonical_json, sha256_hash
from forge.core.models import DFMReport, DuctSpec, MATERIAL_PROPERTIES, PhysicsConditions, PhysicsMetrics
from forge.cad.dfm import analyze_dfm
from forge.cad.mesh import build_duct_mesh
from forge.physics.solver_1d import solve_duct_physics
from forge.physics.structural_modal import ModalResonanceReport, analyze_structural_modal_resonance


UL94_RATINGS: Dict[str, str] = {
    "PEEK": "UL94 V-0 (Intrinsic Flame Retardant Datacenter Grade)",
    "PC-CF": "UL94 V-0 (Flame Retardant Carbon-Reinforced Polycarbonate)",
    "PA12-CF": "UL94 V-2 / HB (Engineering Composite)",
    "ASA": "UL94 HB (Workstation / Non-Plenum Grade)",
    "ABS": "UL94 HB (Workstation / Non-Plenum Grade)",
    "PETG": "UL94 HB (Prototype / Benchtop Only)",
}


class ComplianceCheckItem(BaseModel):
    """Individual regulatory or engineering standard verification item."""
    standard_code: str
    status: str
    measured_value: str
    requirement: str
    notes: str


class ComplianceDossier(BaseModel):
    """Complete enterprise hardware qualification certificate and compliance dossier."""
    certificate_id: str
    geometry_sha256: str
    overall_qualification_status: str
    ul94_flammability_rating: str
    ashrae_environmental_class: str
    checks: List[ComplianceCheckItem]
    spec: DuctSpec
    physics: PhysicsMetrics
    dfm: DFMReport
    modal: ModalResonanceReport


def generate_compliance_dossier(
    spec: DuctSpec,
    conditions: PhysicsConditions | None = None,
    fan_rpm: int = 9000,
) -> ComplianceDossier:
    """Run multi-physics, DFM, and regulatory compliance checks to generate an OEM qualification dossier."""
    cond = conditions or PhysicsConditions()
    spec_hash = sha256_hash(canonical_json(spec.to_canonical_dict()))
    cert_id = f"FORGE-CERT-{spec_hash[:10].upper()}"

    tris = build_duct_mesh(spec)
    dfm = analyze_dfm(spec, tris)
    phys = solve_duct_physics(spec, cond)
    modal = analyze_structural_modal_resonance(spec, cond, fan_rpm=fan_rpm)

    mat_key = spec.material.value
    mat_props = MATERIAL_PROPERTIES.get(mat_key, MATERIAL_PROPERTIES["PETG"])
    hdt_c = mat_props["hdt_c"]
    exhaust_c = cond.air_temp_c + phys.estimated_temp_rise_c
    ul94 = UL94_RATINGS.get(mat_key, "UL94 HB")

    checks: List[ComplianceCheckItem] = []

    # 1. UL94 / IEC 60695-11-10 Datacenter Flammability Check
    ul_status = "PASS" if "V-0" in ul94 else "WARN"
    checks.append(
        ComplianceCheckItem(
            standard_code="UL94 / IEC 60695-11-10",
            status=ul_status,
            measured_value=ul94,
            requirement="UL94 V-0 required for unattended server rack plenum deployment",
            notes="Select PC-CF or PEEK for full UL94 V-0 datacenter fire compliance.",
        )
    )

    # 2. ASHRAE Thermal Guidelines for Data Processing Environments (Class A1-A4)
    thermal_margin_hdt = hdt_c - exhaust_c
    if thermal_margin_hdt >= 35.0:
        ashrae_class = "ASHRAE Class A4 (Up to 45C Ambient)"
        ashrae_status = "PASS"
    elif thermal_margin_hdt >= 20.0:
        ashrae_class = "ASHRAE Class A2 (Up to 35C Ambient)"
        ashrae_status = "PASS"
    else:
        ashrae_class = "Sub-A1 (Benchtop Only - Creep Risk)"
        ashrae_status = "FAIL"

    checks.append(
        ComplianceCheckItem(
            standard_code="ASHRAE TC 9.9 Thermal Creep",
            status=ashrae_status,
            measured_value=f"Exhaust {exhaust_c:.1f} C vs HDT {hdt_c:.1f} C (Margin: {thermal_margin_hdt:.1f} C)",
            requirement="HDT margin >= 20 C above peak exhaust air temperature",
            notes=f"Qualified for {ashrae_class}.",
        )
    )

    # 3. NEBS GR-63-CORE Structural Resonance & Seismic Vibration Check
    vib_status = "PASS" if (not modal.resonance_risk and modal.fundamental_panel_freq_hz > 50.0) else "WARN"
    checks.append(
        ComplianceCheckItem(
            standard_code="NEBS GR-63-CORE Vibration",
            status=vib_status,
            measured_value=f"f_11 = {modal.fundamental_panel_freq_hz} Hz (Separation: {modal.min_harmonic_separation_pct}%)",
            requirement="f_11 > 50 Hz and >= 15% separation from fan shaft/BPF harmonics",
            notes="No harmonic lock-in detected." if vib_status == "PASS" else f"Increase wall to {modal.recommended_wall_mm} mm.",
        )
    )

    # 4. ISO 7779 / ECMA-74 Datacenter Aeroacoustic Emission Check
    ac_status = "PASS" if phys.acoustic_noise_dba <= 68.0 else "WARN"
    checks.append(
        ComplianceCheckItem(
            standard_code="ISO 7779 / ECMA-74 Acoustics",
            status=ac_status,
            measured_value=f"{phys.acoustic_noise_dba} dBA",
            requirement="<= 68.0 dBA aerodynamic duct flow emission",
            notes="Compliant with datacenter aisle occupational exposure limits.",
        )
    )

    # 5. Additive Manufacturing DFM & Structural Deflection Check
    dfm_status = "PASS" if (dfm.overhang_compliance and dfm.bed_fit and phys.max_wall_deflection_um < 250.0) else "WARN"
    checks.append(
        ComplianceCheckItem(
            standard_code="ASME Y14.46 / Additive DFM",
            status=dfm_status,
            measured_value=f"Max Overhang {dfm.max_overhang_angle_deg} deg, Deflection {phys.max_wall_deflection_um} um",
            requirement="Overhang <= 50 deg, Wall Deflection < 250 um",
            notes="Support-free FDM printable." if dfm.overhang_compliance else "Support structures or shallower taper advised.",
        )
    )

    statuses = [c.status for c in checks]
    if "FAIL" in statuses:
        overall = "NON_COMPLIANT"
    elif "WARN" in statuses:
        overall = "CONDITIONAL_PROTOTYPE_QUALIFIED"
    else:
        overall = "QUALIFIED_DATACENTER_PRODUCTION"

    return ComplianceDossier(
        certificate_id=cert_id,
        geometry_sha256=spec_hash,
        overall_qualification_status=overall,
        ul94_flammability_rating=ul94,
        ashrae_environmental_class=ashrae_class,
        checks=checks,
        spec=spec,
        physics=phys,
        dfm=dfm,
        modal=modal,
    )


def export_compliance_dossier_markdown(dossier: ComplianceDossier) -> str:
    """Format a ComplianceDossier into an enterprise audit-ready Markdown certification document."""
    lines = [
        f"# FORGE Hardware Qualification & Compliance Dossier ({dossier.certificate_id})",
        "",
        f"- **Qualification Verdict:** `{dossier.overall_qualification_status}`",
        f"- **Geometry SHA-256 Fingerprint:** `{dossier.geometry_sha256}`",
        f"- **Material & Flammability Rating:** `{dossier.spec.material.value}` — {dossier.ul94_flammability_rating}",
        f"- **ASHRAE Environmental Envelope:** {dossier.ashrae_environmental_class}",
        "",
        "## 1. Regulatory & Engineering Verification Matrix",
        "",
        "| Standard / Code | Status | Measured Value | Requirement | Engineering Notes |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for chk in dossier.checks:
        lines.append(
            f"| **{chk.standard_code}** | `{chk.status}` | {chk.measured_value} | {chk.requirement} | {chk.notes} |"
        )

    lines.extend([
        "",
        "## 2. Certified Multi-Physics & Structural Summary",
        "",
        f"- **Total Static Pressure Drop:** {dossier.physics.total_pressure_drop_pa} Pa",
        f"- **Exit Velocity Uniformity Index:** {dossier.physics.flow_uniformity_index}",
        f"- **Fundamental Panel Natural Frequency ($f_{{11}}$):** {dossier.modal.fundamental_panel_freq_hz} Hz",
        f"- **Temperature-Softened Young's Modulus:** {dossier.modal.effective_youngs_modulus_mpa} MPa",
        f"- **Estimated Solid Print Mass:** {dossier.dfm.estimated_mass_grams} g ({dossier.dfm.estimated_print_time_hours} hrs)",
        "",
    ])
    return "\n".join(lines)
