"""Open JSON-Schema and portable surrogate specification exporter."""

from __future__ import annotations

from typing import Any, Dict

from forge.cad.honeycomb import HoneycombStatorSpec
from forge.cad.manifold import ManifoldSpec
from forge.cad.tpms import TPMSSpec, TPMSThermoHydraulicReport
from forge.cloud.rack_twin import RackTwinROIReport
from forge.compliance.dossier import ComplianceDossier
from forge.compliance.provenance import HardwareProvenancePassport
from forge.core.models import DFMReport, DuctSpec, PhysicsConditions, PhysicsMetrics
from forge.opt.differentiable_adjoint import AdjointCoDesignResult
from forge.physics.heatsink import CoupledHeatsinkReport, HeatsinkSpec
from forge.physics.structural_modal import ModalResonanceReport
from forge.physics.transient_thermal import TransientThermalReport


def export_all_json_schemas() -> Dict[str, Any]:
    """Export Draft 2020-12 JSON Schemas for all 16 primary FORGE domain models for cross-language interoperability."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "platform": "FORGE",
        "schemas": {
            "DuctSpec": DuctSpec.model_json_schema(),
            "PhysicsConditions": PhysicsConditions.model_json_schema(),
            "PhysicsMetrics": PhysicsMetrics.model_json_schema(),
            "DFMReport": DFMReport.model_json_schema(),
            "HeatsinkSpec": HeatsinkSpec.model_json_schema(),
            "CoupledHeatsinkReport": CoupledHeatsinkReport.model_json_schema(),
            "ManifoldSpec": ManifoldSpec.model_json_schema(),
            "HoneycombStatorSpec": HoneycombStatorSpec.model_json_schema(),
            "ComplianceDossier": ComplianceDossier.model_json_schema(),
            "TPMSSpec": TPMSSpec.model_json_schema(),
            "TPMSThermoHydraulicReport": TPMSThermoHydraulicReport.model_json_schema(),
            "AdjointCoDesignResult": AdjointCoDesignResult.model_json_schema(),
            "HardwareProvenancePassport": HardwareProvenancePassport.model_json_schema(),
            "ModalResonanceReport": ModalResonanceReport.model_json_schema(),
            "TransientThermalReport": TransientThermalReport.model_json_schema(),
            "RackTwinROIReport": RackTwinROIReport.model_json_schema(),
        },
    }
