"""Open JSON-Schema and portable surrogate specification exporter."""
from __future__ import annotations
from typing import Any, Dict
from forge.cad.honeycomb import HoneycombStatorSpec
from forge.cad.manifold import ManifoldSpec
from forge.compliance.dossier import ComplianceDossier
from forge.core.models import DFMReport, DuctSpec, PhysicsConditions, PhysicsMetrics
from forge.physics.heatsink import CoupledHeatsinkReport, HeatsinkSpec


def export_all_json_schemas() -> Dict[str, Any]:
    """Export Draft 2020-12 JSON Schemas for all primary FORGE domain models for cross-language interoperability."""
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
        },
    }

