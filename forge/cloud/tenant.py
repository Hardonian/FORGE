"""Multi-tenant SaaS entitlement, feature gating, and usage metering for FORGE Cloud."""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class SubscriptionTier(str, Enum):
    OPEN_CORE = "open_core"
    CLOUD_PRO = "cloud_pro"
    ENTERPRISE_OEM = "enterprise_oem"


TIER_PRICING_USD_MONTHLY: Dict[SubscriptionTier, float] = {
    SubscriptionTier.OPEN_CORE: 0.0,
    SubscriptionTier.CLOUD_PRO: 199.0,
    SubscriptionTier.ENTERPRISE_OEM: 4500.0,
}

TIER_QUOTAS: Dict[SubscriptionTier, int] = {
    SubscriptionTier.OPEN_CORE: 100,
    SubscriptionTier.CLOUD_PRO: 10000,
    SubscriptionTier.ENTERPRISE_OEM: 1000000,
}

TIER_ALLOWED_FEATURES: Dict[SubscriptionTier, List[str]] = {
    SubscriptionTier.OPEN_CORE: [
        "evaluate_1d",
        "export_stl",
        "export_obj",
        "dfm_check",
    ],
    SubscriptionTier.CLOUD_PRO: [
        "evaluate_1d",
        "export_stl",
        "export_obj",
        "dfm_check",
        "export_step",
        "export_glb",
        "export_slicer_3mf",
        "copilot_synthesis",
        "obstacle_routing",
        "nsga2_optimize",
        "bezier_optimize",
        "topopt_darcy",
        "fno_flow_field",
    ],
    SubscriptionTier.ENTERPRISE_OEM: [
        "evaluate_1d",
        "export_stl",
        "export_obj",
        "dfm_check",
        "export_step",
        "export_glb",
        "export_slicer_3mf",
        "copilot_synthesis",
        "obstacle_routing",
        "nsga2_optimize",
        "bezier_optimize",
        "topopt_darcy",
        "fno_flow_field",
        "manifold_network_balance",
        "coupled_cht_heatsink",
        "datacenter_rack_twin",
        "ipmi_telemetry_sync",
    ],
}


class TenantAccount(BaseModel):
    """SaaS tenant organization and usage state."""
    tenant_id: str
    organization: str
    tier: SubscriptionTier = Field(SubscriptionTier.CLOUD_PRO)
    monthly_quota: int = Field(10000)
    used_evaluations: int = Field(0)
    used_topopt_runs: int = Field(0)
    compute_credits_used: float = Field(0.0)


class CloudEntitlementManager:
    """In-memory / registry-compatible SaaS entitlement and usage metering engine."""

    def __init__(self) -> None:
        self._tenants: Dict[str, TenantAccount] = {
            "default-pro": TenantAccount(
                tenant_id="default-pro",
                organization="FORGE Pro Studio",
                tier=SubscriptionTier.CLOUD_PRO,
                monthly_quota=TIER_QUOTAS[SubscriptionTier.CLOUD_PRO],
            ),
            "enterprise-hyperscale": TenantAccount(
                tenant_id="enterprise-hyperscale",
                organization="Hyperscale AI Cluster Corp",
                tier=SubscriptionTier.ENTERPRISE_OEM,
                monthly_quota=TIER_QUOTAS[SubscriptionTier.ENTERPRISE_OEM],
            ),
        }

    def register_tenant(self, tenant_id: str, organization: str, tier: SubscriptionTier) -> TenantAccount:
        acct = TenantAccount(
            tenant_id=tenant_id,
            organization=organization,
            tier=tier,
            monthly_quota=TIER_QUOTAS[tier],
        )
        self._tenants[tenant_id] = acct
        return acct

    def get_tenant(self, tenant_id: str) -> TenantAccount:
        if tenant_id not in self._tenants:
            raise KeyError(f"Unknown tenant_id '{tenant_id}'")
        return self._tenants[tenant_id]

    def verify_feature_access(self, tenant_id: str, feature: str) -> bool:
        acct = self.get_tenant(tenant_id)
        if acct.used_evaluations >= acct.monthly_quota:
            return False
        return feature in TIER_ALLOWED_FEATURES.get(acct.tier, [])

    def record_usage(self, tenant_id: str, feature: str, compute_credits: float = 1.0) -> TenantAccount:
        acct = self.get_tenant(tenant_id)
        acct.used_evaluations += 1
        if feature == "topopt_darcy":
            acct.used_topopt_runs += 1
        acct.compute_credits_used = round(acct.compute_credits_used + compute_credits, 3)
        return acct

    def get_billing_summary(self, tenant_id: str) -> Dict[str, Any]:
        acct = self.get_tenant(tenant_id)
        base_fee = TIER_PRICING_USD_MONTHLY[acct.tier]
        overage_evals = max(0, acct.used_evaluations - acct.monthly_quota)
        overage_usd = overage_evals * 0.05
        return {
            "tenant_id": acct.tenant_id,
            "organization": acct.organization,
            "tier": acct.tier.value,
            "base_monthly_usd": base_fee,
            "used_evaluations": acct.used_evaluations,
            "monthly_quota": acct.monthly_quota,
            "used_topopt_runs": acct.used_topopt_runs,
            "compute_credits_used": acct.compute_credits_used,
            "overage_usd": round(overage_usd, 2),
            "estimated_total_invoice_usd": round(base_fee + overage_usd, 2),
        }
