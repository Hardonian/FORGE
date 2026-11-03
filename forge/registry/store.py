"""SQLite-backed provenance registry for evaluated designs, Pareto fronts, and Sim2Real telemetry."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from forge.core.hashes import canonical_json, sha256_hash
from forge.core.models import DFMReport, DuctSpec, PhysicsMetrics


class ForgeRegistry:
    """Zero-dependency SQLite lineage and experiment store for FORGE hardware designs."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS designs (
                spec_hash TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                dfm_json TEXT,
                dp_pa REAL NOT NULL,
                uniformity REAL NOT NULL,
                mass_g REAL NOT NULL,
                tag TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry_runs (
                run_id TEXT PRIMARY KEY,
                spec_hash TEXT NOT NULL,
                telemetry_json TEXT NOT NULL,
                calibrated_roughness_mm REAL NOT NULL,
                calibrated_k_mult REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def record_evaluation(
        self,
        spec: DuctSpec,
        metrics: PhysicsMetrics,
        dfm: Optional[DFMReport] = None,
        tag: str = "default",
    ) -> str:
        """Persist a design evaluation and return its deterministic SHA-256 spec_hash."""
        spec_hash = sha256_hash(canonical_json(spec.to_canonical_dict()))
        spec_json = spec.model_dump_json()
        metrics_json = metrics.model_dump_json()
        dfm_json = dfm.model_dump_json() if dfm else None
        mass_g = dfm.estimated_mass_grams if dfm else 0.0
        now_iso = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO designs (spec_hash, spec_json, metrics_json, dfm_json, dp_pa, uniformity, mass_g, tag, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(spec_hash) DO UPDATE SET
                metrics_json = excluded.metrics_json,
                dfm_json = COALESCE(excluded.dfm_json, designs.dfm_json),
                dp_pa = excluded.dp_pa,
                uniformity = excluded.uniformity,
                mass_g = CASE WHEN excluded.mass_g > 0 THEN excluded.mass_g ELSE designs.mass_g END,
                tag = excluded.tag
            """,
            (
                spec_hash,
                spec_json,
                metrics_json,
                dfm_json,
                float(metrics.total_pressure_drop_pa),
                float(metrics.flow_uniformity_index),
                float(mass_g),
                tag,
                now_iso,
            ),
        )
        self._conn.commit()
        return spec_hash

    def get_design(self, spec_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve a recorded design and its metrics by SHA-256 hash."""
        row = self._conn.execute("SELECT * FROM designs WHERE spec_hash = ?", (spec_hash,)).fetchone()
        if not row:
            return None
        return {
            "spec_hash": row["spec_hash"],
            "spec": json.loads(row["spec_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "dfm": json.loads(row["dfm_json"]) if row["dfm_json"] else None,
            "dp_pa": row["dp_pa"],
            "uniformity": row["uniformity"],
            "mass_g": row["mass_g"],
            "tag": row["tag"],
            "created_at": row["created_at"],
        }

    def list_top_designs(
        self,
        limit: int = 10,
        max_dp_pa: Optional[float] = None,
        min_uniformity: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query top recorded designs ordered by lowest pressure drop and highest uniformity."""
        clauses: List[str] = []
        params: List[Any] = []
        if max_dp_pa is not None:
            clauses.append("dp_pa <= ?")
            params.append(float(max_dp_pa))
        if min_uniformity is not None:
            clauses.append("uniformity >= ?")
            params.append(float(min_uniformity))
        if tag is not None:
            clauses.append("tag = ?")
            params.append(tag)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM designs {where_sql} ORDER BY dp_pa ASC, uniformity DESC LIMIT ?"
        params.append(int(limit))

        rows = self._conn.execute(sql, params).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append({
                "spec_hash": row["spec_hash"],
                "spec": json.loads(row["spec_json"]),
                "metrics": json.loads(row["metrics_json"]),
                "dp_pa": row["dp_pa"],
                "uniformity": row["uniformity"],
                "mass_g": row["mass_g"],
                "tag": row["tag"],
            })
        return results

    def export_surrogate_training_specs(self, limit: int = 500) -> List[DuctSpec]:
        """Export stored DuctSpecs to train or fine-tune the DuctFlowSurrogate model."""
        rows = self._conn.execute("SELECT spec_json FROM designs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [DuctSpec(**json.loads(r["spec_json"])) for r in rows]

    def record_telemetry_calibration(
        self,
        spec_hash: str,
        run_id: str,
        telemetry_dict: Dict[str, Any],
        roughness_mm: float,
        k_mult: float,
    ) -> None:
        """Store a physical test-bench Sim2Real calibration run linked to a design spec_hash."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO telemetry_runs
            (run_id, spec_hash, telemetry_json, calibrated_roughness_mm, calibrated_k_mult, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                spec_hash,
                json.dumps(telemetry_dict, sort_keys=True),
                float(roughness_mm),
                float(k_mult),
                now_iso,
            ),
        )
        self._conn.commit()

    def count_designs(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM designs").fetchone()
        return int(row["cnt"]) if row else 0

    def close(self) -> None:
        self._conn.close()
