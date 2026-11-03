"""Engineering report generator with embedded Pareto front charts."""
from __future__ import annotations
from pathlib import Path
from typing import List
from forge.opt.optimizer import CandidateEvaluation


def generate_pareto_svg(candidates: List[CandidateEvaluation], width: int = 600, height: int = 350) -> str:
    """Generate clean inline SVG scatter plot of Pareto Front: Pressure Drop vs Uniformity."""
    if not candidates:
        return "<svg></svg>"

    pad = 50
    p_vals = [c.physics.total_pressure_drop_pa for c in candidates]
    u_vals = [c.physics.flow_uniformity_index for c in candidates]

    min_p, max_p = min(p_vals), max(p_vals)
    min_u, max_u = min(u_vals), max(u_vals)

    dp = max(1.0, max_p - min_p)
    du = max(0.01, max_u - min_u)

    def tx(p: float) -> float:
        return pad + ((p - min_p) / dp) * (width - 2 * pad)

    def ty(u: float) -> float:
        # Uniformity goes up
        return (height - pad) - ((u - min_u) / du) * (height - 2 * pad)

    circles = []
    for c in candidates:
        cx = tx(c.physics.total_pressure_drop_pa)
        cy = ty(c.physics.flow_uniformity_index)
        circles.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="#2563eb" stroke="#1e40af" stroke-width="2">'
            f'<title>L={c.spec.length}mm, Vanes={c.spec.guide_vanes}, dP={c.physics.total_pressure_drop_pa}Pa, Uni={c.physics.flow_uniformity_index}</title>'
            f'</circle>'
        )

    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; font-family:sans-serif;">
  <text x="{width/2}" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">Pareto Optimal Frontier (ΔP vs Flow Uniformity)</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#94a3b8" stroke-width="2" />
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#94a3b8" stroke-width="2" />
  <text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="12" fill="#475569">Total Pressure Drop ΔP (Pa) [Lower is Better]</text>
  <text x="15" y="{height/2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 15 {height/2})">Uniformity γ [Higher is Better]</text>
  <text x="{pad}" y="{height-pad+15}" font-size="10" fill="#64748b">{min_p:.1f}</text>
  <text x="{width-pad}" y="{height-pad+15}" font-size="10" fill="#64748b" text-anchor="end">{max_p:.1f}</text>
  <text x="{pad-10}" y="{height-pad}" font-size="10" fill="#64748b" text-anchor="end">{min_u:.2f}</text>
  <text x="{pad-10}" y="{pad+10}" font-size="10" fill="#64748b" text-anchor="end">{max_u:.2f}</text>
  {' '.join(circles)}
</svg>"""
    return svg


def generate_html_report(candidates: List[CandidateEvaluation], output_path: Path) -> Path:
    """Generate self-contained HTML engineering report."""
    svg_chart = generate_pareto_svg(candidates)
    rows = []
    for i, c in enumerate(candidates, 1):
        rows.append(f"""<tr>
  <td>#{i}</td>
  <td>{c.spec.length} mm</td>
  <td>{c.spec.offset_x:+0.1f}, {c.spec.offset_y:+0.1f} mm</td>
  <td>{c.spec.corner_radius} mm</td>
  <td>{c.spec.guide_vanes}</td>
  <td><strong>{c.physics.total_pressure_drop_pa:.1f} Pa</strong></td>
  <td><strong>{c.physics.flow_uniformity_index:.3f}</strong></td>
  <td>{c.physics.outlet_velocity_m_s:.2f} m/s</td>
  <td>{c.mass_grams:.1f} g</td>
</tr>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FORGE Generative Design Optimization Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px auto; max-width: 900px; color: #1e293b; background: #ffffff; line-height: 1.6; }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
    .chart-container {{ margin: 24px 0; text-align: center; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 24px; font-size: 14px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #f8fafc; font-weight: 600; color: #334155; }}
    tr:hover {{ background: #f1f5f9; }}
    .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background: #dbeafe; color: #1d4ed8; font-weight: 600; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>FORGE Generative Inverse-Design Report</h1>
  <p>Multi-objective NSGA-II Pareto optimization results for AI accelerator transition ducting.</p>

  <div class="chart-container">
    {svg_chart}
  </div>

  <h2>Pareto-Optimal Candidates ({len(candidates)} solutions)</h2>
  <table>
    <thead>
      <tr>
        <th>Candidate</th>
        <th>Length</th>
        <th>Offset (dx, dy)</th>
        <th>Fillet</th>
        <th>Vanes</th>
        <th>Total ΔP</th>
        <th>Uniformity γ</th>
        <th>Exit Velocity</th>
        <th>Est. Mass</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
