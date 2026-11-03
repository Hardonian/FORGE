"""Automated SU2 CFD case and configuration generator."""
from __future__ import annotations
from pathlib import Path
from forge.core.models import DuctSpec, PhysicsConditions


def generate_su2_config(spec: DuctSpec, conditions: PhysicsConditions, output_path: Path) -> Path:
    """Generate production-ready SU2 configuration file for compressible/incompressible RANS."""
    q_m3_s = conditions.volumetric_flow_cfm * 0.000471947443
    in_area_m2 = max(1e-4, ((spec.inlet_width - 2 * spec.wall) * (spec.inlet_height - 2 * spec.wall)) / 1e6)
    v_inlet = q_m3_s / in_area_m2

    cfg_text = f"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
% SU2 Configuration File - FORGE AI Cooling Duct Incompressible RANS           %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SOLVER= INC_RANS
KIND_TURB_MODEL= SST
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% -------------------- INCOMPRESSIBLE FLOW PROPERTIES ------------------------ %
INC_DENSITY_MODEL= CONSTANT
INC_DENSITY_INIT= {conditions.air_density_kg_m3:.3f}
INC_VELOCITY_INIT= ( 0.0, 0.0, {v_inlet:.3f} )
VISCOSITY_MODEL= CONSTANT_VISCOSITY
MU_CONSTANT= {conditions.air_viscosity_pa_s:.6e}

% ------------------------- BOUNDARY CONDITIONS ------------------------------ %
MARKER_INLET= ( inlet, {conditions.air_temp_c + 273.15:.2f}, {v_inlet:.3f}, 0.0, 0.0, 1.0 )
MARKER_OUTLET= ( outlet, 0.0 )
MARKER_ISOTHERMAL= ( walls, {conditions.air_temp_c + 273.15 + 10.0:.2f} )
MARKER_PLOTTING= ( walls, inlet, outlet )
MARKER_MONITORING= ( outlet )

% ------------------------- CONVERGENCE CONTROLS ----------------------------- %
ITER= 1000
CONV_FIELD= RMS_PRESSURE
CONV_RESIDUAL_MINVAL= -6
CONV_STARTITER= 10

% ------------------------- NUMERICAL SCHEMES -------------------------------- %
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 10.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.1, 2.0, 10.0, 1e5 )

% ------------------------- OUTPUT SETTINGS ---------------------------------- %
OUTPUT_FILES= ( RESTART, PARAVIEW, SURFACE_PARAVIEW )
VOLUME_OUTPUT= ( COORDINATES, SOLUTION, PRIMITIVE )
CONV_FILENAME= history
BREAKDOWN_FILENAME= forces_breakdown.dat
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cfg_text, encoding="utf-8")
    return output_path

