"""Automated OpenFOAM case generator for turbulent Navier-Stokes CFD simulations."""
from __future__ import annotations
from pathlib import Path
from forge.core.models import DuctSpec, PhysicsConditions
from forge.cad.exporters import export_ascii_stl
from forge.physics.fluid_domain import extract_fluid_domain


def generate_openfoam_case(spec: DuctSpec, conditions: PhysicsConditions, output_dir: Path) -> Path:
    """Generate a complete, mesher-ready OpenFOAM simpleFoam case directory."""
    case_dir = Path(output_dir)
    sys_dir = case_dir / "system"
    zero_dir = case_dir / "0"
    const_dir = case_dir / "constant"
    tri_dir = const_dir / "triSurface"

    for d in (sys_dir, zero_dir, const_dir, tri_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Export fluid domain STLs
    patches = extract_fluid_domain(spec)
    all_fluid_tris = []
    for patch_name, p_tris in patches.items():
        all_fluid_tris.extend(p_tris)
        stl_content = export_ascii_stl(p_tris, solid_name=f"patch_{patch_name}")
        (tri_dir / f"{patch_name}.stl").write_text(stl_content, encoding="ascii")

    (tri_dir / "duct_fluid.stl").write_text(export_ascii_stl(all_fluid_tris, solid_name="duct_fluid"), encoding="ascii")

    # Bounding box for background hex mesh (with 10mm margin around geometry)
    all_pts = [p for t in all_fluid_tris for p in (t.a, t.b, t.c)]
    bx_min = min(p.x for p in all_pts) - 10.0
    bx_max = max(p.x for p in all_pts) + 10.0
    by_min = min(p.y for p in all_pts) - 10.0
    by_max = max(p.y for p in all_pts) + 10.0
    bz_min = min(p.z for p in all_pts) - 5.0
    bz_max = max(p.z for p in all_pts) + 5.0

    # Internal point guaranteed to be inside the fluid domain (midpoint along S-curve centerline)
    mid_x = 0.5 * spec.offset_x
    mid_y = 0.5 * spec.offset_y
    mid_z = 0.5 * spec.length

    # 2. Flow velocity calculation
    q_m3_s = conditions.volumetric_flow_cfm * 0.000471947443
    in_area_m2 = max(1e-4, ((spec.inlet_width - 2 * spec.wall) * (spec.inlet_height - 2 * spec.wall)) / 1e6)
    u_inlet = q_m3_s / in_area_m2

    # Turbulence parameters for inlet (5% intensity)
    i_turb = 0.05
    k_inlet = 1.5 * (u_inlet * i_turb) ** 2
    l_turb = 0.07 * (min(spec.inlet_width, spec.inlet_height) / 1000.0)
    omega_inlet = (k_inlet ** 0.5) / (0.09 ** 0.25 * l_turb)

    # system/blockMeshDict
    (sys_dir / "blockMeshDict").write_text(f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      blockMeshDict;
}}

scale   0.001;

vertices
(
    ({bx_min:.2f} {by_min:.2f} {bz_min:.2f})
    ({bx_max:.2f} {by_min:.2f} {bz_min:.2f})
    ({bx_max:.2f} {by_max:.2f} {bz_min:.2f})
    ({bx_min:.2f} {by_max:.2f} {bz_min:.2f})
    ({bx_min:.2f} {by_min:.2f} {bz_max:.2f})
    ({bx_max:.2f} {by_min:.2f} {bz_max:.2f})
    ({bx_max:.2f} {by_max:.2f} {bz_max:.2f})
    ({bx_min:.2f} {by_max:.2f} {bz_max:.2f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (24 24 32) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    background
    {{
        type patch;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (2 3 7 6)
            (1 2 6 5)
            (3 0 4 7)
        );
    }}
);
""", encoding="utf-8")

    # system/snappyHexMeshDict
    (sys_dir / "snappyHexMeshDict").write_text(f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    inlet.stl   {{ type triSurfaceMesh; name inlet; }}
    outlet.stl  {{ type triSurfaceMesh; name outlet; }}
    walls.stl   {{ type triSurfaceMesh; name walls; }}
}};

castellatedMeshControls
{{
    maxLocalCells       100000;
    maxGlobalCells      500000;
    minRefinementCells  10;
    nCellsBetweenLevels 3;
    features            ();
    refinementSurfaces
    {{
        inlet  {{ level (1 1); patchInfo {{ type patch; }} }}
        outlet {{ level (1 1); patchInfo {{ type patch; }} }}
        walls  {{ level (2 2); patchInfo {{ type wall; }} }}
    }}
    resolveFeatureAngle 30;
    refinementRegions   {{}}
    locationInMesh      ({mid_x * 0.001:.5f} {mid_y * 0.001:.5f} {mid_z * 0.001:.5f});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      30;
    nRelaxIter      5;
}}

addLayersControls
{{
    relativeSizes       true;
    layers
    {{
        walls
        {{
            nSurfaceLayers 3;
        }}
    }}
    expansionRatio      1.2;
    finalLayerThickness 0.3;
    minThickness        0.1;
    nGrow               0;
    featureAngle        60;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
}}

mergeTolerance 1e-6;
""", encoding="utf-8")

    # system/controlDict
    (sys_dir / "controlDict").write_text("""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      2;
writeFormat     ascii;
writePrecision  7;
runTimeModifiable true;
""", encoding="utf-8")

    # system/fvSchemes
    (sys_dir / "fvSchemes").write_text("""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
""", encoding="utf-8")

    # system/fvSolution
    (sys_dir / "fvSolution").write_text("""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent               yes;
    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|omega)"     1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.7;
        k               0.7;
        omega           0.7;
    }
    fields
    {
        p               0.3;
    }
}
""", encoding="utf-8")

    # 0/U
    (zero_dir / "U").write_text(f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 {u_inlet:.4f});

boundaryField
{{
    inlet
    {{
        type            surfaceNormalFixedValue;
        refValue        uniform -{u_inlet:.4f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
    vanes
    {{
        type            noSlip;
    }}
}}
""", encoding="utf-8")

    # 0/p
    (zero_dir / "p").write_text("""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
    vanes
    {
        type            zeroGradient;
    }
}
""", encoding="utf-8")

    # 0/k
    (zero_dir / "k").write_text(f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      k;
}}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_inlet:.6f};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {k_inlet:.6f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.6f};
    }}
    vanes
    {{
        type            kqRWallFunction;
        value           uniform {k_inlet:.6f};
    }}
}}
""", encoding="utf-8")

    # 0/omega
    (zero_dir / "omega").write_text(f"""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      omega;
}}

dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_inlet:.4f};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {omega_inlet:.4f};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.4f};
    }}
    vanes
    {{
        type            omegaWallFunction;
        value           uniform {omega_inlet:.4f};
    }}
}}
""", encoding="utf-8")

    # 0/nut
    (zero_dir / "nut").write_text("""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nut;
}

dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            calculated;
        value           uniform 0;
    }
    outlet
    {
        type            calculated;
        value           uniform 0;
    }
    walls
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
    vanes
    {
        type            nutkWallFunction;
        value           uniform 0;
    }
}
""", encoding="utf-8")

    # constant/transportProperties
    (const_dir / "transportProperties").write_text("""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1.5e-05;
""", encoding="utf-8")

    # constant/turbulenceProperties
    (const_dir / "turbulenceProperties").write_text("""/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""", encoding="utf-8")

    # Allrun executable script
    (case_dir / "Allrun").write_text("""#!/bin/sh
cd "${0%/*}" || exit 1
blockMesh
snappyHexMesh -overwrite
checkMesh
simpleFoam
""", encoding="utf-8")

    return case_dir
