"""Navier-Stokes Brinkman-Darcy fluid topology optimization engine."""
from forge.topopt.darcy_solver import (
    DarcyTopOptResult,
    run_fluid_topology_optimization,
)

__all__ = ["DarcyTopOptResult", "run_fluid_topology_optimization"]

