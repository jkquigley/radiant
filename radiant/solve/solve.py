__all__ = [
    "CollocationSolver",
    "GalerkinSolver",
    "InterpolationSolver",
    "MultilevelSolver",
]

from .collocation import CollocationSolver
from .galerkin import GalerkinSolver
from .interpolate import InterpolationSolver
from .multilevel import MultilevelSolver
