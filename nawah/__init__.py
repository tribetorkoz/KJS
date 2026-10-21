"""NAWAH — a layer over any model.

It accumulates, verifies, routes and transfers skills.
"""

from . import bench, consolidate, curriculum, theory
from .agent import BaselineAgent, NawahAgent, Result
from .backend import (
    ENDPOINTS,
    PRICES,
    AnthropicBackend,
    Backend,
    BudgetExceeded,
    CostLedger,
    OllamaBackend,
    OpenAIBackend,
    SimBackend,
    estimate,
    make_backend,
)
from .gate import validate as validate_skill
from .skill import Skill, distill
from .store import Route, SkillStore, capacity_limit, measure_precision
from .theory import (
    beta_cross_check,
    check_predictions,
    critical_library,
    critical_library_joint,
    fit_beta_costs,
    optimal_branching,
    p_correct_retrieval,
    regime,
    sigma_effective,
)

__version__ = "3.0.0"
__all__ = [
    "ENDPOINTS",
    "PRICES",
    "AnthropicBackend",
    "Backend",
    "BaselineAgent",
    "BudgetExceeded",
    "CostLedger",
    "NawahAgent",
    "OllamaBackend",
    "OpenAIBackend",
    "Result",
    "Route",
    "SimBackend",
    "Skill",
    "SkillStore",
    "bench",
    "beta_cross_check",
    "capacity_limit",
    "check_predictions",
    "consolidate",
    "critical_library",
    "critical_library_joint",
    "curriculum",
    "distill",
    "estimate",
    "fit_beta_costs",
    "make_backend",
    "measure_precision",
    "optimal_branching",
    "p_correct_retrieval",
    "regime",
    "sigma_effective",
    "theory",
    "validate_skill",
]
