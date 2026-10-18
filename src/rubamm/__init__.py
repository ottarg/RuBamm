"""RuBaMM: Native Modelica battery models compiled with Rumoca."""

from __future__ import annotations

from pathlib import Path
import rumoca as rm
from .experiment import Experiment, ParsedStep, parse_step

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def create_session() -> rm.Session:
    """Create a Rumoca Session rooted at the RuBaMM Modelica package directory (`models/`)."""
    return rm.Session(roots=[str(MODELS_DIR)])


def compile_system(system_mo: str, model_name: str, files: list[str] | None = None) -> rm.Model:
    """Compile a Modelica system string using the RuBaMM package tree in `models/`."""
    session = create_session()
    return session.loads(system_mo, model=model_name)


def compile_experiment(
    cell_class: str,
    experiment: Experiment | list[str],
    cell_params: str = "",
    model_name: str = "ExperimentSystem",
) -> rm.Model:
    """Compile a RuBaMM cell model coupled acausally to a PyBaMM-style `Experiment` recipe."""
    exp = Experiment(experiment) if not isinstance(experiment, Experiment) else experiment
    cycler_mo = exp.to_modelica(model_name="GeneratedCycler")
    params_str = f"({cell_params})" if cell_params else ""
    system_mo = f"""
{cycler_mo}

model {model_name}
  {cell_class} cell{params_str};
  GeneratedCycler cycler;
  RuBaMM.Interfaces.Ground gnd;
equation
  connect(cell.p, cycler.p);
  connect(cell.n, cycler.n);
  connect(cell.n, gnd.p);
end {model_name};
"""
    return compile_system(system_mo, model_name=model_name)


__all__ = [
    "Experiment",
    "ParsedStep",
    "parse_step",
    "MODELS_DIR",
    "REPO_ROOT",
    "compile_experiment",
    "compile_system",
    "create_session",
]
