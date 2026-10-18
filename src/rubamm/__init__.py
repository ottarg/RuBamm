"""RuBaMM: Native Modelica battery models compiled with Rumoca."""

from __future__ import annotations

from pathlib import Path
import rumoca as rm

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def create_session() -> rm.Session:
    """Create a Rumoca Session rooted at the RuBaMM Modelica package directory (`models/`)."""
    return rm.Session(roots=[str(MODELS_DIR)])


def compile_system(system_mo: str, model_name: str, files: list[str] | None = None) -> rm.Model:
    """Compile a Modelica system string using the RuBaMM package tree in `models/`."""
    session = create_session()
    return session.loads(system_mo, model=model_name)
