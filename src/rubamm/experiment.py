"""PyBaMM-style Experiment recipe compiler for RuBaMM / Rumoca.

Translates human-readable experiment step strings such as:
- "Discharge at 1C until 3.6 V"
- "Discharge at 0.680616 A for 1000 seconds"
- "Rest for 300 seconds"
- "Charge at 1C until 4.0 V"
- "Hold at 4.0 V for 600 seconds"
- "Hold at 4.0 V until 0.05 A"
- "Discharge at 0.5 W for 500 seconds"

into a native Modelica acausal `TwoPin` cycler model with `when`/`elsewhen` event transitions
and implicit `if ... then ... else` operating-mode equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(frozen=True)
class ParsedStep:
    """Structured representation of a single experiment protocol step."""

    raw: str
    control_type: str  # "current", "voltage", "power", "rest"
    value: float  # Signed value: discharge > 0, charge < 0 (or V_set for voltage)
    termination_type: str  # "time", "voltage_low", "voltage_high", "current_abs"
    termination_value: float
    max_duration: float | None = None


_TIME_UNITS = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


def parse_step(step_str: str, i_typ: float = 0.680616) -> ParsedStep:
    """Parse a PyBaMM-style experiment string into a `ParsedStep`."""
    s = step_str.strip()

    # 1. Rest for X <time_unit>
    m_rest = re.match(
        r"^Rest\s+for\s+([0-9.eE+-]+)\s*([a-zA-Z]+)$", s, flags=re.IGNORECASE
    )
    if m_rest:
        val = float(m_rest.group(1))
        unit = m_rest.group(2).lower()
        return ParsedStep(
            raw=s,
            control_type="rest",
            value=0.0,
            termination_type="time",
            termination_value=val * _TIME_UNITS[unit],
        )

    # 2. Discharge / Charge at X (A|C|W) (for T | until V)
    m_dc = re.match(
        r"^(Discharge|Charge)\s+at\s+([0-9.eE+-]+)\s*(A|C|W)\s+(for|until)\s+([0-9.eE+-]+)\s*([a-zA-Z]+)$",
        s,
        flags=re.IGNORECASE,
    )
    if m_dc:
        direction = m_dc.group(1).lower()
        mag = float(m_dc.group(2))
        unit = m_dc.group(3).upper()
        cond_kw = m_dc.group(4).lower()
        cond_val = float(m_dc.group(5))
        cond_unit = m_dc.group(6)

        sign = 1.0 if direction == "discharge" else -1.0
        if unit == "C":
            ctrl_type = "current"
            ctrl_val = sign * mag * i_typ
        elif unit == "A":
            ctrl_type = "current"
            ctrl_val = sign * mag
        else:
            ctrl_type = "power"
            ctrl_val = sign * mag

        if cond_kw == "for":
            term_type = "time"
            term_val = cond_val * _TIME_UNITS[cond_unit.lower()]
        else:
            if cond_unit.upper() != "V":
                raise ValueError(f"Unsupported cutoff unit in step: {s!r}")
            term_type = "voltage_low" if direction == "discharge" else "voltage_high"
            term_val = cond_val

        return ParsedStep(
            raw=s,
            control_type=ctrl_type,
            value=ctrl_val,
            termination_type=term_type,
            termination_value=term_val,
        )

    # 3. Hold at X V (for T | until I (A|C))
    m_hold = re.match(
        r"^Hold\s+at\s+([0-9.eE+-]+)\s*V\s+(for|until)\s+([0-9.eE+-]+)\s*([a-zA-Z]+)$",
        s,
        flags=re.IGNORECASE,
    )
    if m_hold:
        v_set = float(m_hold.group(1))
        cond_kw = m_hold.group(2).lower()
        cond_val = float(m_hold.group(3))
        cond_unit = m_hold.group(4)

        if cond_kw == "for":
            term_type = "time"
            term_val = cond_val * _TIME_UNITS[cond_unit.lower()]
        else:
            u_up = cond_unit.upper()
            if u_up == "C":
                term_val = cond_val * i_typ
            elif u_up == "A":
                term_val = cond_val
            else:
                raise ValueError(f"Unsupported hold cutoff unit in step: {s!r}")
            term_type = "current_abs"

        return ParsedStep(
            raw=s,
            control_type="voltage",
            value=v_set,
            termination_type=term_type,
            termination_value=term_val,
        )

    raise ValueError(f"Unrecognized Experiment step syntax: {step_str!r}")


class Experiment:
    """PyBaMM-compatible Experiment protocol that compiles to an acausal Modelica cycler component."""

    def __init__(self, steps: Sequence[str], i_typ: float = 0.680616):
        if not steps:
            raise ValueError("Experiment requires at least one step.")
        self.steps_raw = list(steps)
        self.i_typ = float(i_typ)
        self.steps = [parse_step(s, i_typ=self.i_typ) for s in self.steps_raw]

    def to_modelica(self, model_name: str = "ExperimentCycler") -> str:
        """Generate a Modelica `TwoPin` component implementing all experiment steps via `when`/`elsewhen`."""
        when_branches: list[str] = []
        for idx, step in enumerate(self.steps, start=1):
            if step.termination_type == "time":
                cond = f"(time - pre(t_stage)) >= {step.termination_value:.12g}"
            elif step.termination_type == "voltage_low":
                cond = f"v <= {step.termination_value:.12g}"
            elif step.termination_type == "voltage_high":
                cond = f"v >= {step.termination_value:.12g}"
            elif step.termination_type == "current_abs":
                cond = f"abs(p.i) <= {step.termination_value:.12g} and (time - pre(t_stage)) >= 1.0"
            else:
                raise ValueError(f"Unknown termination_type: {step.termination_type}")

            kw = "when" if idx == 1 else "elsewhen"
            when_branches.append(
                f"    {kw} pre(stage) == {idx} and ({cond}) then\n"
                f"      stage = {idx + 1};\n"
                f"      t_stage = time;"
            )

        when_block = "\n".join(when_branches) + "\n    end when;"

        # Build the implicit control residual: 0.0 = if stage == 1 then ...
        ctrl_branches: list[str] = []
        for idx, step in enumerate(self.steps, start=1):
            if step.control_type in ("current", "rest"):
                expr = f"(p.i - ({step.value:.12g}))"
            elif step.control_type == "voltage":
                expr = f"(v - ({step.value:.12g}))"
            elif step.control_type == "power":
                expr = f"(v * p.i - ({step.value:.12g}))"
            else:
                raise ValueError(f"Unknown control_type: {step.control_type}")

            prefix = "if" if idx == 1 else "else if"
            ctrl_branches.append(f"{prefix} stage == {idx} then {expr}")

        # After the final step completes (stage == N + 1), default to open-circuit rest (p.i = 0)
        ctrl_branches.append("else (p.i - 0.0)")
        ctrl_eq = "  0.0 = " + "\n        ".join(ctrl_branches) + ";"

        return (
            f"model {model_name}\n"
            f'  "Auto-generated RuBaMM Experiment Cycler ({len(self.steps)} steps)"\n'
            f"  extends RuBaMM.Interfaces.TwoPin;\n"
            f"  Integer stage(start = 1);\n"
            f"  Real t_stage(start = 0.0);\n"
            f"equation\n"
            f"{when_block}\n\n"
            f"{ctrl_eq}\n"
            f"end {model_name};\n"
        )
