"""Checkpoint 1 equivalence tests: Acausal Interfaces & Thevenin ECM vs PyBaMM."""

import numpy as np
import pybamm
import pytest
from rubamm import compile_system


def _ocv_thevenin_pybamm(z):
    """Exact same analytical OCV curve used in RuBaMM EquivalentCircuit.ocv_thevenin."""
    return 3.5 + 0.55 * z - 0.08 / (z + 0.05) + 0.08 * pybamm.exp(-25.0 * (1.0 - z))


def test_cp1_thevenin_constant_current_equivalence():
    """Verify RuBaMM Thevenin ECM connected to an acausal CurrentSource matches PyBaMM Thevenin."""
    # 1. Run PyBaMM Thevenin model
    pybamm_model = pybamm.equivalent_circuit.Thevenin()
    params = pybamm_model.default_parameter_values.copy()
    params.update(
        {
            "Cell capacity [A.h]": 100.0,
            "Initial SoC": 0.8,
            "Current function [A]": 100.0,
            "R0 [Ohm]": 0.001,
            "R1 [Ohm]": 0.0005,
            "C1 [F]": 10000.0,
            "Element-1 initial overpotential [V]": 0.0,
            "Entropic change [V/K]": -0.0001,
            "Open-circuit voltage [V]": _ocv_thevenin_pybamm,
        }
    )
    t_eval = np.linspace(0.0, 1800.0, 181)
    sim = pybamm.Simulation(pybamm_model, parameter_values=params)
    sol = sim.solve(t_eval)

    # 2. Run RuBaMM Modelica Thevenin connected acausally to CurrentSource and Ground
    system_mo = """
    model TestTheveninCC
      RuBaMM.EquivalentCircuit.Thevenin cell(
        Q_Ah = 100.0,
        z_init = 0.8,
        R0 = 0.001,
        R1 = 0.0005,
        C1 = 10000.0,
        v_rc1_init = 0.0,
        dUdT = -0.0001
      );
      RuBaMM.Interfaces.CurrentSource load(I_load = 100.0);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end TestTheveninCC;
    """
    rm_model = compile_system(system_mo, model_name="TestTheveninCC")
    rm_res = rm_model.simulate(t=(0.0, 1800.0), dt=10.0)

    # 3. Compare trajectories
    np.testing.assert_allclose(
        rm_res["cell.z"],
        sol["SoC"](t_eval),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rm_res["cell.v_rc1"],
        sol["Element-1 overpotential [V]"](t_eval),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        rm_res["cell.v"],
        sol["Voltage [V]"](t_eval),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        rm_res["cell.T_cell_C"],
        sol["Cell temperature [degC]"](t_eval),
        rtol=1e-4,
        atol=1e-4,
    )


def test_cp1_thevenin_dynamic_pulse_equivalence():
    """Verify RuBaMM Thevenin ECM under sinusoidal dynamic load matches PyBaMM."""
    pybamm_model = pybamm.equivalent_circuit.Thevenin()
    params = pybamm_model.default_parameter_values.copy()

    def dynamic_current(t):
        return 50.0 + 40.0 * pybamm.sin(0.02 * t)

    params.update(
        {
            "Cell capacity [A.h]": 100.0,
            "Initial SoC": 0.6,
            "Current function [A]": dynamic_current,
            "R0 [Ohm]": 0.0015,
            "R1 [Ohm]": 0.0008,
            "C1 [F]": 5000.0,
            "Element-1 initial overpotential [V]": 0.0,
            "Entropic change [V/K]": 0.0,
            "Open-circuit voltage [V]": _ocv_thevenin_pybamm,
        }
    )
    t_eval = np.linspace(0.0, 600.0, 121)
    sim = pybamm.Simulation(pybamm_model, parameter_values=params)
    sol = sim.solve(t_eval)

    system_mo = """
    model TestTheveninPulse
      RuBaMM.EquivalentCircuit.Thevenin cell(
        Q_Ah = 100.0,
        z_init = 0.6,
        R0 = 0.0015,
        R1 = 0.0008,
        C1 = 5000.0,
        v_rc1_init = 0.0,
        dUdT = 0.0
      );
      RuBaMM.Interfaces.PulseCurrentSource load(I_dc = 50.0, I_amp = 40.0, omega = 0.02);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end TestTheveninPulse;
    """
    rm_model = compile_system(system_mo, model_name="TestTheveninPulse")
    rm_res = rm_model.simulate(t=(0.0, 600.0), dt=5.0)

    max_v_err = np.max(np.abs(rm_res["cell.v"] - sol["Voltage [V]"](t_eval)))
    assert max_v_err < 1e-4, f"Max voltage discrepancy {max_v_err} V exceeded 1e-4 V"
