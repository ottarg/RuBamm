"""Checkpoint 3 equivalence tests: Single Particle Model (SPM) vs PyBaMM."""

import numpy as np
import pybamm
import rumoca as rm
from rubamm import compile_system


def test_cp3_spm_constant_current_1c_discharge():
    """Verify RuBaMM SPM matches pybamm.lithium_ion.SPM() across a 1C discharge."""
    Nr = 10
    pybamm_spm = pybamm.lithium_ion.SPM()
    vp = pybamm_spm.default_var_pts.copy()
    vp["r_n"] = Nr
    vp["r_p"] = Nr

    t_eval = np.linspace(0.0, 3400.0, 171)
    solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    sim = pybamm.Simulation(pybamm_spm, var_pts=vp, solver=solver)
    sol = sim.solve(t_eval, t_interp=t_eval)

    system_mo = f"""
    model TestSPM1C
      import P = RuBaMM.Parameters.Marquis2019;
      RuBaMM.Cells.SPM cell(Nr = {Nr});
      RuBaMM.Interfaces.CurrentSource load(I_load = P.I_typ);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end TestSPM1C;
    """
    rm_model = compile_system(system_mo, model_name="TestSPM1C")
    rm_res = rm_model.simulate(
        t=(0.0, 3400.0),
        dt=20.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    # 1. Terminal voltage [V]
    np.testing.assert_allclose(
        rm_res["cell.v"],
        sol["Terminal voltage [V]"].entries,
        rtol=1e-5,
        atol=1e-4,
    )

    # 2. Open-circuit potentials [V]
    np.testing.assert_allclose(
        rm_res["cell.U_n"],
        sol["X-averaged negative electrode open-circuit potential [V]"].entries,
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        rm_res["cell.U_p"],
        sol["X-averaged positive electrode open-circuit potential [V]"].entries,
        rtol=1e-5,
        atol=1e-4,
    )

    # 3. Reaction overpotentials [V]
    np.testing.assert_allclose(
        rm_res["cell.eta_n"],
        sol["X-averaged negative electrode reaction overpotential [V]"].entries,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        rm_res["cell.eta_p"],
        sol["X-averaged positive electrode reaction overpotential [V]"].entries,
        rtol=1e-5,
        atol=1e-5,
    )

    # 4. Exchange current densities [A/m^2]
    np.testing.assert_allclose(
        rm_res["cell.j0_n"],
        sol["X-averaged negative electrode exchange current density [A.m-2]"].entries,
        rtol=1e-5,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        rm_res["cell.j0_p"],
        sol["X-averaged positive electrode exchange current density [A.m-2]"].entries,
        rtol=1e-5,
        atol=1e-4,
    )


def test_cp3_spm_dynamic_current_profile():
    """Verify RuBaMM SPM under a sinusoidal discharge/charge current profile matches PyBaMM."""
    Nr = 10
    pybamm_spm = pybamm.lithium_ion.SPM()
    vp = pybamm_spm.default_var_pts.copy()
    vp["r_n"] = Nr
    vp["r_p"] = Nr

    params = pybamm_spm.default_parameter_values.copy()

    def dynamic_current(t):
        return 0.4 + 0.6 * pybamm.sin(0.01 * t)

    params["Current function [A]"] = dynamic_current

    t_eval = np.linspace(0.0, 1200.0, 121)
    solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    sim = pybamm.Simulation(
        pybamm_spm, parameter_values=params, var_pts=vp, solver=solver
    )
    sol = sim.solve(t_eval, t_interp=t_eval)

    system_mo = f"""
    model TestSPMDynamic
      RuBaMM.Cells.SPM cell(Nr = {Nr});
      RuBaMM.Interfaces.PulseCurrentSource load(I_dc = 0.4, I_amp = 0.6, omega = 0.01);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end TestSPMDynamic;
    """
    rm_model = compile_system(system_mo, model_name="TestSPMDynamic")
    rm_res = rm_model.simulate(
        t=(0.0, 1200.0),
        dt=10.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    np.testing.assert_allclose(
        rm_res["cell.v"],
        sol["Terminal voltage [V]"].entries,
        rtol=1e-5,
        atol=1e-4,
    )
