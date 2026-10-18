"""Checkpoint 4 equivalence tests: 1D Electrolyte FVM & SPMe vs PyBaMM."""

import numpy as np
import pybamm
import rumoca as rm
from rubamm import compile_system


def test_cp4_spme_1c_discharge_equivalence():
    """Verify RuBaMM SPMe matches pybamm.lithium_ion.SPMe() on terminal voltage, overpotentials, and electrolyte profile."""
    Nr = 8
    Nn, Ns, Np = 4, 3, 4

    pybamm_spme = pybamm.lithium_ion.SPMe()
    vp = pybamm_spme.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})

    t_eval = np.linspace(0.0, 3200.0, 161)
    solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    sim = pybamm.Simulation(pybamm_spme, var_pts=vp, solver=solver)
    sol = sim.solve(t_eval, t_interp=t_eval)

    system_mo = f"""
    model TestSPMe1C
      import P = RuBaMM.Parameters.Marquis2019;
      RuBaMM.Cells.SPMe cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.CurrentSource load(I_load = P.I_typ);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end TestSPMe1C;
    """
    rm_model = compile_system(system_mo, model_name="TestSPMe1C")
    rm_res = rm_model.simulate(
        t=(0.0, 3200.0),
        dt=20.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    # 1. Compare domain-averaged electrolyte concentrations [mol/m^3]
    np.testing.assert_allclose(
        rm_res["cell.electrolyte.c_e_n_avg"],
        sol["X-averaged negative electrolyte concentration [mol.m-3]"].entries,
        rtol=2e-3,
        atol=2.0,
    )
    np.testing.assert_allclose(
        rm_res["cell.electrolyte.c_e_s_avg"],
        sol["X-averaged separator electrolyte concentration [mol.m-3]"].entries,
        rtol=2e-3,
        atol=2.0,
    )
    np.testing.assert_allclose(
        rm_res["cell.electrolyte.c_e_p_avg"],
        sol["X-averaged positive electrolyte concentration [mol.m-3]"].entries,
        rtol=2e-3,
        atol=2.0,
    )

    # 2. Compare spatial electrolyte concentration profile c_e(x, t) across all Nn + Ns + Np cells
    c_e_pybamm = sol["Electrolyte concentration [mol.m-3]"].entries
    for i in range(1, Nn + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.electrolyte.c_e_n[{i}]"],
            c_e_pybamm[i - 1, :],
            rtol=3e-3,
            atol=3.0,
        )
    for i in range(1, Ns + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.electrolyte.c_e_s[{i}]"],
            c_e_pybamm[Nn + i - 1, :],
            rtol=3e-3,
            atol=3.0,
        )
    for i in range(1, Np + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.electrolyte.c_e_p[{i}]"],
            c_e_pybamm[Nn + Ns + i - 1, :],
            rtol=3e-3,
            atol=3.0,
        )

    # 3. Compare SPMe overpotential components [V]
    np.testing.assert_allclose(
        rm_res["cell.eta_conc"],
        sol["X-averaged concentration overpotential [V]"].entries,
        rtol=5e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        rm_res["cell.dphi_e_ohm"],
        sol["X-averaged electrolyte ohmic losses [V]"].entries,
        rtol=1e-5,
        atol=1e-5,
    )

    # 4. Compare Terminal voltage [V]
    np.testing.assert_allclose(
        rm_res["cell.v"],
        sol["Terminal voltage [V]"].entries,
        rtol=1e-4,
        atol=5e-4,
    )
