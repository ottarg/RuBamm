"""Checkpoint 5 equivalence tests: Full Doyle-Fuller-Newman (DFN / P2D) Cell vs PyBaMM."""

import numpy as np
import pybamm
import rumoca as rm
from rubamm import compile_system


def test_cp5_dfn_1c_discharge_equivalence():
    """Verify RuBaMM DFN matches pybamm.lithium_ion.DFN() on terminal voltage, potentials, reaction fluxes, and concentrations."""
    Nr = 6
    Nn, Ns, Np = 3, 2, 3

    pybamm_dfn = pybamm.lithium_ion.DFN()
    vp = pybamm_dfn.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})

    t_eval = np.linspace(0.0, 3000.0, 151)
    solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    sim = pybamm.Simulation(pybamm_dfn, var_pts=vp, solver=solver)
    sol = sim.solve(t_eval, t_interp=t_eval)

    system_mo = f"""
    model TestDFN1C
      import P = RuBaMM.Parameters.Marquis2019;
      RuBaMM.Cells.DFN cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.CurrentSource load(I_load = P.I_typ);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end TestDFN1C;
    """
    rm_model = compile_system(system_mo, model_name="TestDFN1C")
    rm_res = rm_model.simulate(
        t=(0.0, 3000.0),
        dt=20.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    # 1. Terminal voltage [V]
    np.testing.assert_allclose(
        rm_res["cell.v"],
        sol["Terminal voltage [V]"].entries,
        rtol=5e-5,
        atol=1e-4,
    )

    # 2. Spatially resolved electrolyte concentrations across all Nn + Ns + Np cells [mol/m^3]
    c_e_pybamm = sol["Electrolyte concentration [mol.m-3]"].entries
    for i in range(1, Nn + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.electrolyte.c_e_n[{i}]"],
            c_e_pybamm[i - 1, :],
            rtol=1e-5,
            atol=1e-2,
        )
    for i in range(1, Ns + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.electrolyte.c_e_s[{i}]"],
            c_e_pybamm[Nn + i - 1, :],
            rtol=1e-5,
            atol=1e-2,
        )
    for i in range(1, Np + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.electrolyte.c_e_p[{i}]"],
            c_e_pybamm[Nn + Ns + i - 1, :],
            rtol=1e-5,
            atol=1e-2,
        )

    # 3. Spatially resolved solid and electrolyte potentials [V]
    phi_s_n_pb = sol["Negative electrode potential [V]"].entries
    phi_s_p_pb = sol["Positive electrode potential [V]"].entries
    phi_e_pb = sol["Electrolyte potential [V]"].entries

    for i in range(1, Nn + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.phi_s_n[{i}]"],
            phi_s_n_pb[i - 1, :],
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            rm_res[f"cell.phi_e_n[{i}]"],
            phi_e_pb[i - 1, :],
            rtol=1e-4,
            atol=1e-4,
        )

    for i in range(1, Ns + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.phi_e_s[{i}]"],
            phi_e_pb[Nn + i - 1, :],
            rtol=1e-4,
            atol=1e-4,
        )

    for i in range(1, Np + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.phi_s_p[{i}]"],
            phi_s_p_pb[i - 1, :],
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            rm_res[f"cell.phi_e_p[{i}]"],
            phi_e_pb[Nn + Ns + i - 1, :],
            rtol=1e-4,
            atol=1e-4,
        )

    # 4. Spatially resolved interfacial Butler-Volmer reaction current densities j_n(x, t) and j_p(x, t) [A/m^2]
    j_n_pb = sol["Negative electrode interfacial current density [A.m-2]"].entries
    j_p_pb = sol["Positive electrode interfacial current density [A.m-2]"].entries
    for i in range(1, Nn + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.j_n[{i}]"],
            j_n_pb[i - 1, :],
            rtol=1e-4,
            atol=1e-4,
        )
    for i in range(1, Np + 1):
        np.testing.assert_allclose(
            rm_res[f"cell.j_p[{i}]"],
            j_p_pb[i - 1, :],
            rtol=1e-4,
            atol=1e-4,
        )

    # 5. X-averaged particle surface concentrations [mol/m^3]
    np.testing.assert_allclose(
        rm_res["cell.c_s_n_surf_avg"],
        sol["X-averaged negative particle surface concentration [mol.m-3]"].entries,
        rtol=1e-5,
        atol=1e-1,
    )
    np.testing.assert_allclose(
        rm_res["cell.c_s_p_surf_avg"],
        sol["X-averaged positive particle surface concentration [mol.m-3]"].entries,
        rtol=1e-5,
        atol=1e-1,
    )
