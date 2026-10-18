"""Checkpoint 2 equivalence tests: Marquis2019 OCP functions & Spherical Particle FVM vs PyBaMM."""

import numpy as np
import pybamm
from rubamm import compile_system


def test_cp2_marquis2019_ocp_and_electrolyte_functions():
    """Verify RuBaMM Marquis2019 U_n, U_p, D_e, and kappa_e match PyBaMM across their full domains."""
    param = pybamm.ParameterValues("Marquis2019")
    u_n_pybamm_fn = param["Negative electrode OCP [V]"]
    u_p_pybamm_fn = param["Positive electrode OCP [V]"]
    d_e_pybamm_fn = param["Electrolyte diffusivity [m2.s-1]"]
    k_e_pybamm_fn = param["Electrolyte conductivity [S.m-1]"]

    # Sweep stoichiometry from 0.01 to 0.99 using time in [0.01, 0.99]
    system_mo = """
    model TestMarquisFunctions
      Real sto(start = 0.01);
      Real c_e(start = 100.0);
      Real u_n(start = 0.5);
      Real u_p(start = 4.0);
      Real d_e(start = 1e-10);
      Real k_e(start = 1.0);
    equation
      der(sto) = 1.0;
      c_e = 2000.0 * sto;
      u_n = RuBaMM.Parameters.Marquis2019.U_n(sto);
      u_p = RuBaMM.Parameters.Marquis2019.U_p(sto);
      d_e = RuBaMM.Parameters.Marquis2019.D_e(c_e);
      k_e = RuBaMM.Parameters.Marquis2019.kappa_e(c_e);
    end TestMarquisFunctions;
    """
    rm_model = compile_system(system_mo, model_name="TestMarquisFunctions")
    rm_res = rm_model.simulate(t=(0.0, 0.98), dt=0.01)

    sto_vals = rm_res["sto"]
    c_e_vals = rm_res["c_e"]

    def _eval(val):
        return float(val.evaluate()) if hasattr(val, "evaluate") else float(val)

    u_n_ref = np.array([_eval(u_n_pybamm_fn(s)) for s in sto_vals])
    u_p_ref = np.array([_eval(u_p_pybamm_fn(s)) for s in sto_vals])
    d_e_ref = np.array([_eval(d_e_pybamm_fn(c, 298.15)) for c in c_e_vals])
    k_e_ref = np.array([_eval(k_e_pybamm_fn(c, 298.15)) for c in c_e_vals])

    np.testing.assert_allclose(rm_res["u_n"], u_n_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rm_res["u_p"], u_p_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rm_res["d_e"], d_e_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(rm_res["k_e"], k_e_ref, rtol=1e-12, atol=1e-12)


def test_cp2_spherical_particle_fvm_equivalence():
    """Verify RuBaMM SphericalParticle FVM radial profiles and surface concentration match PyBaMM."""
    Nr = 10
    pybamm_spm = pybamm.lithium_ion.SPM()
    vp = pybamm_spm.default_var_pts.copy()
    vp["r_n"] = Nr
    vp["r_p"] = Nr

    t_eval = np.linspace(0.0, 3600.0, 181)
    solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    sim = pybamm.Simulation(pybamm_spm, var_pts=vp, solver=solver)
    sol = sim.solve(t_eval, t_interp=t_eval)

    # In SPM under constant current I, the molar surface fluxes are exact constants:
    # j_n = I / (F * a_n * L_n * A_cc), j_p = -I / (F * a_p * L_p * A_cc)
    system_mo = f"""
    model TestParticles
      import P = RuBaMM.Parameters.Marquis2019;
      parameter Real I_app = P.I_typ;

      RuBaMM.Submodels.SphericalParticle neg_particle(
        Nr = {Nr},
        R_p = P.R_n,
        D_s = P.D_s_n_ref,
        c_init = P.c_n_init,
        c_max = P.c_n_max
      );

      RuBaMM.Submodels.SphericalParticle pos_particle(
        Nr = {Nr},
        R_p = P.R_p,
        D_s = P.D_s_p_ref,
        c_init = P.c_p_init,
        c_max = P.c_p_max
      );
    equation
      neg_particle.j_flux = I_app / (P.F * P.a_n * P.L_n * P.A_cc);
      pos_particle.j_flux = -I_app / (P.F * P.a_p * P.L_p * P.A_cc);
    end TestParticles;
    """
    import rumoca as rm

    rm_model = compile_system(system_mo, model_name="TestParticles")
    rm_res = rm_model.simulate(
        t=(0.0, 3600.0),
        dt=20.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    # Compare surface concentrations over the entire 1-hour discharge
    np.testing.assert_allclose(
        rm_res["neg_particle.c_s_surf"],
        sol["X-averaged negative particle surface concentration [mol.m-3]"].entries,
        rtol=1e-5,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        rm_res["pos_particle.c_s_surf"],
        sol["X-averaged positive particle surface concentration [mol.m-3]"].entries,
        rtol=1e-5,
        atol=1e-2,
    )

    # Compare every single radial shell i = 1..Nr for both electrodes
    c_s_n_pybamm = sol["X-averaged negative particle concentration [mol.m-3]"].entries
    c_s_p_pybamm = sol["X-averaged positive particle concentration [mol.m-3]"].entries
    for i in range(1, Nr + 1):
        np.testing.assert_allclose(
            rm_res[f"neg_particle.c_s[{i}]"],
            c_s_n_pybamm[i - 1, :],
            rtol=1e-5,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            rm_res[f"pos_particle.c_s[{i}]"],
            c_s_p_pybamm[i - 1, :],
            rtol=1e-5,
            atol=1e-2,
        )
