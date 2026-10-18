"""Checkpoint 7 equivalence tests: PyBaMM-style Experiment recipes (CCCV cycling) vs PyBaMM."""

import numpy as np
import pybamm
import rumoca as rm
from rubamm import Experiment, compile_experiment


def test_cp7_spm_experiment_cccv_equivalence():
    """Verify RuBaMM Experiment recipe compiler on SPM matches pybamm.Experiment across a 4-stage CCCV cycle."""
    recipe = [
        "Discharge at 1C until 3.6 V",
        "Rest for 300 seconds",
        "Charge at 1C until 4.0 V",
        "Hold at 4.0 V for 600 seconds",
    ]
    Nr = 8

    # 1. Run PyBaMM Experiment
    pb_spm = pybamm.lithium_ion.SPM()
    vp = pb_spm.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr})
    pb_exp = pybamm.Experiment(recipe, period="2 seconds")
    sim = pybamm.Simulation(
        pb_spm,
        experiment=pb_exp,
        var_pts=vp,
        solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
    )
    sol = sim.solve()

    # 2. Run RuBaMM Experiment
    rm_exp = Experiment(recipe)
    rm_model = compile_experiment(
        cell_class="RuBaMM.Cells.SPM",
        experiment=rm_exp,
        cell_params=f"Nr = {Nr}",
        model_name="TestSPMExperiment",
    )
    t_end = float(sol.t[-1])
    rm_res = rm_model.simulate(
        t=(0.0, t_end),
        dt=5.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    # Compare stage transition timestamps (within < 0.2 seconds across 7000+ seconds)
    for stage_idx in range(1, 5):
        pb_sub = sol.sub_solutions[stage_idx - 1]
        rm_mask = rm_res["cycler.stage"] == stage_idx
        assert np.any(rm_mask), f"Stage {stage_idx} was not reached in Rumoca"
        rm_t_start = float(rm_res.time[rm_mask][0])
        rm_t_end = float(rm_res.time[rm_mask][-1])
        assert abs(rm_t_start - float(pb_sub.t[0])) < 0.2
        assert abs(rm_t_end - float(pb_sub.t[-1])) < 0.2

        # Interpolate PyBaMM stage solution onto interior Rumoca timestamps for this stage
        t_interior = rm_res.time[rm_mask][1:-1]
        v_rm = rm_res["cell.v"][rm_mask][1:-1]
        i_rm = rm_res["cell.i_cell"][rm_mask][1:-1]

        v_pb = np.interp(t_interior, pb_sub.t, pb_sub["Terminal voltage [V]"].entries)
        i_pb = np.interp(t_interior, pb_sub.t, pb_sub["Current [A]"].entries)

        np.testing.assert_allclose(v_rm, v_pb, rtol=1e-4, atol=5e-4)
        np.testing.assert_allclose(i_rm, i_pb, rtol=1e-3, atol=1e-3)


def test_cp7_dfn_experiment_cccv_equivalence():
    """Verify RuBaMM Experiment recipe compiler on full DFN matches pybamm.Experiment across a 4-stage CCCV cycle."""
    recipe = [
        "Discharge at 1C until 3.6 V",
        "Rest for 300 seconds",
        "Charge at 1C until 4.0 V",
        "Hold at 4.0 V for 600 seconds",
    ]
    Nr = 6
    Nn, Ns, Np = 3, 2, 3

    # 1. Run PyBaMM DFN Experiment
    pb_dfn = pybamm.lithium_ion.DFN()
    vp = pb_dfn.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})
    pb_exp = pybamm.Experiment(recipe, period="2 seconds")
    sim = pybamm.Simulation(
        pb_dfn,
        experiment=pb_exp,
        var_pts=vp,
        solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
    )
    sol = sim.solve()

    # 2. Run RuBaMM DFN Experiment
    rm_model = compile_experiment(
        cell_class="RuBaMM.Cells.DFN",
        experiment=recipe,
        cell_params=f"Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np}",
        model_name="TestDFNExperiment",
    )
    t_end = float(sol.t[-1])
    rm_res = rm_model.simulate(
        t=(0.0, t_end),
        dt=10.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    for stage_idx in range(1, 5):
        pb_sub = sol.sub_solutions[stage_idx - 1]
        rm_mask = rm_res["cycler.stage"] == stage_idx
        assert np.any(rm_mask), f"Stage {stage_idx} was not reached in Rumoca DFN"
        rm_t_start = float(rm_res.time[rm_mask][0])
        rm_t_end = float(rm_res.time[rm_mask][-1])
        pb_t_start = float(pb_sub.t[0])
        pb_t_end = float(pb_sub.t[-1])

        # Stage switching times agree within < 0.1% of total cycle time (< 5 seconds over 7500s)
        assert abs(rm_t_start - pb_t_start) < 5.0
        assert abs(rm_t_end - pb_t_end) < 5.0

        # Compare trajectories aligned by elapsed time within each stage
        t_rel_rm = rm_res.time[rm_mask][1:-1] - rm_t_start
        t_rel_pb = pb_sub.t - pb_t_start
        valid = t_rel_rm <= t_rel_pb[-1]
        t_rel_rm = t_rel_rm[valid]

        v_rm = rm_res["cell.v"][rm_mask][1:-1][valid]
        i_rm = rm_res["cell.i_cell"][rm_mask][1:-1][valid]

        v_pb = np.interp(t_rel_rm, t_rel_pb, pb_sub["Terminal voltage [V]"].entries)
        i_pb = np.interp(t_rel_rm, t_rel_pb, pb_sub["Current [A]"].entries)

        np.testing.assert_allclose(v_rm, v_pb, rtol=5e-4, atol=1e-3)
        np.testing.assert_allclose(i_rm, i_pb, rtol=5e-3, atol=5e-3)

