"""Generate visual equivalence comparison plots with explicit residual/error subplots and runtime bar charts."""

from pathlib import Path
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pybamm
import rumoca as rm
from rubamm import Experiment, compile_experiment, compile_system

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "docs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# Global registry collecting timing records across all checkpoints for the summary figure
TIMING_RECORDS = []


def _record_timing(label: str, pb_build_ms: float, pb_solve_ms: float, rm_compile_ms: float, rm_sim_ms: float):
    rec = {
        "label": label,
        "pb_build_ms": pb_build_ms,
        "pb_solve_ms": pb_solve_ms,
        "pb_total_ms": pb_build_ms + pb_solve_ms,
        "rm_compile_ms": rm_compile_ms,
        "rm_sim_ms": rm_sim_ms,
        "rm_total_ms": rm_compile_ms + rm_sim_ms,
    }
    TIMING_RECORDS.append(rec)
    return rec


def _time_pybamm(make_sim_fn, solve_fn):
    """Time PyBaMM model construction/compilation vs warm solve time in milliseconds."""
    t0 = time.perf_counter()
    sim = make_sim_fn()
    sol_first = solve_fn(sim)
    t_first_total = (time.perf_counter() - t0) * 1e3

    # Warm solve (pure solver execution without model build & CasADi codegen)
    t1 = time.perf_counter()
    sol = solve_fn(sim)
    t_solve = (time.perf_counter() - t1) * 1e3
    t_build = max(0.0, t_first_total - t_solve)
    return sol, t_build, t_solve


def _time_rumoca(compile_fn, sim_fn):
    """Time Rumoca model compilation vs warm simulation time in milliseconds."""
    t0 = time.perf_counter()
    rm_model = compile_fn()
    t_compile = (time.perf_counter() - t0) * 1e3

    # Warm simulation
    _ = sim_fn(rm_model)
    t1 = time.perf_counter()
    rm_res = sim_fn(rm_model)
    t_sim = (time.perf_counter() - t1) * 1e3
    return rm_res, t_compile, t_sim


def _make_paired_figure_with_runtime(suptitle: str):
    """Create a 2x2 grid of (main_ax, error_ax) vertical pairs plus a bottom runtime comparison bar chart."""
    fig = plt.figure(figsize=(14, 13.5))
    fig.suptitle(suptitle, fontweight="bold", fontsize=12.5, y=0.988)
    gs = GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=[2.0, 0.95, 2.0, 0.95, 1.35],
        hspace=0.36,
        wspace=0.24,
        top=0.94,
        bottom=0.045,
        left=0.07,
        right=0.97,
    )
    pairs = []
    for row_pair in (0, 2):
        for col in (0, 1):
            ax_main = fig.add_subplot(gs[row_pair, col])
            ax_err = fig.add_subplot(gs[row_pair + 1, col], sharex=ax_main)
            ax_err.axhline(0.0, color="k", ls=":", lw=0.9, alpha=0.7)
            pairs.append((ax_main, ax_err))
    ax_runtime = fig.add_subplot(gs[4, :])
    return fig, pairs, ax_runtime


def _plot_runtime_bars(ax, records, title="Runtime Comparison: PyBaMM vs RuBaMM (Rumoca)"):
    """Plot grouped stacked/side-by-side runtime comparison bars on `ax`."""
    labels = [r["label"] for r in records]
    n = len(records)
    x = np.arange(n)
    width = 0.18

    pb_build = np.array([r["pb_build_ms"] for r in records])
    pb_solve = np.array([r["pb_solve_ms"] for r in records])
    pb_total = np.array([r["pb_total_ms"] for r in records])

    rm_comp = np.array([r["rm_compile_ms"] for r in records])
    rm_sim = np.array([r["rm_sim_ms"] for r in records])
    rm_total = np.array([r["rm_total_ms"] for r in records])

    b1 = ax.bar(x - 1.5 * width, pb_solve, width, color="#2b2b2b", label="PyBaMM Solve [ms]")
    b2 = ax.bar(x - 0.5 * width, rm_sim, width, color="#e63946", label="RuBaMM Simulate [ms]")
    b3 = ax.bar(x + 0.5 * width, pb_total, width, color="#7f8c8d", label="PyBaMM Total (Build+Solve) [ms]")
    b4 = ax.bar(x + 1.5 * width, rm_total, width, color="#f4a261", label="RuBaMM Total (Compile+Sim) [ms]")

    max_h = max(np.max(pb_total), np.max(rm_total))
    ax.set_ylim(0, max_h * 1.45)

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f} ms",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Wall Time [ms]")
    ax.set_title(title, fontweight="bold")
    ax.legend(ncol=4, loc="upper right")


def _ocv_thevenin_pybamm(z):
    return 3.5 + 0.55 * z - 0.08 / (z + 0.05) + 0.08 * pybamm.exp(-25.0 * (1.0 - z))


def plot_cp1_thevenin_ecm():
    print("Generating CP1: Thevenin ECM comparison + error + runtime plots...")
    t_cc = np.linspace(0.0, 1800.0, 181)

    def make_pb_cc():
        m = pybamm.equivalent_circuit.Thevenin()
        p = m.default_parameter_values.copy()
        p.update(
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
        return pybamm.Simulation(m, parameter_values=p, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol_cc, pb_b_cc, pb_s_cc = _time_pybamm(make_pb_cc, lambda s: s.solve(t_cc, t_interp=t_cc))

    system_cc = """
    model PlotTheveninCC
      RuBaMM.EquivalentCircuit.Thevenin cell(
        Q_Ah = 100.0, z_init = 0.8, R0 = 0.001, R1 = 0.0005,
        C1 = 10000.0, v_rc1_init = 0.0, dUdT = -0.0001
      );
      RuBaMM.Interfaces.CurrentSource load(I_load = 100.0);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end PlotTheveninCC;
    """
    rm_cc, rm_c_cc, rm_s_cc = _time_rumoca(
        lambda: compile_system(system_cc, model_name="PlotTheveninCC"),
        lambda m: m.simulate(t=(0.0, 1800.0), dt=10.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_cc = _record_timing("CP1: Thevenin 1C CC", pb_b_cc, pb_s_cc, rm_c_cc, rm_s_cc)

    t_dyn = np.linspace(0.0, 600.0, 121)

    def make_pb_dyn():
        m = pybamm.equivalent_circuit.Thevenin()
        p = m.default_parameter_values.copy()
        p.update(
            {
                "Cell capacity [A.h]": 100.0,
                "Initial SoC": 0.6,
                "Current function [A]": lambda t: 50.0 + 40.0 * pybamm.sin(0.02 * t),
                "R0 [Ohm]": 0.0015,
                "R1 [Ohm]": 0.0008,
                "C1 [F]": 5000.0,
                "Element-1 initial overpotential [V]": 0.0,
                "Entropic change [V/K]": 0.0,
                "Open-circuit voltage [V]": _ocv_thevenin_pybamm,
            }
        )
        return pybamm.Simulation(m, parameter_values=p, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol_dyn, pb_b_dyn, pb_s_dyn = _time_pybamm(make_pb_dyn, lambda s: s.solve(t_dyn, t_interp=t_dyn))

    system_dyn = """
    model PlotTheveninDyn
      RuBaMM.EquivalentCircuit.Thevenin cell(
        Q_Ah = 100.0, z_init = 0.6, R0 = 0.0015, R1 = 0.0008,
        C1 = 5000.0, v_rc1_init = 0.0, dUdT = 0.0
      );
      RuBaMM.Interfaces.PulseCurrentSource load(I_dc = 50.0, I_amp = 40.0, omega = 0.02);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end PlotTheveninDyn;
    """
    rm_dyn, rm_c_dyn, rm_s_dyn = _time_rumoca(
        lambda: compile_system(system_dyn, model_name="PlotTheveninDyn"),
        lambda m: m.simulate(t=(0.0, 600.0), dt=5.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_dyn = _record_timing("CP1: Thevenin Sine", pb_b_dyn, pb_s_dyn, rm_c_dyn, rm_s_dyn)

    fig, pairs, ax_rt = _make_paired_figure_with_runtime(
        "Checkpoint 1: Thevenin 1-RC Equivalent Circuit Model — Trajectories, Explicit Errors & Runtimes"
    )

    # Pair 0: CC Voltage
    ax, ax_e = pairs[0]
    v_pb = sol_cc["Voltage [V]"].entries
    v_rm = rm_cc["cell.v"]
    err_v = (v_rm - v_pb) * 1e6
    ax.plot(t_cc, v_pb, "k-", lw=2, label="PyBaMM")
    ax.plot(rm_cc.time, v_rm, "r--", lw=1.8, label="RuBaMM")
    ax.set_title(f"1C Constant-Current Voltage (Max |ΔV| = {np.max(np.abs(err_v)):.2f} µV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_cc, err_v, "m-", lw=1.4, label="ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 1: Cell Temperature
    ax, ax_e = pairs[1]
    t_pb_c = sol_cc["Cell temperature [degC]"].entries
    t_rm_c = rm_cc["cell.T_cell_C"]
    err_t = (t_rm_c - t_pb_c) * 1e6
    ax.plot(t_cc, t_pb_c, "k-", lw=2, label="PyBaMM T_cell")
    ax.plot(rm_cc.time, t_rm_c, "r--", lw=1.8, label="RuBaMM T_cell")
    ax.set_title(f"Lumped Cell Temperature (Max |ΔT| = {np.max(np.abs(err_t)):.2f} µK)")
    ax.set_ylabel("Temperature [°C]")
    ax.legend()
    ax_e.plot(t_cc, err_t, "m-", lw=1.4, label="ΔT = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔT [µK]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 2: Sinusoidal Load Voltage
    ax, ax_e = pairs[2]
    vd_pb = sol_dyn["Voltage [V]"].entries
    vd_rm = rm_dyn["cell.v"]
    err_vd = (vd_rm - vd_pb) * 1e6
    ax.plot(t_dyn, vd_pb, "k-", lw=2, label="PyBaMM")
    ax.plot(rm_dyn.time, vd_rm, "r--", lw=1.8, label="RuBaMM")
    ax.set_title(f"Sinusoidal Load Voltage (Max |ΔV| = {np.max(np.abs(err_vd)):.2f} µV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_dyn, err_vd, "m-", lw=1.4, label="ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 3: 1-RC Overpotential
    ax, ax_e = pairs[3]
    vrc_pb = sol_cc["Element-1 overpotential [V]"].entries * 1e3
    vrc_rm = rm_cc["cell.v_rc1"] * 1e3
    err_vrc = (rm_cc["cell.v_rc1"] - sol_cc["Element-1 overpotential [V]"].entries) * 1e6
    ax.plot(t_cc, vrc_pb, "k-", lw=2, label="PyBaMM v_rc1")
    ax.plot(rm_cc.time, vrc_rm, "r--", lw=1.8, label="RuBaMM v_rc1")
    ax.set_title(f"1-RC Polarization Overpotential (Max |Δv_rc1| = {np.max(np.abs(err_vrc)):.2f} µV)")
    ax.set_ylabel("v_rc1 [mV]")
    ax.legend()
    ax_e.plot(t_cc, err_vrc, "m-", lw=1.4, label="Δv_rc1 = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error Δv_rc1 [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    _plot_runtime_bars(ax_rt, [rec_cc, rec_dyn])
    fig.savefig(PLOTS_DIR / "cp1_thevenin_ecm.png")
    plt.close(fig)


def plot_cp2_particle_ocp():
    print("Generating CP2: Marquis2019 OCP & SphericalParticle FVM comparison + error + runtime plots...")
    param = pybamm.ParameterValues("Marquis2019")
    u_n_fn = param["Negative electrode OCP [V]"]
    u_p_fn = param["Positive electrode OCP [V]"]
    k_e_fn = param["Electrolyte conductivity [S.m-1]"]

    system_fn = """
    model PlotMarquisFn
      Real sto(start = 0.02);
      Real c_e(start = 40.0);
      Real u_n(start = 0.5);
      Real u_p(start = 4.0);
      Real k_e(start = 1.0);
    equation
      der(sto) = 1.0;
      c_e = 2000.0 * sto;
      u_n = RuBaMM.Parameters.Marquis2019.U_n(sto);
      u_p = RuBaMM.Parameters.Marquis2019.U_p(sto);
      k_e = RuBaMM.Parameters.Marquis2019.kappa_e(c_e);
    end PlotMarquisFn;
    """
    rm_fn = compile_system(system_fn, model_name="PlotMarquisFn").simulate(t=(0.0, 0.96), dt=0.01)
    sto = rm_fn["sto"]
    c_e = rm_fn["c_e"]

    def _ev(v):
        return float(v.evaluate()) if hasattr(v, "evaluate") else float(v)

    u_n_pb = np.array([_ev(u_n_fn(s)) for s in sto])
    u_p_pb = np.array([_ev(u_p_fn(s)) for s in sto])
    k_e_pb = np.array([_ev(k_e_fn(c, 298.15)) for c in c_e])

    Nr = 10
    t_eval = np.linspace(0.0, 3600.0, 181)

    def make_pb_part():
        m = pybamm.lithium_ion.SPM()
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr, "r_p": Nr})
        return pybamm.Simulation(m, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol, pb_b_part, pb_s_part = _time_pybamm(make_pb_part, lambda s: s.solve(t_eval, t_interp=t_eval))

    system_part = f"""
    model PlotParticles
      import P = RuBaMM.Parameters.Marquis2019;
      RuBaMM.Submodels.SphericalParticle neg_particle(
        Nr = {Nr}, R_p = P.R_n, D_s = P.D_s_n_ref, c_init = P.c_n_init, c_max = P.c_n_max
      );
      RuBaMM.Submodels.SphericalParticle pos_particle(
        Nr = {Nr}, R_p = P.R_p, D_s = P.D_s_p_ref, c_init = P.c_p_init, c_max = P.c_p_max
      );
    equation
      neg_particle.j_flux = P.I_typ / (P.F * P.a_n * P.L_n * P.A_cc);
      pos_particle.j_flux = -P.I_typ / (P.F * P.a_p * P.L_p * P.A_cc);
    end PlotParticles;
    """
    rm_part, rm_c_part, rm_s_part = _time_rumoca(
        lambda: compile_system(system_part, model_name="PlotParticles"),
        lambda m: m.simulate(t=(0.0, 3600.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_part = _record_timing("CP2: Particle FVM (Nr=10)", pb_b_part, pb_s_part, rm_c_part, rm_s_part)

    fig, pairs, ax_rt = _make_paired_figure_with_runtime(
        "Checkpoint 2: Marquis2019 Functions & 1D Spherical Particle FVM — Trajectories, Errors & Runtimes"
    )

    # Pair 0: U_n and U_p
    ax, ax_e = pairs[0]
    ax.plot(sto, u_p_pb, "k-", lw=2, label="PyBaMM U_p(θ)")
    ax.plot(sto, rm_fn["u_p"], "r--", lw=1.8, label="RuBaMM U_p(θ)")
    ax.plot(sto, u_n_pb, "b-", lw=2, label="PyBaMM U_n(θ)")
    ax.plot(sto, rm_fn["u_n"], "c--", lw=1.8, label="RuBaMM U_n(θ)")
    ax.set_title("Open-Circuit Potentials U_n(θ) & U_p(θ)")
    ax.set_ylabel("OCP [V]")
    ax.legend(ncol=2)
    ax_e.plot(sto, (rm_fn["u_p"] - u_p_pb) * 1e15, "r-", lw=1.3, label="ΔU_p")
    ax_e.plot(sto, (rm_fn["u_n"] - u_n_pb) * 1e15, "b-", lw=1.3, label="ΔU_n")
    ax_e.set_ylabel("Error [10⁻¹⁵ V]")
    ax_e.set_xlabel("Stoichiometry θ [-]")
    ax_e.legend(loc="upper right", ncol=2)

    # Pair 1: kappa_e
    ax, ax_e = pairs[1]
    ax.plot(c_e, k_e_pb, "k-", lw=2, label="PyBaMM κ_e(c_e)")
    ax.plot(c_e, rm_fn["k_e"], "r--", lw=1.8, label="RuBaMM κ_e(c_e)")
    ax.set_title("Capiglia1999 Electrolyte Conductivity κ_e(c_e)")
    ax.set_ylabel("Conductivity [S/m]")
    ax.legend()
    ax_e.plot(c_e, (rm_fn["k_e"] - k_e_pb) * 1e15, "m-", lw=1.3, label="Δκ_e = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error [10⁻¹⁵ S/m]")
    ax_e.set_xlabel("Electrolyte Concentration c_e [mol/m³]")
    ax_e.legend(loc="upper right")

    # Pair 2: Surface concentrations vs time
    ax, ax_e = pairs[2]
    cs_n_pb = sol["X-averaged negative particle surface concentration [mol.m-3]"].entries
    cs_p_pb = sol["X-averaged positive particle surface concentration [mol.m-3]"].entries
    cs_n_rm = rm_part["neg_particle.c_s_surf"]
    cs_p_rm = rm_part["pos_particle.c_s_surf"]
    ax.plot(t_eval, cs_p_pb, "k-", lw=2, label="PyBaMM Pos c_s,surf")
    ax.plot(rm_part.time, cs_p_rm, "r--", lw=1.8, label="RuBaMM Pos c_s,surf")
    ax.plot(t_eval, cs_n_pb, "b-", lw=2, label="PyBaMM Neg c_s,surf")
    ax.plot(rm_part.time, cs_n_rm, "c--", lw=1.8, label="RuBaMM Neg c_s,surf")
    ax.set_title(
        f"Particle Surface Concentrations (Max |Δc_s| = {max(np.max(np.abs(cs_n_rm - cs_n_pb)), np.max(np.abs(cs_p_rm - cs_p_pb)))*1e3:.2f} mmol/m³)"
    )
    ax.set_ylabel("c_s,surf [mol/m³]")
    ax.legend(ncol=2)
    ax_e.plot(t_eval, (cs_p_rm - cs_p_pb) * 1e3, "r-", lw=1.4, label="Pos Δc_s,surf")
    ax_e.plot(t_eval, (cs_n_rm - cs_n_pb) * 1e3, "b-", lw=1.4, label="Neg Δc_s,surf")
    ax_e.set_ylabel("Error [mmol/m³]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right", ncol=2)

    # Pair 3: Radial shells c_s(r, t)
    ax, ax_e = pairs[3]
    r_centers = (np.arange(1, Nr + 1) - 0.5) / Nr
    c_n_rad_pb = sol["X-averaged negative particle concentration [mol.m-3]"].entries
    for t_idx, label_t, col in [(45, "t=900s", "tab:blue"), (90, "t=1800s", "tab:orange"), (180, "t=3600s", "tab:green")]:
        pb_rad = c_n_rad_pb[:, t_idx]
        rm_rad = np.array([rm_part[f"neg_particle.c_s[{i}]"][t_idx] for i in range(1, Nr + 1)])
        ax.plot(r_centers, pb_rad, "-", color=col, lw=2, label=f"PyBaMM {label_t}")
        ax.plot(r_centers, rm_rad, "o--", color=col, ms=4, label=f"RuBaMM {label_t}")
        ax_e.plot(r_centers, (rm_rad - pb_rad) * 1e3, "o-", color=col, ms=3.5, lw=1.3, label=f"Δc_s {label_t}")
    ax.set_title("Negative Particle Radial FVM Shells c_s(r, t)")
    ax.set_ylabel("c_s(r) [mol/m³]")
    ax.legend(ncol=2, fontsize=7.5)
    ax_e.set_ylabel("Error [mmol/m³]")
    ax_e.set_xlabel("Normalized Radius r / R_n [-]")
    ax_e.legend(loc="upper right", ncol=3, fontsize=7.5)

    _plot_runtime_bars(ax_rt, [rec_part])
    fig.savefig(PLOTS_DIR / "cp2_particle_ocp.png")
    plt.close(fig)


def plot_cp3_spm():
    print("Generating CP3: Single Particle Model (SPM) comparison + error + runtime plots...")
    Nr = 10
    t_cc = np.linspace(0.0, 3400.0, 171)

    def make_pb_cc():
        m = pybamm.lithium_ion.SPM()
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr, "r_p": Nr})
        return pybamm.Simulation(m, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol_cc, pb_b_cc, pb_s_cc = _time_pybamm(make_pb_cc, lambda s: s.solve(t_cc, t_interp=t_cc))

    system_cc = f"""
    model PlotSPM1C
      import P = RuBaMM.Parameters.Marquis2019;
      RuBaMM.Cells.SPM cell(Nr = {Nr});
      RuBaMM.Interfaces.CurrentSource load(I_load = P.I_typ);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end PlotSPM1C;
    """
    rm_cc, rm_c_cc, rm_s_cc = _time_rumoca(
        lambda: compile_system(system_cc, model_name="PlotSPM1C"),
        lambda m: m.simulate(t=(0.0, 3400.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_cc = _record_timing("CP3: SPM 1C Discharge", pb_b_cc, pb_s_cc, rm_c_cc, rm_s_cc)

    t_dyn = np.linspace(0.0, 1200.0, 121)

    def make_pb_dyn():
        m = pybamm.lithium_ion.SPM()
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr, "r_p": Nr})
        p = m.default_parameter_values.copy()
        p["Current function [A]"] = lambda t: 0.4 + 0.6 * pybamm.sin(0.01 * t)
        return pybamm.Simulation(m, parameter_values=p, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol_dyn, pb_b_dyn, pb_s_dyn = _time_pybamm(make_pb_dyn, lambda s: s.solve(t_dyn, t_interp=t_dyn))

    system_dyn = f"""
    model PlotSPMDyn
      RuBaMM.Cells.SPM cell(Nr = {Nr});
      RuBaMM.Interfaces.PulseCurrentSource load(I_dc = 0.4, I_amp = 0.6, omega = 0.01);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end PlotSPMDyn;
    """
    rm_dyn, rm_c_dyn, rm_s_dyn = _time_rumoca(
        lambda: compile_system(system_dyn, model_name="PlotSPMDyn"),
        lambda m: m.simulate(t=(0.0, 1200.0), dt=10.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_dyn = _record_timing("CP3: SPM Sine Load", pb_b_dyn, pb_s_dyn, rm_c_dyn, rm_s_dyn)

    fig, pairs, ax_rt = _make_paired_figure_with_runtime(
        "Checkpoint 3: Single Particle Model (SPM) — Trajectories, Explicit Errors & Runtimes"
    )

    # Pair 0: 1C Terminal Voltage
    ax, ax_e = pairs[0]
    v_pb = sol_cc["Terminal voltage [V]"].entries
    v_rm = rm_cc["cell.v"]
    err_v = (v_rm - v_pb) * 1e6
    ax.plot(t_cc, v_pb, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_cc.time, v_rm, "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title(f"1C Discharge Terminal Voltage (Max |ΔV| = {np.max(np.abs(err_v)):.2f} µV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_cc, err_v, "m-", lw=1.4, label="ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 1: Reaction Overpotentials eta_n & eta_p
    ax, ax_e = pairs[1]
    eta_n_pb = sol_cc["X-averaged negative electrode reaction overpotential [V]"].entries
    eta_p_pb = sol_cc["X-averaged positive electrode reaction overpotential [V]"].entries
    eta_n_rm = rm_cc["cell.eta_n"]
    eta_p_rm = rm_cc["cell.eta_p"]
    ax.plot(t_cc, eta_n_pb * 1e3, "b-", lw=2, label="PyBaMM η_n")
    ax.plot(rm_cc.time, eta_n_rm * 1e3, "c--", lw=1.8, label="RuBaMM η_n")
    ax.plot(t_cc, eta_p_pb * 1e3, "k-", lw=2, label="PyBaMM η_p")
    ax.plot(rm_cc.time, eta_p_rm * 1e3, "r--", lw=1.8, label="RuBaMM η_p")
    ax.set_title("Butler-Volmer Reaction Overpotentials η_n & η_p")
    ax.set_ylabel("Overpotential [mV]")
    ax.legend(ncol=2)
    ax_e.plot(t_cc, (eta_n_rm - eta_n_pb) * 1e6, "b-", lw=1.4, label="Δη_n")
    ax_e.plot(t_cc, (eta_p_rm - eta_p_pb) * 1e6, "r-", lw=1.4, label="Δη_p")
    ax_e.set_ylabel("Error Δη [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right", ncol=2)

    # Pair 2: Sinusoidal Load Voltage
    ax, ax_e = pairs[2]
    vd_pb = sol_dyn["Terminal voltage [V]"].entries
    vd_rm = rm_dyn["cell.v"]
    err_vd = (vd_rm - vd_pb) * 1e6
    ax.plot(t_dyn, vd_pb, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_dyn.time, vd_rm, "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title(f"Dynamic Sinusoidal Load Voltage (Max |ΔV| = {np.max(np.abs(err_vd)):.2f} µV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_dyn, err_vd, "m-", lw=1.4, label="ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 3: Electrode OCPs U_n & U_p during 1C discharge
    ax, ax_e = pairs[3]
    u_n_pb = sol_cc["X-averaged negative electrode open-circuit potential [V]"].entries
    u_p_pb = sol_cc["X-averaged positive electrode open-circuit potential [V]"].entries
    u_n_rm = rm_cc["cell.U_n"]
    u_p_rm = rm_cc["cell.U_p"]
    ax.plot(t_cc, u_p_pb, "k-", lw=2, label="PyBaMM U_p(t)")
    ax.plot(rm_cc.time, u_p_rm, "r--", lw=1.8, label="RuBaMM U_p(t)")
    ax.plot(t_cc, u_n_pb, "b-", lw=2, label="PyBaMM U_n(t)")
    ax.plot(rm_cc.time, u_n_rm, "c--", lw=1.8, label="RuBaMM U_n(t)")
    ax.set_title("Electrode Open-Circuit Potentials U_n(t) & U_p(t)")
    ax.set_ylabel("OCP [V]")
    ax.legend(ncol=2)
    ax_e.plot(t_cc, (u_p_rm - u_p_pb) * 1e6, "r-", lw=1.4, label="ΔU_p")
    ax_e.plot(t_cc, (u_n_rm - u_n_pb) * 1e6, "b-", lw=1.4, label="ΔU_n")
    ax_e.set_ylabel("Error ΔU [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right", ncol=2)

    _plot_runtime_bars(ax_rt, [rec_cc, rec_dyn])
    fig.savefig(PLOTS_DIR / "cp3_spm.png")
    plt.close(fig)


def plot_cp4_spme():
    print("Generating CP4: SPMe & 1D Electrolyte FVM comparison + error + runtime plots...")
    Nr = 8
    Nn, Ns, Np = 4, 3, 4
    t_eval = np.linspace(0.0, 3200.0, 161)

    def make_pb_spme():
        m = pybamm.lithium_ion.SPMe()
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})
        return pybamm.Simulation(m, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol, pb_b_spme, pb_s_spme = _time_pybamm(make_pb_spme, lambda s: s.solve(t_eval, t_interp=t_eval))

    system_mo = f"""
    model PlotSPMe1C
      import P = RuBaMM.Parameters.Marquis2019;
      RuBaMM.Cells.SPMe cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.CurrentSource load(I_load = P.I_typ);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end PlotSPMe1C;
    """
    rm_res, rm_c_spme, rm_s_spme = _time_rumoca(
        lambda: compile_system(system_mo, model_name="PlotSPMe1C"),
        lambda m: m.simulate(t=(0.0, 3200.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_spme = _record_timing("CP4: SPMe 1C Discharge", pb_b_spme, pb_s_spme, rm_c_spme, rm_s_spme)

    fig, pairs, ax_rt = _make_paired_figure_with_runtime(
        "Checkpoint 4: Single Particle Model with Electrolyte (SPMe) — Trajectories, Errors & Runtimes"
    )

    # Pair 0: 1C Discharge Voltage
    ax, ax_e = pairs[0]
    v_pb = sol["Terminal voltage [V]"].entries
    v_rm = rm_res["cell.v"]
    err_v = (v_rm - v_pb) * 1e3
    ax.plot(t_eval, v_pb, "k-", lw=2, label="PyBaMM SPMe")
    ax.plot(rm_res.time, v_rm, "r--", lw=1.8, label="RuBaMM SPMe")
    ax.set_title(f"1C Discharge Voltage (Max |ΔV| = {np.max(np.abs(err_v)):.3f} mV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_eval, err_v, "m-", lw=1.4, label="ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [mV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="lower right")

    # Pair 1: Domain-Averaged Electrolyte Concentrations
    ax, ax_e = pairs[1]
    ce_n_pb = sol["X-averaged negative electrolyte concentration [mol.m-3]"].entries
    ce_s_pb = sol["X-averaged separator electrolyte concentration [mol.m-3]"].entries
    ce_p_pb = sol["X-averaged positive electrolyte concentration [mol.m-3]"].entries
    ce_n_rm = rm_res["cell.electrolyte.c_e_n_avg"]
    ce_s_rm = rm_res["cell.electrolyte.c_e_s_avg"]
    ce_p_rm = rm_res["cell.electrolyte.c_e_p_avg"]
    ax.plot(t_eval, ce_n_pb, "b-", lw=2, label="PyBaMM Neg c_e,avg")
    ax.plot(rm_res.time, ce_n_rm, "c--", lw=1.8, label="RuBaMM Neg c_e,avg")
    ax.plot(t_eval, ce_s_pb, "g-", lw=2, label="PyBaMM Sep c_e,avg")
    ax.plot(rm_res.time, ce_s_rm, "y--", lw=1.8, label="RuBaMM Sep c_e,avg")
    ax.plot(t_eval, ce_p_pb, "k-", lw=2, label="PyBaMM Pos c_e,avg")
    ax.plot(rm_res.time, ce_p_rm, "r--", lw=1.8, label="RuBaMM Pos c_e,avg")
    ax.set_title("Domain-Averaged Electrolyte Concentrations")
    ax.set_ylabel("c_e [mol/m³]")
    ax.legend(fontsize=7.5, ncol=2)
    ax_e.plot(t_eval, ce_n_rm - ce_n_pb, "b-", lw=1.3, label="Neg Δc_e,avg")
    ax_e.plot(t_eval, ce_s_rm - ce_s_pb, "g-", lw=1.3, label="Sep Δc_e,avg")
    ax_e.plot(t_eval, ce_p_rm - ce_p_pb, "r-", lw=1.3, label="Pos Δc_e,avg")
    ax_e.set_ylabel("Error Δc_e [mol/m³]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right", ncol=3, fontsize=7.5)

    # Pair 2: Spatial Electrolyte Profile c_e(x, t)
    ax, ax_e = pairs[2]
    c_e_pb = sol["Electrolyte concentration [mol.m-3]"].entries
    x_idx = np.arange(1, Nn + Ns + Np + 1)
    for t_i, lbl, col in [(25, "t=500s", "tab:blue"), (80, "t=1600s", "tab:orange"), (160, "t=3200s", "tab:green")]:
        pb_prof = c_e_pb[:, t_i]
        rm_prof = np.array(
            [rm_res[f"cell.electrolyte.c_e_n[{i}]"][t_i] for i in range(1, Nn + 1)]
            + [rm_res[f"cell.electrolyte.c_e_s[{i}]"][t_i] for i in range(1, Ns + 1)]
            + [rm_res[f"cell.electrolyte.c_e_p[{i}]"][t_i] for i in range(1, Np + 1)]
        )
        ax.plot(x_idx, pb_prof, "-", color=col, lw=2, label=f"PyBaMM {lbl}")
        ax.plot(x_idx, rm_prof, "o--", color=col, ms=4, label=f"RuBaMM {lbl}")
        ax_e.plot(x_idx, rm_prof - pb_prof, "o-", color=col, ms=3.5, lw=1.3, label=f"Δc_e {lbl}")
    for a in (ax, ax_e):
        a.axvline(Nn + 0.5, color="gray", ls=":", lw=1)
        a.axvline(Nn + Ns + 0.5, color="gray", ls=":", lw=1)
    ax.set_title("Spatial Electrolyte Profile c_e(x, t) [Neg | Sep | Pos]")
    ax.set_ylabel("c_e [mol/m³]")
    ax.legend(fontsize=7.5, ncol=2)
    ax_e.set_ylabel("Error Δc_e [mol/m³]")
    ax_e.set_xlabel("FVM Control Volume Index [-]")
    ax_e.legend(loc="upper right", ncol=3, fontsize=7.5)

    # Pair 3: Electrolyte Concentration & Ohmic Overpotentials
    ax, ax_e = pairs[3]
    eta_c_pb = sol["X-averaged concentration overpotential [V]"].entries * 1e3
    ohm_e_pb = sol["X-averaged electrolyte ohmic losses [V]"].entries * 1e3
    eta_c_rm = rm_res["cell.eta_conc"] * 1e3
    ohm_e_rm = rm_res["cell.dphi_e_ohm"] * 1e3
    ax.plot(t_eval, eta_c_pb, "k-", lw=2, label="PyBaMM η_conc")
    ax.plot(rm_res.time, eta_c_rm, "r--", lw=1.8, label="RuBaMM η_conc")
    ax.plot(t_eval, ohm_e_pb, "b-", lw=2, label="PyBaMM Δφ_e,ohm")
    ax.plot(rm_res.time, ohm_e_rm, "c--", lw=1.8, label="RuBaMM Δφ_e,ohm")
    ax.set_title("Electrolyte Concentration & Ohmic Overpotentials")
    ax.set_ylabel("Overpotential [mV]")
    ax.legend(ncol=2)
    ax_e.plot(t_eval, eta_c_rm - eta_c_pb, "r-", lw=1.4, label="Δη_conc [mV]")
    ax_e.plot(t_eval, ohm_e_rm - ohm_e_pb, "b-", lw=1.4, label="Δ(Δφ_e,ohm) [mV]")
    ax_e.set_ylabel("Error [mV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="lower right", ncol=2)

    _plot_runtime_bars(ax_rt, [rec_spme])
    fig.savefig(PLOTS_DIR / "cp4_spme.png")
    plt.close(fig)


def plot_cp5_dfn():
    print("Generating CP5: Full P2D Doyle-Fuller-Newman (DFN) comparison + error + runtime plots...")
    Nr = 6
    Nn, Ns, Np = 3, 2, 3
    t_eval = np.linspace(0.0, 3000.0, 151)

    def make_pb_dfn():
        m = pybamm.lithium_ion.DFN()
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})
        return pybamm.Simulation(m, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol, pb_b_dfn, pb_s_dfn = _time_pybamm(make_pb_dfn, lambda s: s.solve(t_eval, t_interp=t_eval))

    system_mo = f"""
    model PlotDFN1C
      import P = RuBaMM.Parameters.Marquis2019;
      RuBaMM.Cells.DFN cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.CurrentSource load(I_load = P.I_typ);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end PlotDFN1C;
    """
    rm_res, rm_c_dfn, rm_s_dfn = _time_rumoca(
        lambda: compile_system(system_mo, model_name="PlotDFN1C"),
        lambda m: m.simulate(t=(0.0, 3000.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_dfn = _record_timing("CP5: DFN 1C Discharge", pb_b_dfn, pb_s_dfn, rm_c_dfn, rm_s_dfn)

    fig, pairs, ax_rt = _make_paired_figure_with_runtime(
        "Checkpoint 5: Doyle-Fuller-Newman P2D Model (DFN) — Trajectories, Explicit Errors & Runtimes"
    )

    # Pair 0: Terminal Voltage
    ax, ax_e = pairs[0]
    v_pb = sol["Terminal voltage [V]"].entries
    v_rm = rm_res["cell.v"]
    err_v = (v_rm - v_pb) * 1e6
    ax.plot(t_eval, v_pb, "k-", lw=2, label="PyBaMM DFN")
    ax.plot(rm_res.time, v_rm, "r--", lw=1.8, label="RuBaMM DFN")
    ax.set_title(f"1C Terminal Voltage (Max |ΔV| = {np.max(np.abs(err_v)):.2f} µV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_eval, err_v, "m-", lw=1.4, label="ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [µV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 1: Spatial Electrolyte Concentration c_e(x, t)
    ax, ax_e = pairs[1]
    c_e_pb = sol["Electrolyte concentration [mol.m-3]"].entries
    x_idx = np.arange(1, Nn + Ns + Np + 1)
    for t_i, lbl, col in [(10, "t=200s", "tab:blue"), (50, "t=1000s", "tab:orange"), (150, "t=3000s", "tab:green")]:
        pb_prof = c_e_pb[:, t_i]
        rm_prof = np.array(
            [rm_res[f"cell.electrolyte.c_e_n[{i}]"][t_i] for i in range(1, Nn + 1)]
            + [rm_res[f"cell.electrolyte.c_e_s[{i}]"][t_i] for i in range(1, Ns + 1)]
            + [rm_res[f"cell.electrolyte.c_e_p[{i}]"][t_i] for i in range(1, Np + 1)]
        )
        ax.plot(x_idx, pb_prof, "-", color=col, lw=2, label=f"PyBaMM {lbl}")
        ax.plot(x_idx, rm_prof, "o--", color=col, ms=4, label=f"RuBaMM {lbl}")
        ax_e.plot(x_idx, (rm_prof - pb_prof) * 1e3, "o-", color=col, ms=3.5, lw=1.3, label=f"Δc_e {lbl}")
    for a in (ax, ax_e):
        a.axvline(Nn + 0.5, color="gray", ls=":", lw=1)
        a.axvline(Nn + Ns + 0.5, color="gray", ls=":", lw=1)
    ax.set_title("Spatial Electrolyte Concentration c_e(x, t)")
    ax.set_ylabel("c_e [mol/m³]")
    ax.legend(fontsize=7.5, ncol=2)
    ax_e.set_ylabel("Error [mmol/m³]")
    ax_e.set_xlabel("FVM Control Volume Index [-]")
    ax_e.legend(loc="upper right", ncol=3, fontsize=7.5)

    # Pair 2: Spatial Electrolyte Potential phi_e(x, t)
    ax, ax_e = pairs[2]
    phi_e_pb = sol["Electrolyte potential [V]"].entries
    for t_i, lbl, col in [(10, "t=200s", "tab:blue"), (75, "t=1500s", "tab:orange"), (150, "t=3000s", "tab:green")]:
        pb_phi = phi_e_pb[:, t_i]
        rm_phi = np.array(
            [rm_res[f"cell.phi_e_n[{i}]"][t_i] for i in range(1, Nn + 1)]
            + [rm_res[f"cell.phi_e_s[{i}]"][t_i] for i in range(1, Ns + 1)]
            + [rm_res[f"cell.phi_e_p[{i}]"][t_i] for i in range(1, Np + 1)]
        )
        ax.plot(x_idx, pb_phi, "-", color=col, lw=2, label=f"PyBaMM {lbl}")
        ax.plot(x_idx, rm_phi, "o--", color=col, ms=4, label=f"RuBaMM {lbl}")
        ax_e.plot(x_idx, (rm_phi - pb_phi) * 1e6, "o-", color=col, ms=3.5, lw=1.3, label=f"Δφ_e {lbl}")
    for a in (ax, ax_e):
        a.axvline(Nn + 0.5, color="gray", ls=":", lw=1)
        a.axvline(Nn + Ns + 0.5, color="gray", ls=":", lw=1)
    ax.set_title("Spatial Electrolyte Potential φ_e(x, t)")
    ax.set_ylabel("Potential φ_e [V]")
    ax.legend(fontsize=7.5, ncol=2)
    ax_e.set_ylabel("Error Δφ_e [µV]")
    ax_e.set_xlabel("FVM Control Volume Index [-]")
    ax_e.legend(loc="upper right", ncol=3, fontsize=7.5)

    # Pair 3: Spatial Reaction Current Density j(x, t)
    ax, ax_e = pairs[3]
    j_n_pb = sol["Negative electrode interfacial current density [A.m-2]"].entries
    j_p_pb = sol["Positive electrode interfacial current density [A.m-2]"].entries
    ax.plot(t_eval, j_n_pb[0, :], "b-", lw=2, label="PyBaMM j_n[1] (CC)")
    ax.plot(rm_res.time, rm_res["cell.j_n[1]"], "c--", lw=1.8, label="RuBaMM j_n[1]")
    ax.plot(t_eval, j_n_pb[-1, :], "g-", lw=2, label=f"PyBaMM j_n[{Nn}] (Sep)")
    ax.plot(rm_res.time, rm_res[f"cell.j_n[{Nn}]"], "y--", lw=1.8, label=f"RuBaMM j_n[{Nn}]")
    ax.plot(t_eval, j_p_pb[0, :], "k-", lw=2, label="PyBaMM j_p[1] (Sep)")
    ax.plot(rm_res.time, rm_res["cell.j_p[1]"], "r--", lw=1.8, label="RuBaMM j_p[1]")
    ax.set_title("Spatial Interfacial Reaction Current Density j(x, t)")
    ax.set_ylabel("j [A/m²]")
    ax.legend(fontsize=7.5, ncol=2)
    ax_e.plot(t_eval, (rm_res["cell.j_n[1]"] - j_n_pb[0, :]) * 1e3, "b-", lw=1.3, label="Δj_n[1]")
    ax_e.plot(t_eval, (rm_res[f"cell.j_n[{Nn}]"] - j_n_pb[-1, :]) * 1e3, "g-", lw=1.3, label=f"Δj_n[{Nn}]")
    ax_e.plot(t_eval, (rm_res["cell.j_p[1]"] - j_p_pb[0, :]) * 1e3, "r-", lw=1.3, label="Δj_p[1]")
    ax_e.set_ylabel("Error Δj [mA/m²]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right", ncol=3, fontsize=7.5)

    _plot_runtime_bars(ax_rt, [rec_dfn])
    fig.savefig(PLOTS_DIR / "cp5_dfn.png")
    plt.close(fig)


def plot_cp6_acausal_coupling():
    print("Generating CP6: Acausal Subsystem Coupling comparison + error + runtime plots...")
    Nr = 6
    Nn, Ns, Np = 3, 2, 3

    # 1. Constant-Voltage Hold on DFN
    V_hold = 3.65
    t_cv = np.linspace(0.0, 600.0, 61)

    def make_pb_cv():
        m = pybamm.lithium_ion.DFN(options={"operating mode": "voltage"})
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})
        p = m.default_parameter_values.copy()
        p["Voltage function [V]"] = V_hold
        return pybamm.Simulation(m, parameter_values=p, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol_cv, pb_b_cv, pb_s_cv = _time_pybamm(make_pb_cv, lambda s: s.solve(t_cv, t_interp=t_cv))

    sys_cv = f"""
    model PlotDFNCV
      RuBaMM.Cells.DFN cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.VoltageSource vsrc(V_set = {V_hold});
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, vsrc.p);
      connect(cell.n, vsrc.n);
      connect(cell.n, gnd.p);
    end PlotDFNCV;
    """
    rm_cv, rm_c_cv, rm_s_cv = _time_rumoca(
        lambda: compile_system(sys_cv, model_name="PlotDFNCV"),
        lambda m: m.simulate(t=(0.0, 600.0), dt=10.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_cv = _record_timing("CP6: DFN Constant Voltage", pb_b_cv, pb_s_cv, rm_c_cv, rm_s_cv)

    # 2. Passive Resistor Load on DFN
    R_load = 5.0
    t_res = np.linspace(0.0, 1200.0, 61)

    def make_pb_res():
        m = pybamm.lithium_ion.DFN(
            options={"operating mode": lambda v: v["Current [A]"] - v["Terminal voltage [V]"] / R_load}
        )
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})
        return pybamm.Simulation(m, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8))

    sol_res, pb_b_res, pb_s_res = _time_pybamm(make_pb_res, lambda s: s.solve(t_res, t_interp=t_res))

    sys_res = f"""
    model PlotDFNRes
      RuBaMM.Cells.DFN cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.Resistor load(R = {R_load});
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end PlotDFNRes;
    """
    rm_res, rm_c_res, rm_s_res = _time_rumoca(
        lambda: compile_system(sys_res, model_name="PlotDFNRes"),
        lambda m: m.simulate(t=(0.0, 1200.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )
    rec_res = _record_timing("CP6: DFN Resistor Load", pb_b_res, pb_s_res, rm_c_res, rm_s_res)

    # 3. Parallel 2-Cell Pack Balancing
    sys_par = """
    model PlotParallelCells
      RuBaMM.Cells.SPM cell1(Nr = 6, c_n_init = 20000.0, c_p_init = 30730.0);
      RuBaMM.Cells.SPM cell2(Nr = 6, c_n_init = 15000.0, c_p_init = 36000.0);
      RuBaMM.Interfaces.Resistor r_bus1(R = 0.005);
      RuBaMM.Interfaces.Resistor r_bus2(R = 0.005);
      RuBaMM.Interfaces.CurrentSource pack_load(I_load = 1.0);
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell1.p, r_bus1.p);
      connect(r_bus1.n, pack_load.p);
      connect(cell2.p, r_bus2.p);
      connect(r_bus2.n, pack_load.p);
      connect(cell1.n, gnd.p);
      connect(cell2.n, gnd.p);
      connect(pack_load.n, gnd.p);
    end PlotParallelCells;
    """
    rm_par, _, _ = _time_rumoca(
        lambda: compile_system(sys_par, model_name="PlotParallelCells"),
        lambda m: m.simulate(t=(0.0, 1000.0), dt=10.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)),
    )

    fig, pairs, ax_rt = _make_paired_figure_with_runtime(
        "Checkpoint 6: Acausal Circuit & Multi-Cell Pack Coupling — Trajectories, Errors & Runtimes"
    )

    # Pair 0: DFN Constant Voltage Hold Current
    ax, ax_e = pairs[0]
    i_cv_pb = sol_cv["Current [A]"].entries[1:]
    i_cv_rm = rm_cv["cell.i_cell"][1:]
    err_icv = (i_cv_rm - i_cv_pb) * 1e3
    ax.plot(t_cv[1:], i_cv_pb, "k-", lw=2, label="PyBaMM DFN (Voltage Mode)")
    ax.plot(rm_cv.time[1:], i_cv_rm, "r--", lw=1.8, label="RuBaMM DFN + VoltageSource")
    ax.set_title(f"DFN Constant-Voltage Hold V=3.65 V (Max |ΔI| = {np.max(np.abs(err_icv)):.2f} mA)")
    ax.set_ylabel("Emergent Current [A]")
    ax.legend()
    ax_e.plot(t_cv[1:], err_icv, "m-", lw=1.4, label="ΔI = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔI [mA]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 1: DFN Passive Resistor Load
    ax, ax_e = pairs[1]
    v_res_pb = sol_res["Terminal voltage [V]"].entries
    v_res_rm = rm_res["cell.v"]
    i_res_pb = sol_res["Current [A]"].entries
    i_res_rm = rm_res["cell.i_cell"]
    err_vres = (v_res_rm - v_res_pb) * 1e6
    err_ires = (i_res_rm - i_res_pb) * 1e6
    ax.plot(t_res, v_res_pb, "k-", lw=2, label="PyBaMM V(t) [V]")
    ax.plot(rm_res.time, v_res_rm, "r--", lw=1.8, label="RuBaMM V(t) [V]")
    ax.plot(t_res, i_res_pb, "b-", lw=1.8, label="PyBaMM I(t) [A]")
    ax.plot(rm_res.time, i_res_rm, "c--", lw=1.8, label="RuBaMM I(t) [A]")
    ax.set_title("DFN Coupled to Passive Resistor Load (R = 5.0 Ω)")
    ax.set_ylabel("V [V] / I [A]")
    ax.legend(ncol=2)
    ax_e.plot(t_res, err_vres, "r-", lw=1.4, label="ΔV [µV]")
    ax_e.plot(t_res, err_ires, "b-", lw=1.4, label="ΔI [µA]")
    ax_e.set_ylabel("Error [µV / µA]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right", ncol=2)

    # Pair 2: Parallel 2-Cell Pack Current Sharing & KCL Residual
    ax, ax_e = pairs[2]
    i1 = rm_par["cell1.i_cell"]
    i2 = rm_par["cell2.i_cell"]
    kcl_err = (i1 + i2 - 1.0) * 1e15
    ax.plot(rm_par.time, i1, "r-", lw=2, label="Cell 1 Current (High Initial SoC)")
    ax.plot(rm_par.time, i2, "b-", lw=2, label="Cell 2 Current (Low Initial SoC)")
    ax.plot(rm_par.time, i1 + i2, "k--", lw=1.5, label="Total Pack Current (I1 + I2 = 1.0 A)")
    ax.set_title("Parallel 2-Cell Pack Current Sharing & Self-Balancing")
    ax.set_ylabel("Current [A]")
    ax.legend()
    ax_e.plot(rm_par.time, kcl_err, "m-", lw=1.4, label="KCL Residual (I1 + I2 - I_load)")
    ax_e.set_ylabel("KCL Error [10⁻¹⁵ A]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 3: Parallel 2-Cell Pack Surface Concentrations & Imbalance Convergence
    ax, ax_e = pairs[3]
    cs1 = rm_par["cell1.neg_particle.c_s_surf"]
    cs2 = rm_par["cell2.neg_particle.c_s_surf"]
    ax.plot(rm_par.time, cs1, "r-", lw=2, label="Cell 1 Neg Surface c_s")
    ax.plot(rm_par.time, cs2, "b-", lw=2, label="Cell 2 Neg Surface c_s")
    ax.set_title("Parallel 2-Cell Pack Surface Concentration Convergence")
    ax.set_ylabel("c_s,surf [mol/m³]")
    ax.legend()
    ax_e.plot(rm_par.time, cs1 - cs2, "g-", lw=1.5, label="Cell-to-Cell Imbalance (c_s,1 - c_s,2)")
    ax_e.set_ylabel("Δc_s,12 [mol/m³]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    _plot_runtime_bars(ax_rt, [rec_cv, rec_res])
    fig.savefig(PLOTS_DIR / "cp6_acausal_coupling.png")
    plt.close(fig)


def _compute_stage_aligned_errors(sol_pb, rm_res):
    """Compute stage-aligned (t, v_err_mV, i_err_mA) across all 4 experiment stages."""
    t_all = []
    v_err_all = []
    i_err_all = []
    for stage_idx in range(1, 5):
        pb_sub = sol_pb.sub_solutions[stage_idx - 1]
        rm_mask = rm_res["cycler.stage"] == stage_idx
        if not np.any(rm_mask):
            continue
        t_rm_stage = rm_res.time[rm_mask][1:-1]
        v_rm_stage = rm_res["cell.v"][rm_mask][1:-1]
        i_rm_stage = rm_res["cell.i_cell"][rm_mask][1:-1]

        t_rel_rm = t_rm_stage - float(rm_res.time[rm_mask][0])
        t_rel_pb = pb_sub.t - float(pb_sub.t[0])
        valid = t_rel_rm <= t_rel_pb[-1]

        t_rm_stage = t_rm_stage[valid]
        t_rel_rm = t_rel_rm[valid]
        v_rm_stage = v_rm_stage[valid]
        i_rm_stage = i_rm_stage[valid]

        v_pb_interp = np.interp(t_rel_rm, t_rel_pb, pb_sub["Terminal voltage [V]"].entries)
        i_pb_interp = np.interp(t_rel_rm, t_rel_pb, pb_sub["Current [A]"].entries)

        t_all.append(t_rm_stage)
        v_err_all.append((v_rm_stage - v_pb_interp) * 1e3)
        i_err_all.append((i_rm_stage - i_pb_interp) * 1e3)

    return np.concatenate(t_all), np.concatenate(v_err_all), np.concatenate(i_err_all)


def plot_cp7_experiment():
    print("Generating CP7: PyBaMM-style Experiment CCCV Recipe comparison + error + runtime plots...")
    recipe = [
        "Discharge at 1C until 3.6 V",
        "Rest for 300 seconds",
        "Charge at 1C until 4.0 V",
        "Hold at 4.0 V for 600 seconds",
    ]

    # 1. SPM Experiment
    Nr_spm = 8

    def make_pb_spm_exp():
        m = pybamm.lithium_ion.SPM()
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr_spm, "r_p": Nr_spm})
        return pybamm.Simulation(
            m,
            experiment=pybamm.Experiment(recipe, period="2 seconds"),
            var_pts=vp,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
        )

    sol_spm, pb_b_spm, pb_s_spm = _time_pybamm(make_pb_spm_exp, lambda s: s.solve())

    rm_spm, rm_c_spm, rm_s_spm = _time_rumoca(
        lambda: compile_experiment(
            cell_class="RuBaMM.Cells.SPM",
            experiment=Experiment(recipe),
            cell_params=f"Nr = {Nr_spm}",
            model_name="PlotSPMExp",
        ),
        lambda m: m.simulate(
            t=(0.0, float(sol_spm.t[-1])),
            dt=5.0,
            config=rm.SimConfig(rtol=1e-8, atol=1e-8),
        ),
    )
    rec_spm_exp = _record_timing("CP7: SPM 4-Stage CCCV", pb_b_spm, pb_s_spm, rm_c_spm, rm_s_spm)
    t_err_spm, v_err_spm, i_err_spm = _compute_stage_aligned_errors(sol_spm, rm_spm)

    # 2. DFN Experiment
    Nr_dfn = 6
    Nn, Ns, Np = 3, 2, 3

    def make_pb_dfn_exp():
        m = pybamm.lithium_ion.DFN()
        vp = m.default_var_pts.copy()
        vp.update({"r_n": Nr_dfn, "r_p": Nr_dfn, "x_n": Nn, "x_s": Ns, "x_p": Np})
        return pybamm.Simulation(
            m,
            experiment=pybamm.Experiment(recipe, period="2 seconds"),
            var_pts=vp,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
        )

    sol_dfn, pb_b_dfn, pb_s_dfn = _time_pybamm(make_pb_dfn_exp, lambda s: s.solve())

    rm_dfn, rm_c_dfn, rm_s_dfn = _time_rumoca(
        lambda: compile_experiment(
            cell_class="RuBaMM.Cells.DFN",
            experiment=Experiment(recipe),
            cell_params=f"Nr = {Nr_dfn}, Nn = {Nn}, Ns = {Ns}, Np = {Np}",
            model_name="PlotDFNExp",
        ),
        lambda m: m.simulate(
            t=(0.0, float(sol_dfn.t[-1])),
            dt=10.0,
            config=rm.SimConfig(rtol=1e-8, atol=1e-8),
        ),
    )
    rec_dfn_exp = _record_timing("CP7: DFN 4-Stage CCCV", pb_b_dfn, pb_s_dfn, rm_c_dfn, rm_s_dfn)
    t_err_dfn, v_err_dfn, i_err_dfn = _compute_stage_aligned_errors(sol_dfn, rm_dfn)

    fig, pairs, ax_rt = _make_paired_figure_with_runtime(
        "Checkpoint 7: Multi-Stage CCCV Experiment Recipe — Trajectories, Stage-Aligned Errors & Runtimes"
    )

    # Pair 0: SPM Voltage
    ax, ax_e = pairs[0]
    ax.plot(sol_spm.t, sol_spm["Terminal voltage [V]"].entries, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_spm.time, rm_spm["cell.v"], "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title(f"SPM Experiment: Terminal Voltage (Max |ΔV| = {np.max(np.abs(v_err_spm)):.3f} mV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_err_spm, v_err_spm, "m-", lw=1.3, label="Stage-Aligned ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [mV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 1: SPM Current
    ax, ax_e = pairs[1]
    ax.plot(sol_spm.t, sol_spm["Current [A]"].entries, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_spm.time, rm_spm["cell.i_cell"], "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title(f"SPM Experiment: Cell Current (Max |ΔI| = {np.max(np.abs(i_err_spm)):.3f} mA)")
    ax.set_ylabel("Current [A]")
    ax.legend()
    ax_e.plot(t_err_spm, i_err_spm, "m-", lw=1.3, label="Stage-Aligned ΔI = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔI [mA]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 2: DFN Voltage
    ax, ax_e = pairs[2]
    ax.plot(sol_dfn.t, sol_dfn["Terminal voltage [V]"].entries, "k-", lw=2, label="PyBaMM DFN")
    ax.plot(rm_dfn.time, rm_dfn["cell.v"], "r--", lw=1.8, label="RuBaMM DFN")
    ax.set_title(f"DFN Experiment: Terminal Voltage (Max |ΔV| = {np.max(np.abs(v_err_dfn)):.3f} mV)")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    ax_e.plot(t_err_dfn, v_err_dfn, "m-", lw=1.3, label="Stage-Aligned ΔV = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔV [mV]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    # Pair 3: DFN Current
    ax, ax_e = pairs[3]
    ax.plot(sol_dfn.t, sol_dfn["Current [A]"].entries, "k-", lw=2, label="PyBaMM DFN")
    ax.plot(rm_dfn.time, rm_dfn["cell.i_cell"], "r--", lw=1.8, label="RuBaMM DFN")
    ax.set_title(f"DFN Experiment: Cell Current (Max |ΔI| = {np.max(np.abs(i_err_dfn)):.3f} mA)")
    ax.set_ylabel("Current [A]")
    ax.legend()
    ax_e.plot(t_err_dfn, i_err_dfn, "m-", lw=1.3, label="Stage-Aligned ΔI = RuBaMM - PyBaMM")
    ax_e.set_ylabel("Error ΔI [mA]")
    ax_e.set_xlabel("Time [s]")
    ax_e.legend(loc="upper right")

    _plot_runtime_bars(ax_rt, [rec_spm_exp, rec_dfn_exp])
    fig.savefig(PLOTS_DIR / "cp7_experiment_cccv.png")
    plt.close(fig)


def plot_overall_runtime_summary():
    print("Generating overall runtime comparison summary bar chart...")
    fig, (ax_solve, ax_total) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        "RuBaMM (Rumoca) vs PyBaMM (IDAKLU) Runtime Benchmark Summary Across All Checkpoints",
        fontweight="bold",
        fontsize=13,
    )

    labels = [r["label"] for r in TIMING_RECORDS]
    y = np.arange(len(labels))
    height = 0.36

    pb_solve = np.array([r["pb_solve_ms"] for r in TIMING_RECORDS])
    rm_sim = np.array([r["rm_sim_ms"] for r in TIMING_RECORDS])

    pb_total = np.array([r["pb_total_ms"] for r in TIMING_RECORDS])
    rm_total = np.array([r["rm_total_ms"] for r in TIMING_RECORDS])

    # Left panel: Pure Solve / Simulate Time [ms] (log scale so both 1ms and 5000ms are clear)
    b1 = ax_solve.barh(y - height / 2, pb_solve, height, color="#2b2b2b", label="PyBaMM Solve [ms]")
    b2 = ax_solve.barh(y + height / 2, rm_sim, height, color="#e63946", label="RuBaMM Simulate [ms]")
    ax_solve.set_xscale("log")
    ax_solve.set_yticks(y)
    ax_solve.set_yticklabels(labels, fontweight="bold")
    ax_solve.invert_yaxis()
    ax_solve.set_xlabel("Warm Solver Execution Time [ms] (log scale)")
    ax_solve.set_title("Pure Solver Time: PyBaMM solve() vs RuBaMM simulate()", fontweight="bold")
    ax_solve.set_xlim(left=0.3, right=max(np.max(pb_solve), np.max(rm_sim)) * 8.0)
    for bar in list(b1) + list(b2):
        w = bar.get_width()
        ax_solve.annotate(
            f" {w:.1f} ms",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
        )
    ax_solve.legend(loc="upper right")

    # Right panel: Total Wall Time (Build/Compile + Solve/Simulate) [ms]
    b3 = ax_total.barh(y - height / 2, pb_total, height, color="#7f8c8d", label="PyBaMM Total (Build+Solve) [ms]")
    b4 = ax_total.barh(y + height / 2, rm_total, height, color="#f4a261", label="RuBaMM Total (Compile+Sim) [ms]")
    ax_total.set_xscale("log")
    ax_total.set_yticks(y)
    ax_total.set_yticklabels(["" for _ in labels])
    ax_total.invert_yaxis()
    ax_total.set_xlabel("Total Wall Time: Setup/Compile + Solve [ms] (log scale)")
    ax_total.set_title("End-to-End Time: Build/Compile + Solve", fontweight="bold")
    ax_total.set_xlim(left=1.0, right=max(np.max(pb_total), np.max(rm_total)) * 8.0)
    for bar in list(b3) + list(b4):
        w = bar.get_width()
        ax_total.annotate(
            f" {w:.1f} ms",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
        )
    ax_total.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "runtime_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_cp1_thevenin_ecm()
    plot_cp2_particle_ocp()
    plot_cp3_spm()
    plot_cp4_spme()
    plot_cp5_dfn()
    plot_cp6_acausal_coupling()
    plot_cp7_experiment()
    plot_overall_runtime_summary()
    print(f"All comparison plots with explicit error panels and runtime bar charts saved to {PLOTS_DIR}")
