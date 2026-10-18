"""Generate visual equivalence comparison plots across all RuBaMM vs PyBaMM checkpoints."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pybamm
import rumoca as rm
from rubamm import Experiment, compile_experiment, compile_system

REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO_ROOT / "docs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _ocv_thevenin_pybamm(z):
    return 3.5 + 0.55 * z - 0.08 / (z + 0.05) + 0.08 * pybamm.exp(-25.0 * (1.0 - z))


def plot_cp1_thevenin_ecm():
    print("Generating CP1: Thevenin ECM comparison plot...")
    # 1. Constant Current
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
    t_cc = np.linspace(0.0, 1800.0, 181)
    sol_cc = pybamm.Simulation(pybamm_model, parameter_values=params).solve(t_cc)

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
    rm_cc = compile_system(system_cc, model_name="PlotTheveninCC").simulate(
        t=(0.0, 1800.0), dt=10.0
    )

    # 2. Dynamic Sinusoidal Current
    params_dyn = params.copy()
    params_dyn.update(
        {
            "Initial SoC": 0.6,
            "Current function [A]": lambda t: 50.0 + 40.0 * pybamm.sin(0.02 * t),
            "R0 [Ohm]": 0.0015,
            "R1 [Ohm]": 0.0008,
            "C1 [F]": 5000.0,
            "Entropic change [V/K]": 0.0,
        }
    )
    t_dyn = np.linspace(0.0, 600.0, 121)
    sol_dyn = pybamm.Simulation(pybamm_model, parameter_values=params_dyn).solve(t_dyn)

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
    rm_dyn = compile_system(system_dyn, model_name="PlotTheveninDyn").simulate(
        t=(0.0, 600.0), dt=5.0
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Checkpoint 1: Thevenin 1-RC Equivalent Circuit Model (PyBaMM vs RuBaMM)", fontweight="bold")

    # (0, 0) CC Voltage & Error
    ax = axes[0, 0]
    v_pb = sol_cc["Voltage [V]"](t_cc)
    v_rm = rm_cc["cell.v"]
    ax.plot(t_cc, v_pb, "k-", lw=2, label="PyBaMM")
    ax.plot(rm_cc.time, v_rm, "r--", lw=1.8, label="RuBaMM (Rumoca)")
    ax.set_title(f"1C Constant-Current Voltage (Max |ΔV| = {np.max(np.abs(v_rm - v_pb))*1e6:.2f} µV)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal Voltage [V]")
    ax.legend()

    # (0, 1) CC Temperature & SoC
    ax = axes[0, 1]
    t_pb_c = sol_cc["Cell temperature [degC]"](t_cc)
    t_rm_c = rm_cc["cell.T_cell_C"]
    ax.plot(t_cc, t_pb_c, "k-", lw=2, label="PyBaMM T_cell")
    ax.plot(rm_cc.time, t_rm_c, "r--", lw=1.8, label="RuBaMM T_cell")
    ax.set_title(f"Lumped Cell Temperature (Max |ΔT| = {np.max(np.abs(t_rm_c - t_pb_c))*1e3:.2f} mK)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.legend()

    # (1, 0) Dynamic Sinusoidal Voltage
    ax = axes[1, 0]
    vd_pb = sol_dyn["Voltage [V]"](t_dyn)
    vd_rm = rm_dyn["cell.v"]
    ax.plot(t_dyn, vd_pb, "k-", lw=2, label="PyBaMM")
    ax.plot(rm_dyn.time, vd_rm, "r--", lw=1.8, label="RuBaMM (Rumoca)")
    ax.set_title(f"Sinusoidal Load Voltage (Max |ΔV| = {np.max(np.abs(vd_rm - vd_pb))*1e6:.2f} µV)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal Voltage [V]")
    ax.legend()

    # (1, 1) RC Overpotential v_rc1
    ax = axes[1, 1]
    vrc_pb = sol_cc["Element-1 overpotential [V]"](t_cc) * 1e3
    vrc_rm = rm_cc["cell.v_rc1"] * 1e3
    ax.plot(t_cc, vrc_pb, "k-", lw=2, label="PyBaMM v_rc1")
    ax.plot(rm_cc.time, vrc_rm, "r--", lw=1.8, label="RuBaMM v_rc1")
    ax.set_title("1-RC Polarization Overpotential")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Overpotential [mV]")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cp1_thevenin_ecm.png")
    plt.close(fig)


def plot_cp2_particle_ocp():
    print("Generating CP2: Marquis2019 OCP & SphericalParticle FVM comparison plot...")
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
    rm_fn = compile_system(system_fn, model_name="PlotMarquisFn").simulate(
        t=(0.0, 0.96), dt=0.01
    )
    sto = rm_fn["sto"]
    c_e = rm_fn["c_e"]

    def _ev(v):
        return float(v.evaluate()) if hasattr(v, "evaluate") else float(v)

    u_n_pb = np.array([_ev(u_n_fn(s)) for s in sto])
    u_p_pb = np.array([_ev(u_p_fn(s)) for s in sto])
    k_e_pb = np.array([_ev(k_e_fn(c, 298.15)) for c in c_e])

    # Spherical Particle FVM
    Nr = 10
    pybamm_spm = pybamm.lithium_ion.SPM()
    vp = pybamm_spm.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr})
    t_eval = np.linspace(0.0, 3600.0, 181)
    sol = pybamm.Simulation(
        pybamm_spm, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    ).solve(t_eval, t_interp=t_eval)

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
    rm_part = compile_system(system_part, model_name="PlotParticles").simulate(
        t=(0.0, 3600.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Checkpoint 2: Marquis2019 Functions & 1D Spherical Particle FVM (PyBaMM vs RuBaMM)", fontweight="bold")

    ax = axes[0, 0]
    ax.plot(sto, u_p_pb, "k-", lw=2, label="PyBaMM U_p(θ)")
    ax.plot(sto, rm_fn["u_p"], "r--", lw=1.8, label="RuBaMM U_p(θ)")
    ax.plot(sto, u_n_pb, "b-", lw=2, label="PyBaMM U_n(θ)")
    ax.plot(sto, rm_fn["u_n"], "c--", lw=1.8, label="RuBaMM U_n(θ)")
    ax.set_title("Open-Circuit Potentials U_n(θ) and U_p(θ)")
    ax.set_xlabel("Stoichiometry θ [-]")
    ax.set_ylabel("OCP [V]")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(c_e, k_e_pb, "k-", lw=2, label="PyBaMM κ_e(c_e)")
    ax.plot(c_e, rm_fn["k_e"], "r--", lw=1.8, label="RuBaMM κ_e(c_e)")
    ax.set_title("Capiglia1999 Electrolyte Conductivity κ_e(c_e)")
    ax.set_xlabel("Electrolyte Concentration c_e [mol/m³]")
    ax.set_ylabel("Conductivity [S/m]")
    ax.legend()

    ax = axes[1, 0]
    cs_n_pb = sol["X-averaged negative particle surface concentration [mol.m-3]"].entries
    cs_p_pb = sol["X-averaged positive particle surface concentration [mol.m-3]"].entries
    ax.plot(t_eval, cs_p_pb, "k-", lw=2, label="PyBaMM Pos Surface c_s")
    ax.plot(rm_part.time, rm_part["pos_particle.c_s_surf"], "r--", lw=1.8, label="RuBaMM Pos Surface c_s")
    ax.plot(t_eval, cs_n_pb, "b-", lw=2, label="PyBaMM Neg Surface c_s")
    ax.plot(rm_part.time, rm_part["neg_particle.c_s_surf"], "c--", lw=1.8, label="RuBaMM Neg Surface c_s")
    ax.set_title("Particle Surface Concentrations vs Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration [mol/m³]")
    ax.legend()

    ax = axes[1, 1]
    r_centers = (np.arange(1, Nr + 1) - 0.5) / Nr
    c_n_rad_pb = sol["X-averaged negative particle concentration [mol.m-3]"].entries
    for t_idx, label_t in [(0, "t=0s"), (45, "t=900s"), (90, "t=1800s"), (180, "t=3600s")]:
        ax.plot(r_centers, c_n_rad_pb[:, t_idx], "-", lw=2, label=f"PyBaMM {label_t}")
        rm_rad = [rm_part[f"neg_particle.c_s[{i}]"][t_idx] for i in range(1, Nr + 1)]
        ax.plot(r_centers, rm_rad, "o--", ms=4, label=f"RuBaMM {label_t}")
    ax.set_title("Negative Particle Radial FVM Shells c_s(r, t)")
    ax.set_xlabel("Normalized Radius r / R_n [-]")
    ax.set_ylabel("Concentration [mol/m³]")
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cp2_particle_ocp.png")
    plt.close(fig)


def plot_cp3_spm():
    print("Generating CP3: Single Particle Model (SPM) comparison plot...")
    Nr = 10
    pybamm_spm = pybamm.lithium_ion.SPM()
    vp = pybamm_spm.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr})

    t_cc = np.linspace(0.0, 3400.0, 171)
    sol_cc = pybamm.Simulation(
        pybamm_spm, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    ).solve(t_cc, t_interp=t_cc)

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
    rm_cc = compile_system(system_cc, model_name="PlotSPM1C").simulate(
        t=(0.0, 3400.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

    # Dynamic sinusoidal current
    params_dyn = pybamm_spm.default_parameter_values.copy()
    params_dyn["Current function [A]"] = lambda t: 0.4 + 0.6 * pybamm.sin(0.01 * t)
    t_dyn = np.linspace(0.0, 1200.0, 121)
    sol_dyn = pybamm.Simulation(
        pybamm_spm, parameter_values=params_dyn, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    ).solve(t_dyn, t_interp=t_dyn)

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
    rm_dyn = compile_system(system_dyn, model_name="PlotSPMDyn").simulate(
        t=(0.0, 1200.0), dt=10.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Checkpoint 3: Single Particle Model — SPM (PyBaMM vs RuBaMM)", fontweight="bold")

    ax = axes[0, 0]
    v_pb = sol_cc["Terminal voltage [V]"].entries
    v_rm = rm_cc["cell.v"]
    ax.plot(t_cc, v_pb, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_cc.time, v_rm, "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title(f"1C Discharge Voltage (Max |ΔV| = {np.max(np.abs(v_rm - v_pb))*1e3:.3f} mV)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal Voltage [V]")
    ax.legend()

    ax = axes[0, 1]
    eta_n_pb = sol_cc["X-averaged negative electrode reaction overpotential [V]"].entries * 1e3
    eta_p_pb = sol_cc["X-averaged positive electrode reaction overpotential [V]"].entries * 1e3
    ax.plot(t_cc, eta_n_pb, "b-", lw=2, label="PyBaMM η_n")
    ax.plot(rm_cc.time, rm_cc["cell.eta_n"] * 1e3, "c--", lw=1.8, label="RuBaMM η_n")
    ax.plot(t_cc, eta_p_pb, "k-", lw=2, label="PyBaMM η_p")
    ax.plot(rm_cc.time, rm_cc["cell.eta_p"] * 1e3, "r--", lw=1.8, label="RuBaMM η_p")
    ax.set_title("Butler-Volmer Reaction Overpotentials η_n & η_p")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Overpotential [mV]")
    ax.legend()

    ax = axes[1, 0]
    vd_pb = sol_dyn["Terminal voltage [V]"].entries
    vd_rm = rm_dyn["cell.v"]
    ax.plot(t_dyn, vd_pb, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_dyn.time, vd_rm, "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title(f"Dynamic Sinusoidal Load Voltage (Max |ΔV| = {np.max(np.abs(vd_rm - vd_pb))*1e3:.3f} mV)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal Voltage [V]")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t_cc, (v_rm - v_pb) * 1e3, "m-", lw=1.5, label="1C Voltage Error (RuBaMM - PyBaMM)")
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_title("1C Discharge Voltage Discrepancy")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error ΔV [mV]")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cp3_spm.png")
    plt.close(fig)


def plot_cp4_spme():
    print("Generating CP4: SPMe & 1D Electrolyte FVM comparison plot...")
    Nr = 8
    Nn, Ns, Np = 4, 3, 4
    pybamm_spme = pybamm.lithium_ion.SPMe()
    vp = pybamm_spme.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})

    t_eval = np.linspace(0.0, 3200.0, 161)
    sol = pybamm.Simulation(
        pybamm_spme, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    ).solve(t_eval, t_interp=t_eval)

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
    rm_res = compile_system(system_mo, model_name="PlotSPMe1C").simulate(
        t=(0.0, 3200.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Checkpoint 4: Single Particle Model with Electrolyte — SPMe (PyBaMM vs RuBaMM)", fontweight="bold")

    ax = axes[0, 0]
    v_pb = sol["Terminal voltage [V]"].entries
    v_rm = rm_res["cell.v"]
    ax.plot(t_eval, v_pb, "k-", lw=2, label="PyBaMM SPMe")
    ax.plot(rm_res.time, v_rm, "r--", lw=1.8, label="RuBaMM SPMe")
    ax.set_title(f"1C Discharge Voltage (Max |ΔV| = {np.max(np.abs(v_rm - v_pb))*1e3:.3f} mV)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal Voltage [V]")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t_eval, sol["X-averaged negative electrolyte concentration [mol.m-3]"].entries, "b-", lw=2, label="PyBaMM Neg c_e_avg")
    ax.plot(rm_res.time, rm_res["cell.electrolyte.c_e_n_avg"], "c--", lw=1.8, label="RuBaMM Neg c_e_avg")
    ax.plot(t_eval, sol["X-averaged separator electrolyte concentration [mol.m-3]"].entries, "g-", lw=2, label="PyBaMM Sep c_e_avg")
    ax.plot(rm_res.time, rm_res["cell.electrolyte.c_e_s_avg"], "y--", lw=1.8, label="RuBaMM Sep c_e_avg")
    ax.plot(t_eval, sol["X-averaged positive electrolyte concentration [mol.m-3]"].entries, "k-", lw=2, label="PyBaMM Pos c_e_avg")
    ax.plot(rm_res.time, rm_res["cell.electrolyte.c_e_p_avg"], "r--", lw=1.8, label="RuBaMM Pos c_e_avg")
    ax.set_title("Domain-Averaged Electrolyte Concentrations")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("c_e [mol/m³]")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    c_e_pb = sol["Electrolyte concentration [mol.m-3]"].entries
    x_idx = np.arange(1, Nn + Ns + Np + 1)
    for t_i, lbl in [(0, "t=0s"), (25, "t=500s"), (80, "t=1600s"), (160, "t=3200s")]:
        ax.plot(x_idx, c_e_pb[:, t_i], "-", lw=2, label=f"PyBaMM {lbl}")
        rm_prof = (
            [rm_res[f"cell.electrolyte.c_e_n[{i}]"][t_i] for i in range(1, Nn + 1)]
            + [rm_res[f"cell.electrolyte.c_e_s[{i}]"][t_i] for i in range(1, Ns + 1)]
            + [rm_res[f"cell.electrolyte.c_e_p[{i}]"][t_i] for i in range(1, Np + 1)]
        )
        ax.plot(x_idx, rm_prof, "o--", ms=4, label=f"RuBaMM {lbl}")
    ax.axvline(Nn + 0.5, color="gray", ls=":", lw=1)
    ax.axvline(Nn + Ns + 0.5, color="gray", ls=":", lw=1)
    ax.set_title("Spatial Electrolyte Profile c_e(x, t) [Neg | Sep | Pos]")
    ax.set_xlabel("FVM Control Volume Index [-]")
    ax.set_ylabel("c_e [mol/m³]")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    eta_c_pb = sol["X-averaged concentration overpotential [V]"].entries * 1e3
    ohm_e_pb = sol["X-averaged electrolyte ohmic losses [V]"].entries * 1e3
    ax.plot(t_eval, eta_c_pb, "k-", lw=2, label="PyBaMM η_conc")
    ax.plot(rm_res.time, rm_res["cell.eta_conc"] * 1e3, "r--", lw=1.8, label="RuBaMM η_conc")
    ax.plot(t_eval, ohm_e_pb, "b-", lw=2, label="PyBaMM Δφ_e,ohm")
    ax.plot(rm_res.time, rm_res["cell.dphi_e_ohm"] * 1e3, "c--", lw=1.8, label="RuBaMM Δφ_e,ohm")
    ax.set_title("Electrolyte Concentration & Ohmic Overpotentials")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Overpotential [mV]")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cp4_spme.png")
    plt.close(fig)


def plot_cp5_dfn():
    print("Generating CP5: Full P2D Doyle-Fuller-Newman (DFN) comparison plot...")
    Nr = 6
    Nn, Ns, Np = 3, 2, 3
    pybamm_dfn = pybamm.lithium_ion.DFN()
    vp = pybamm_dfn.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})

    t_eval = np.linspace(0.0, 3000.0, 151)
    sol = pybamm.Simulation(
        pybamm_dfn, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    ).solve(t_eval, t_interp=t_eval)

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
    rm_res = compile_system(system_mo, model_name="PlotDFN1C").simulate(
        t=(0.0, 3000.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Checkpoint 5: Doyle-Fuller-Newman P2D Model — DFN (PyBaMM vs RuBaMM)", fontweight="bold")

    ax = axes[0, 0]
    v_pb = sol["Terminal voltage [V]"].entries
    v_rm = rm_res["cell.v"]
    ax.plot(t_eval, v_pb, "k-", lw=2, label="PyBaMM DFN")
    ax.plot(rm_res.time, v_rm, "r--", lw=1.8, label="RuBaMM DFN")
    ax.set_title(f"1C Terminal Voltage (Max |ΔV| = {np.max(np.abs(v_rm - v_pb))*1e3:.3f} mV)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Terminal Voltage [V]")
    ax.legend()

    ax = axes[0, 1]
    c_e_pb = sol["Electrolyte concentration [mol.m-3]"].entries
    x_idx = np.arange(1, Nn + Ns + Np + 1)
    for t_i, lbl in [(10, "t=200s"), (50, "t=1000s"), (150, "t=3000s")]:
        ax.plot(x_idx, c_e_pb[:, t_i], "-", lw=2, label=f"PyBaMM {lbl}")
        rm_prof = (
            [rm_res[f"cell.electrolyte.c_e_n[{i}]"][t_i] for i in range(1, Nn + 1)]
            + [rm_res[f"cell.electrolyte.c_e_s[{i}]"][t_i] for i in range(1, Ns + 1)]
            + [rm_res[f"cell.electrolyte.c_e_p[{i}]"][t_i] for i in range(1, Np + 1)]
        )
        ax.plot(x_idx, rm_prof, "o--", ms=4, label=f"RuBaMM {lbl}")
    ax.axvline(Nn + 0.5, color="gray", ls=":", lw=1)
    ax.axvline(Nn + Ns + 0.5, color="gray", ls=":", lw=1)
    ax.set_title("Spatial Electrolyte Concentration c_e(x, t)")
    ax.set_xlabel("FVM Control Volume Index [-]")
    ax.set_ylabel("c_e [mol/m³]")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    phi_e_pb = sol["Electrolyte potential [V]"].entries
    for t_i, lbl in [(10, "t=200s"), (75, "t=1500s"), (150, "t=3000s")]:
        ax.plot(x_idx, phi_e_pb[:, t_i], "-", lw=2, label=f"PyBaMM {lbl}")
        rm_phi = (
            [rm_res[f"cell.phi_e_n[{i}]"][t_i] for i in range(1, Nn + 1)]
            + [rm_res[f"cell.phi_e_s[{i}]"][t_i] for i in range(1, Ns + 1)]
            + [rm_res[f"cell.phi_e_p[{i}]"][t_i] for i in range(1, Np + 1)]
        )
        ax.plot(x_idx, rm_phi, "o--", ms=4, label=f"RuBaMM {lbl}")
    ax.axvline(Nn + 0.5, color="gray", ls=":", lw=1)
    ax.axvline(Nn + Ns + 0.5, color="gray", ls=":", lw=1)
    ax.set_title("Spatial Electrolyte Potential φ_e(x, t)")
    ax.set_xlabel("FVM Control Volume Index [-]")
    ax.set_ylabel("Potential φ_e [V]")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    j_n_pb = sol["Negative electrode interfacial current density [A.m-2]"].entries
    j_p_pb = sol["Positive electrode interfacial current density [A.m-2]"].entries
    ax.plot(t_eval, j_n_pb[0, :], "b-", lw=2, label="PyBaMM j_n[1] (CC side)")
    ax.plot(rm_res.time, rm_res["cell.j_n[1]"], "c--", lw=1.8, label="RuBaMM j_n[1]")
    ax.plot(t_eval, j_n_pb[-1, :], "g-", lw=2, label=f"PyBaMM j_n[{Nn}] (Sep side)")
    ax.plot(rm_res.time, rm_res[f"cell.j_n[{Nn}]"], "y--", lw=1.8, label=f"RuBaMM j_n[{Nn}]")
    ax.plot(t_eval, j_p_pb[0, :], "k-", lw=2, label="PyBaMM j_p[1] (Sep side)")
    ax.plot(rm_res.time, rm_res["cell.j_p[1]"], "r--", lw=1.8, label="RuBaMM j_p[1]")
    ax.set_title("Spatial Interfacial Reaction Current Density j(x, t)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reaction Current Density [A/m²]")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cp5_dfn.png")
    plt.close(fig)


def plot_cp6_acausal_coupling():
    print("Generating CP6: Acausal Subsystem Coupling comparison plot...")
    Nr = 6
    Nn, Ns, Np = 3, 2, 3

    # 1. Constant-Voltage Hold on DFN
    V_hold = 3.65
    pybamm_cv = pybamm.lithium_ion.DFN(options={"operating mode": "voltage"})
    vp = pybamm_cv.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})
    params_cv = pybamm_cv.default_parameter_values.copy()
    params_cv["Voltage function [V]"] = V_hold
    t_cv = np.linspace(0.0, 600.0, 61)
    sol_cv = pybamm.Simulation(
        pybamm_cv, parameter_values=params_cv, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    ).solve(t_cv, t_interp=t_cv)

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
    rm_cv = compile_system(sys_cv, model_name="PlotDFNCV").simulate(
        t=(0.0, 600.0), dt=10.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

    # 2. Passive Resistor Load on DFN
    R_load = 5.0
    pybamm_res = pybamm.lithium_ion.DFN(
        options={"operating mode": lambda v: v["Current [A]"] - v["Terminal voltage [V]"] / R_load}
    )
    t_res = np.linspace(0.0, 1200.0, 61)
    sol_res = pybamm.Simulation(
        pybamm_res, var_pts=vp, solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    ).solve(t_res, t_interp=t_res)

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
    rm_res = compile_system(sys_res, model_name="PlotDFNRes").simulate(
        t=(0.0, 1200.0), dt=20.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

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
    rm_par = compile_system(sys_par, model_name="PlotParallelCells").simulate(
        t=(0.0, 1000.0), dt=10.0, config=rm.SimConfig(rtol=1e-8, atol=1e-8)
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Checkpoint 6: Acausal Circuit & Multi-Cell Pack Coupling in Rumoca", fontweight="bold")

    ax = axes[0, 0]
    ax.plot(t_cv[1:], sol_cv["Current [A]"].entries[1:], "k-", lw=2, label="PyBaMM DFN (Voltage Mode)")
    ax.plot(rm_cv.time[1:], rm_cv["cell.i_cell"][1:], "r--", lw=1.8, label="RuBaMM DFN + VoltageSource")
    ax.set_title("DFN Coupled to Constant-Voltage Source (V = 3.65 V)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Emergent Cell Current [A]")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t_res, sol_res["Terminal voltage [V]"].entries, "k-", lw=2, label="PyBaMM V(t)")
    ax.plot(rm_res.time, rm_res["cell.v"], "r--", lw=1.8, label="RuBaMM V(t)")
    ax2 = ax.twinx()
    ax2.plot(t_res, sol_res["Current [A]"].entries, "b-", lw=1.5, label="PyBaMM I(t)")
    ax2.plot(rm_res.time, rm_res["cell.i_cell"], "c--", lw=1.5, label="RuBaMM I(t)")
    ax.set_title("DFN Coupled to Passive Resistor Load (R = 5.0 Ω)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax2.set_ylabel("Current [A]", color="b")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.plot(rm_par.time, rm_par["cell1.i_cell"], "r-", lw=2, label="Cell 1 Current (High Initial SoC)")
    ax.plot(rm_par.time, rm_par["cell2.i_cell"], "b-", lw=2, label="Cell 2 Current (Low Initial SoC)")
    ax.plot(rm_par.time, rm_par["cell1.i_cell"] + rm_par["cell2.i_cell"], "k--", lw=1.5, label="Total Pack Current (I1 + I2 = 1.0 A)")
    ax.set_title("Parallel 2-Cell Pack Current Sharing & Self-Balancing")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Branch Current [A]")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(rm_par.time, rm_par["cell1.neg_particle.c_s_surf"], "r-", lw=2, label="Cell 1 Neg Surface c_s")
    ax.plot(rm_par.time, rm_par["cell2.neg_particle.c_s_surf"], "b-", lw=2, label="Cell 2 Neg Surface c_s")
    ax.set_title("Parallel 2-Cell Pack Surface Concentration Convergence")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Anode Surface Concentration [mol/m³]")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cp6_acausal_coupling.png")
    plt.close(fig)


def plot_cp7_experiment():
    print("Generating CP7: PyBaMM-style Experiment CCCV Recipe comparison plot...")
    recipe = [
        "Discharge at 1C until 3.6 V",
        "Rest for 300 seconds",
        "Charge at 1C until 4.0 V",
        "Hold at 4.0 V for 600 seconds",
    ]

    # 1. SPM Experiment
    Nr_spm = 8
    pb_spm = pybamm.lithium_ion.SPM()
    vp_spm = pb_spm.default_var_pts.copy()
    vp_spm.update({"r_n": Nr_spm, "r_p": Nr_spm})
    sol_spm = pybamm.Simulation(
        pb_spm,
        experiment=pybamm.Experiment(recipe, period="5 seconds"),
        var_pts=vp_spm,
        solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
    ).solve()

    rm_spm = compile_experiment(
        cell_class="RuBaMM.Cells.SPM",
        experiment=Experiment(recipe),
        cell_params=f"Nr = {Nr_spm}",
        model_name="PlotSPMExp",
    ).simulate(
        t=(0.0, float(sol_spm.t[-1])),
        dt=10.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    # 2. DFN Experiment
    Nr_dfn = 6
    Nn, Ns, Np = 3, 2, 3
    pb_dfn = pybamm.lithium_ion.DFN()
    vp_dfn = pb_dfn.default_var_pts.copy()
    vp_dfn.update({"r_n": Nr_dfn, "r_p": Nr_dfn, "x_n": Nn, "x_s": Ns, "x_p": Np})
    sol_dfn = pybamm.Simulation(
        pb_dfn,
        experiment=pybamm.Experiment(recipe, period="5 seconds"),
        var_pts=vp_dfn,
        solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
    ).solve()

    rm_dfn = compile_experiment(
        cell_class="RuBaMM.Cells.DFN",
        experiment=Experiment(recipe),
        cell_params=f"Nr = {Nr_dfn}, Nn = {Nn}, Ns = {Ns}, Np = {Np}",
        model_name="PlotDFNExp",
    ).simulate(
        t=(0.0, float(sol_dfn.t[-1])),
        dt=15.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "Checkpoint 7: Multi-Stage CCCV Experiment Recipe (Discharge → Rest → Charge → CV Hold)",
        fontweight="bold",
    )

    # (0, 0) SPM Voltage
    ax = axes[0, 0]
    ax.plot(sol_spm.t, sol_spm["Terminal voltage [V]"].entries, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_spm.time, rm_spm["cell.v"], "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title("SPM Experiment: Terminal Voltage [V]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.legend()

    # (0, 1) SPM Current
    ax = axes[0, 1]
    ax.plot(sol_spm.t, sol_spm["Current [A]"].entries, "k-", lw=2, label="PyBaMM SPM")
    ax.plot(rm_spm.time, rm_spm["cell.i_cell"], "r--", lw=1.8, label="RuBaMM SPM")
    ax.set_title("SPM Experiment: Cell Current [A] (CC → Rest → CC → CV Taper)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.legend()

    # (1, 0) DFN Voltage
    ax = axes[1, 0]
    ax.plot(sol_dfn.t, sol_dfn["Terminal voltage [V]"].entries, "k-", lw=2, label="PyBaMM DFN")
    ax.plot(rm_dfn.time, rm_dfn["cell.v"], "r--", lw=1.8, label="RuBaMM DFN")
    ax.set_title("DFN Experiment: Terminal Voltage [V]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.legend()

    # (1, 1) DFN Current & Stage
    ax = axes[1, 1]
    ax.plot(sol_dfn.t, sol_dfn["Current [A]"].entries, "k-", lw=2, label="PyBaMM DFN Current")
    ax.plot(rm_dfn.time, rm_dfn["cell.i_cell"], "r--", lw=1.8, label="RuBaMM DFN Current")
    ax2 = ax.twinx()
    ax2.step(rm_dfn.time, rm_dfn["cycler.stage"], "b:", lw=1.5, where="post", label="RuBaMM Stage #")
    ax2.set_ylabel("Active Recipe Stage #", color="b")
    ax2.set_yticks([1, 2, 3, 4])
    ax.set_title("DFN Experiment: Cell Current [A] & Stage Transitions")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cp7_experiment_cccv.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_cp1_thevenin_ecm()
    plot_cp2_particle_ocp()
    plot_cp3_spm()
    plot_cp4_spme()
    plot_cp5_dfn()
    plot_cp6_acausal_coupling()
    plot_cp7_experiment()
    print(f"All comparison plots saved to {PLOTS_DIR}")
