"""Checkpoint 6 equivalence & subsystem coupling tests: Acausal Coupling of SPM, SPMe, and DFN in Rumoca."""

import numpy as np
import pybamm
import rumoca as rm
from rubamm import compile_system


def test_cp6_dfn_constant_voltage_acausal_coupling():
    """Verify connecting RuBaMM.Cells.DFN acausally to a VoltageSource matches PyBaMM voltage-controlled DFN."""
    Nr = 6
    Nn, Ns, Np = 3, 2, 3
    V_hold = 3.65

    pybamm_dfn = pybamm.lithium_ion.DFN(options={"operating mode": "voltage"})
    vp = pybamm_dfn.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})
    params = pybamm_dfn.default_parameter_values.copy()
    params["Voltage function [V]"] = V_hold

    t_eval = np.linspace(0.0, 600.0, 61)
    solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    sim = pybamm.Simulation(
        pybamm_dfn, parameter_values=params, var_pts=vp, solver=solver
    )
    sol = sim.solve(t_eval, t_interp=t_eval)

    # Connect exactly the same RuBaMM.Cells.DFN component to an external VoltageSource
    system_mo = f"""
    model TestDFNConstantVoltage
      RuBaMM.Cells.DFN cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.VoltageSource vsrc(V_set = {V_hold});
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, vsrc.p);
      connect(cell.n, vsrc.n);
      connect(cell.n, gnd.p);
    end TestDFNConstantVoltage;
    """
    rm_model = compile_system(system_mo, model_name="TestDFNConstantVoltage")
    rm_res = rm_model.simulate(
        t=(0.0, 600.0),
        dt=10.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    # Terminal voltage is held at V_hold
    np.testing.assert_allclose(rm_res["cell.v"], V_hold, rtol=1e-10, atol=1e-10)

    # Emergent current i_cell(t) solved algebraically by Rumoca matches PyBaMM
    np.testing.assert_allclose(
        rm_res["cell.i_cell"][1:],
        sol["Current [A]"].entries[1:],
        rtol=2e-3,
        atol=2e-3,
    )


def test_cp6_dfn_resistive_load_acausal_coupling():
    """Verify connecting RuBaMM.Cells.DFN across a passive Resistor load R_load = 5.0 Ohm matches PyBaMM."""
    Nr = 6
    Nn, Ns, Np = 3, 2, 3
    R_load = 5.0

    # In PyBaMM, a passive resistor load is modeled via custom algebraic operating mode: I - V / R = 0
    def resistor_load_residual(variables):
        I = variables["Current [A]"]
        V = variables["Terminal voltage [V]"]
        return I - V / R_load

    pybamm_dfn = pybamm.lithium_ion.DFN(
        options={"operating mode": resistor_load_residual}
    )
    vp = pybamm_dfn.default_var_pts.copy()
    vp.update({"r_n": Nr, "r_p": Nr, "x_n": Nn, "x_s": Ns, "x_p": Np})

    t_eval = np.linspace(0.0, 1200.0, 61)
    solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8)
    sim = pybamm.Simulation(pybamm_dfn, var_pts=vp, solver=solver)
    sol = sim.solve(t_eval, t_interp=t_eval)

    # In RuBaMM / Modelica, we simply connect the cell pins to a Resistor!
    system_mo = f"""
    model TestDFNResistorLoad
      RuBaMM.Cells.DFN cell(Nr = {Nr}, Nn = {Nn}, Ns = {Ns}, Np = {Np});
      RuBaMM.Interfaces.Resistor load(R = {R_load});
      RuBaMM.Interfaces.Ground gnd;
    equation
      connect(cell.p, load.p);
      connect(cell.n, load.n);
      connect(cell.n, gnd.p);
    end TestDFNResistorLoad;
    """
    rm_model = compile_system(system_mo, model_name="TestDFNResistorLoad")
    rm_res = rm_model.simulate(
        t=(0.0, 1200.0),
        dt=20.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    np.testing.assert_allclose(
        rm_res["cell.v"],
        sol["Terminal voltage [V]"].entries,
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        rm_res["cell.i_cell"],
        sol["Current [A]"].entries,
        rtol=1e-4,
        atol=1e-4,
    )


def test_cp6_parallel_pack_balancing_and_discharge():
    """Verify two electrochemical cells at different initial SOCs connected in parallel balance and discharge together."""
    system_mo = """
    model TestParallelCells
      import P = RuBaMM.Parameters.Marquis2019;

      // Cell 1 at higher initial state of charge (c_n_init = 20000), Cell 2 at lower initial state of charge (c_n_init = 15000)
      RuBaMM.Cells.SPM cell1(Nr = 6, c_n_init = 20000.0, c_p_init = 30730.0);
      RuBaMM.Cells.SPM cell2(Nr = 6, c_n_init = 15000.0, c_p_init = 36000.0);

      // Small interconnect resistances on each parallel branch + external pack current load
      RuBaMM.Interfaces.Resistor r_bus1(R = 0.005);
      RuBaMM.Interfaces.Resistor r_bus2(R = 0.005);
      RuBaMM.Interfaces.CurrentSource pack_load(I_load = 1.0);
      RuBaMM.Interfaces.Ground gnd;
    equation
      // Connect cell1 -> r_bus1 -> pack positive bus
      connect(cell1.p, r_bus1.p);
      connect(r_bus1.n, pack_load.p);

      // Connect cell2 -> r_bus2 -> pack positive bus
      connect(cell2.p, r_bus2.p);
      connect(r_bus2.n, pack_load.p);

      // Common negative bus grounded
      connect(cell1.n, gnd.p);
      connect(cell2.n, gnd.p);
      connect(pack_load.n, gnd.p);
    end TestParallelCells;
    """
    rm_model = compile_system(system_mo, model_name="TestParallelCells")
    rm_res = rm_model.simulate(
        t=(0.0, 1000.0),
        dt=10.0,
        config=rm.SimConfig(rtol=1e-8, atol=1e-8),
    )

    i1 = rm_res["cell1.i_cell"]
    i2 = rm_res["cell2.i_cell"]

    # Kirchhoff's current law holds across the parallel branches at all times: i1 + i2 == I_load = 1.0 A
    np.testing.assert_allclose(i1 + i2, 1.0, rtol=1e-10, atol=1e-10)

    # Initially, higher-SOC cell1 supplies much more current than lower-SOC cell2
    assert i1[0] > 2.0 * i2[0]

    # As cell1 depletes faster than cell2, their surface concentrations and currents converge
    assert abs(i1[-1] - i2[-1]) < abs(i1[0] - i2[0])
