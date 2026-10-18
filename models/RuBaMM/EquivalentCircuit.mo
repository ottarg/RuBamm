within RuBaMM;
package EquivalentCircuit
  "Equivalent Circuit Battery Models (Thevenin ECM)"

  function ocv_thevenin
    "Smooth analytical Open-Circuit Voltage [V] as a function of SoC z in [0, 1]"
    input Real z;
    output Real V_ocv;
  algorithm
    V_ocv := 3.5 + 0.55 * z - 0.08 / (z + 0.05) + 0.08 * exp(-25.0 * (1.0 - z));
  end ocv_thevenin;

  model Thevenin
    "1-RC Thevenin Equivalent Circuit Model with Lumped Thermal Dynamics (matching pybamm.equivalent_circuit.Thevenin)"
    extends Interfaces.TwoPin;

    parameter Real Q_Ah = 100.0 "Cell capacity [A.h]";
    parameter Real z_init = 0.5 "Initial SoC [-]";
    parameter Real R0 = 0.001 "Series resistance [Ohm]";
    parameter Real R1 = 0.0005 "RC branch resistance [Ohm]";
    parameter Real C1 = 10000.0 "RC branch capacitance [F]";
    parameter Real v_rc1_init = 0.0 "Element-1 initial overpotential [V]";

    // Thermal parameters (matching PyBaMM Thevenin defaults)
    parameter Real T_amb_K = 298.15 "Ambient temperature [K]";
    parameter Real T_init_K = 298.15 "Initial temperature [K]";
    parameter Real dUdT = 0.0 "Entropic change [V/K]";
    parameter Real c_th_cell = 1000.0 "Cell thermal mass [J/K]";
    parameter Real c_th_jig = 500.0 "Jig thermal mass [J/K]";
    parameter Real h_cell_jig = 10.0 "Cell-jig heat transfer coefficient [W/K]";
    parameter Real h_jig_air = 10.0 "Jig-air heat transfer coefficient [W/K]";

    Real z(start = z_init) "State of Charge (SoC) [-]";
    Real v_rc1(start = v_rc1_init) "Element-1 overpotential [V] (negative during discharge)";
    Real T_cell_C(start = T_init_K - 273.15) "Cell temperature [degC]";
    Real T_jig_C(start = T_init_K - 273.15) "Jig temperature [degC]";

    Real V_ocv(start = 3.7) "Open-circuit voltage [V]";
    Real v_r0(start = 0.0) "Element-0 overpotential [V]";
  equation
    // SoC evolution: der(SoC) = -i_cell / (3600 * Q_Ah)
    der(z) = -i_cell / (3600.0 * Q_Ah);

    // RC polarization branch: der(v_rc1) = -v_rc1 / (R1 * C1) - i_cell / C1
    der(v_rc1) = -v_rc1 / (R1 * C1) - i_cell / C1;

    // Lumped cell & jig thermal ODEs
    der(T_cell_C) = (
      (i_cell ^ 2) * R0
      - i_cell * v_rc1
      - i_cell * (273.15 + T_cell_C) * dUdT
      - h_cell_jig * (T_cell_C - T_jig_C)
    ) / c_th_cell;

    der(T_jig_C) = (
      h_cell_jig * (T_cell_C - T_jig_C)
      - h_jig_air * (T_jig_C - (T_amb_K - 273.15))
    ) / c_th_jig;

    // Terminal voltage equation
    V_ocv = ocv_thevenin(z);
    v_r0 = -i_cell * R0;
    v = V_ocv + v_r0 + v_rc1;
  end Thevenin;

end EquivalentCircuit;
