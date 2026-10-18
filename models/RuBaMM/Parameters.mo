within RuBaMM;
package Parameters
  "Standard Lithium-Ion Electrochemical Parameters and Material Functions"

  package Marquis2019
    "Graphite / LiCoO2 parameter set from Marquis et al. (2019) / Dualfoil (matching pybamm.ParameterValues('Marquis2019'))"

    // Universal physical constants (CODATA values used by PyBaMM)
    constant Real F = 96485.33212331001 "Faraday constant [C/mol]";
    constant Real R_gas = 8.31446261815324 "Ideal gas constant [J/(mol.K)]";
    constant Real T_ref = 298.15 "Reference temperature [K]";

    // Geometric parameters
    constant Real L_n = 1.0e-4 "Negative electrode thickness [m]";
    constant Real L_s = 2.5e-5 "Separator thickness [m]";
    constant Real L_p = 1.0e-4 "Positive electrode thickness [m]";
    constant Real H_elec = 0.137 "Electrode height [m]";
    constant Real W_elec = 0.207 "Electrode width [m]";
    constant Real A_cc = H_elec * W_elec "Current collector cross-sectional area [m^2]";

    // Solid particle parameters
    constant Real R_n = 1.0e-5 "Negative particle radius [m]";
    constant Real R_p = 1.0e-5 "Positive particle radius [m]";
    constant Real c_n_max = 24983.2619938437 "Maximum concentration in negative electrode [mol/m^3]";
    constant Real c_p_max = 51217.9257309275 "Maximum concentration in positive electrode [mol/m^3]";
    constant Real c_n_init = 19986.609595075 "Initial concentration in negative electrode [mol/m^3]";
    constant Real c_p_init = 30730.7554385565 "Initial concentration in positive electrode [mol/m^3]";
    constant Real D_s_n_ref = 3.9e-14 "Negative electrode solid diffusivity at T_ref [m^2/s]";
    constant Real D_s_p_ref = 1.0e-13 "Positive electrode solid diffusivity at T_ref [m^2/s]";
    constant Real eps_s_n = 0.6 "Negative electrode active material volume fraction [-]";
    constant Real eps_s_p = 0.5 "Positive electrode active material volume fraction [-]";
    constant Real a_n = 3.0 * eps_s_n / R_n "Negative electrode specific interfacial surface area [m^-1]";
    constant Real a_p = 3.0 * eps_s_p / R_p "Positive electrode specific interfacial surface area [m^-1]";
    constant Real sigma_n = 100.0 "Negative electrode solid conductivity [S/m]";
    constant Real sigma_p = 10.0 "Positive electrode solid conductivity [S/m]";
    constant Real b_e_n = 1.5 "Negative electrode Bruggeman coefficient [-]";
    constant Real b_e_s = 1.5 "Separator Bruggeman coefficient [-]";
    constant Real b_e_p = 1.5 "Positive electrode Bruggeman coefficient [-]";

    // Electrolyte parameters
    constant Real eps_e_n = 0.3 "Negative electrode electrolyte porosity [-]";
    constant Real eps_e_s = 1.0 "Separator electrolyte porosity [-]";
    constant Real eps_e_p = 0.3 "Positive electrode electrolyte porosity [-]";
    constant Real c_e_init = 1000.0 "Initial electrolyte concentration [mol/m^3]";
    constant Real t_plus = 0.4 "Cation transference number [-]";

    // Kinetics & Operating parameters
    constant Real m_ref_n = 2.0e-5 "Negative electrode reaction rate constant [(A/m^2)(m^3/mol)^1.5]";
    constant Real m_ref_p = 6.0e-7 "Positive electrode reaction rate constant [(A/m^2)(m^3/mol)^1.5]";
    constant Real I_typ = 0.680616 "Typical 1C current [A]";

    function U_n
      "Graphite MCMB 2528 Open-Circuit Potential [V] as a function of stoichiometry sto in (0, 1)"
      input Real sto;
      output Real u_eq;
    algorithm
      u_eq := (
        0.194
        + 1.5 * exp(-120.0 * sto)
        + 0.0351 * tanh((sto - 0.286) / 0.083)
        - 0.0045 * tanh((sto - 0.849) / 0.119)
        - 0.035 * tanh((sto - 0.9233) / 0.05)
        - 0.0147 * tanh((sto - 0.5) / 0.034)
        - 0.102 * tanh((sto - 0.194) / 0.142)
        - 0.022 * tanh((sto - 0.9) / 0.0164)
        - 0.011 * tanh((sto - 0.124) / 0.0226)
        + 0.0155 * tanh((sto - 0.105) / 0.029)
      );
    end U_n;

    function U_p
      "LiCoO2 Open-Circuit Potential [V] as a function of stoichiometry sto in (0, 1)"
      input Real sto;
      output Real u_eq;
      protected
      Real s;
    algorithm
      s := 1.062 * sto;
      u_eq := (
        2.16216
        + 0.07645 * tanh(30.834 - 54.4806 * s)
        + 2.1581 * tanh(52.294 - 50.294 * s)
        - 0.14169 * tanh(11.0923 - 19.8543 * s)
        + 0.2051 * tanh(1.4684 - 5.4888 * s)
        + 0.2531 * tanh((-s + 0.56478) / 0.1316)
        - 0.02167 * tanh((s - 0.525) / 0.006)
      );
    end U_p;

    function D_e
      "LiPF6 in EC:DMC electrolyte diffusivity [m^2/s] at T_ref (Capiglia 1999)"
      input Real c_e "Electrolyte concentration [mol/m^3]";
      output Real D_val;
    algorithm
      D_val := 5.34e-10 * exp(-0.65 * c_e / 1000.0);
    end D_e;

    function kappa_e
      "LiPF6 in EC:DMC electrolyte ionic conductivity [S/m] at T_ref (Capiglia 1999)"
      input Real c_e "Electrolyte concentration [mol/m^3]";
      output Real k_val;
      protected
      Real c_k;
    algorithm
      c_k := c_e / 1000.0;
      k_val := 0.0911 + 1.9101 * c_k - 1.052 * (c_k ^ 2) + 0.1554 * (c_k ^ 3);
    end kappa_e;

  end Marquis2019;

end Parameters;
