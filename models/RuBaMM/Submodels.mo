within RuBaMM;
package Submodels
  "Reusable Electrochemical Submodels (Spherical Particle FVM, Kinetics, Electrolyte)"

  model SphericalParticle
    "1D Spherical Finite Volume Method (FVM) solid-phase Fickian diffusion in a single electrode particle"
    parameter Integer Nr = 10 "Number of radial control volume shells";
    parameter Real R_p = 1.0e-5 "Particle radius [m]";
    parameter Real D_s = 3.9e-14 "Solid diffusion coefficient [m^2/s]";
    parameter Real c_init = 19986.609595075 "Initial uniform lithium concentration [mol/m^3]";
    parameter Real c_max = 24983.2619938437 "Maximum lithium concentration [mol/m^3]";

    // Radial shell concentrations c_s[1..Nr] at cell centers r_i = (i - 0.5) * dr
    Real c_s[Nr](each start = c_init) "Radial shell concentrations [mol/m^3]";
    Real j_flux(start = 0.0) "Outward molar flux at particle surface r = R_p [mol/(m^2.s)] (j_flux = j_current / F)";
    Real c_s_surf(start = c_init) "Extrapolated particle surface concentration at r = R_p [mol/m^3]";
    Real sto_surf(start = c_init / c_max) "Surface stoichiometry c_s_surf / c_max [-]";

  equation
    // Innermost shell (i = 1): zero flux at r = 0, diffusive flux across r_1 = dr
    der(c_s[1]) = 3.0 * ((Nr / R_p) ^ 2) * (
      1.0 * D_s * (c_s[2] - c_s[1])
    );

    // Interior shells (i = 2 .. Nr - 1): diffusive flux across r_i = i*dr and r_{i-1} = (i-1)*dr
    for i in 2:(Nr - 1) loop
      der(c_s[i]) = (3.0 / ((i ^ 3) - ((i - 1) ^ 3))) * ((Nr / R_p) ^ 2) * (
        (i ^ 2) * D_s * (c_s[i + 1] - c_s[i])
        - ((i - 1) ^ 2) * D_s * (c_s[i] - c_s[i - 1])
      );
    end for;

    // Outermost shell (i = Nr): inward diffusive flux at r_{Nr-1} and surface molar flux j_flux at r = R_p
    der(c_s[Nr]) = (3.0 / ((Nr ^ 3) - ((Nr - 1) ^ 3))) * (
      - ((Nr ^ 3) / R_p) * j_flux
      - ((Nr - 1) ^ 2) * ((Nr / R_p) ^ 2) * D_s * (c_s[Nr] - c_s[Nr - 1])
    );

    // Linear surface concentration extrapolation matching PyBaMM FiniteVolume boundary_value
    c_s_surf = 1.5 * c_s[Nr] - 0.5 * c_s[Nr - 1];
    sto_surf = c_s_surf / c_max;
  end SphericalParticle;

  function j0_butler_volmer
    "Symmetric Butler-Volmer exchange current density [A/m^2]"
    input Real m_ref "Reaction rate constant [(A/m^2)(m^3/mol)^1.5]";
    input Real c_e "Electrolyte concentration [mol/m^3]";
    input Real c_s_surf "Surface solid concentration [mol/m^3]";
    input Real c_s_max "Maximum solid concentration [mol/m^3]";
    output Real j0 "Exchange current density [A/m^2]";
  algorithm
    j0 := m_ref * sqrt(c_e) * sqrt(c_s_surf) * sqrt(c_s_max - c_s_surf);
  end j0_butler_volmer;

  function eta_butler_volmer
    "Inverse-sinh symmetric Butler-Volmer reaction overpotential [V]"
    input Real j "Interfacial reaction current density [A/m^2]";
    input Real j0 "Exchange current density [A/m^2]";
    input Real T "Temperature [K]";
    output Real eta "Reaction overpotential [V]";
    protected
    Real u;
  algorithm
    u := j / (2.0 * j0);
    eta := (2.0 * Parameters.Marquis2019.R_gas * T / Parameters.Marquis2019.F) * log(u + sqrt(u ^ 2 + 1.0));
  end eta_butler_volmer;

  model Electrolyte1D
    "1D Planar Finite Volume Method (FVM) electrolyte mass transport across Negative Electrode (Nn), Separator (Ns), and Positive Electrode (Np)"
    import P = RuBaMM.Parameters.Marquis2019;

    parameter Integer Nn = 4 "Number of FVM control volumes in negative electrode";
    parameter Integer Ns = 3 "Number of FVM control volumes in separator";
    parameter Integer Np = 4 "Number of FVM control volumes in positive electrode";

    parameter Real dx_n = P.L_n / Nn "Negative electrode FVM cell width [m]";
    parameter Real dx_s = P.L_s / Ns "Separator FVM cell width [m]";
    parameter Real dx_p = P.L_p / Np "Positive electrode FVM cell width [m]";

    parameter Real b_n = P.eps_e_n ^ P.b_e_n "Negative electrode Bruggeman tortuosity factor [-]";
    parameter Real b_s = P.eps_e_s ^ P.b_e_s "Separator Bruggeman tortuosity factor [-]";
    parameter Real b_p = P.eps_e_p ^ P.b_e_p "Positive electrode Bruggeman tortuosity factor [-]";

    // Harmonic mean distance weights at the n/s and s/p interfaces
    parameter Real w_ns_n = dx_n / (dx_n + dx_s);
    parameter Real w_ns_s = dx_s / (dx_n + dx_s);
    parameter Real dx_ns = 0.5 * (dx_n + dx_s);

    parameter Real w_sp_s = dx_s / (dx_s + dx_p);
    parameter Real w_sp_p = dx_p / (dx_s + dx_p);
    parameter Real dx_sp = 0.5 * (dx_s + dx_p);

    // Conserved state variables: porosity times electrolyte concentration eps_k * c_e [mol/m^3]
    Real eps_c_e_n[Nn](each start = P.eps_e_n * P.c_e_init) "Porosity times concentration in negative electrode [mol/m^3]";
    Real eps_c_e_s[Ns](each start = P.eps_e_s * P.c_e_init) "Porosity times concentration in separator [mol/m^3]";
    Real eps_c_e_p[Np](each start = P.eps_e_p * P.c_e_init) "Porosity times concentration in positive electrode [mol/m^3]";

    // Intrinsic electrolyte concentrations c_e [mol/m^3]
    Real c_e_n[Nn](each start = P.c_e_init) "Electrolyte concentration in negative electrode [mol/m^3]";
    Real c_e_s[Ns](each start = P.c_e_init) "Electrolyte concentration in separator [mol/m^3]";
    Real c_e_p[Np](each start = P.c_e_init) "Electrolyte concentration in positive electrode [mol/m^3]";

    // Cell-centered effective diffusivities D_eff [m^2/s]
    Real D_eff_n[Nn](each start = (P.eps_e_n ^ P.b_e_n) * 2.79e-10);
    Real D_eff_s[Ns](each start = (P.eps_e_s ^ P.b_e_s) * 2.79e-10);
    Real D_eff_p[Np](each start = (P.eps_e_p ^ P.b_e_p) * 2.79e-10);

    // Diffusive molar fluxes at internal faces [mol/(m^2.s)]
    Real N_diff_n[Nn - 1](each start = 0.0) "Interior diffusive fluxes in negative electrode";
    Real N_diff_ns(start = 0.0) "Diffusive flux at negative electrode / separator interface";
    Real N_diff_s[Ns - 1](each start = 0.0) "Interior diffusive fluxes in separator";
    Real N_diff_sp(start = 0.0) "Diffusive flux at separator / positive electrode interface";
    Real N_diff_p[Np - 1](each start = 0.0) "Interior diffusive fluxes in positive electrode";

    // Volumetric reaction current source terms a_k * j_k [A/m^3] (coupled by parent cell model)
    Real a_j_n[Nn](each start = 0.0) "Volumetric interfacial current density in negative electrode [A/m^3]";
    Real a_j_p[Np](each start = 0.0) "Volumetric interfacial current density in positive electrode [A/m^3]";

    // Electrolyte ionic current density at cell right/left faces [A/m^2] (for migration flux div(t_plus * i_e / F))
    Real i_e_n_right[Nn](each start = 0.0) "Electrolyte current density at right face of negative electrode cells [A/m^2]";
    Real i_e_p_left[Np](each start = 0.0) "Electrolyte current density at left face of positive electrode cells [A/m^2]";

    // Domain-averaged concentrations
    Real c_e_n_avg(start = P.c_e_init) "X-averaged negative electrolyte concentration [mol/m^3]";
    Real c_e_s_avg(start = P.c_e_init) "X-averaged separator electrolyte concentration [mol/m^3]";
    Real c_e_p_avg(start = P.c_e_init) "X-averaged positive electrolyte concentration [mol/m^3]";
    Real c_e_av(start = 1161.3) "Pore-volume-averaged electrolyte concentration across cell [mol/m^3]";

  equation
    // 1. Intrinsic concentrations and effective diffusivities
    for i in 1:Nn loop
      c_e_n[i] = eps_c_e_n[i] / P.eps_e_n;
      D_eff_n[i] = b_n * P.D_e(c_e_n[i]);
    end for;
    for i in 1:Ns loop
      c_e_s[i] = eps_c_e_s[i] / P.eps_e_s;
      D_eff_s[i] = b_s * P.D_e(c_e_s[i]);
    end for;
    for i in 1:Np loop
      c_e_p[i] = eps_c_e_p[i] / P.eps_e_p;
      D_eff_p[i] = b_p * P.D_e(c_e_p[i]);
    end for;

    // 2. Harmonic-mean diffusive fluxes at cell faces
    for i in 1:(Nn - 1) loop
      N_diff_n[i] = -(2.0 * D_eff_n[i] * D_eff_n[i + 1] / (D_eff_n[i] + D_eff_n[i + 1])) * (c_e_n[i + 1] - c_e_n[i]) / dx_n;
    end for;

    N_diff_ns = -(1.0 / (w_ns_n / D_eff_n[Nn] + w_ns_s / D_eff_s[1])) * (c_e_s[1] - c_e_n[Nn]) / dx_ns;

    for i in 1:(Ns - 1) loop
      N_diff_s[i] = -(2.0 * D_eff_s[i] * D_eff_s[i + 1] / (D_eff_s[i] + D_eff_s[i + 1])) * (c_e_s[i + 1] - c_e_s[i]) / dx_s;
    end for;

    N_diff_sp = -(1.0 / (w_sp_s / D_eff_s[Ns] + w_sp_p / D_eff_p[1])) * (c_e_p[1] - c_e_s[Ns]) / dx_sp;

    for i in 1:(Np - 1) loop
      N_diff_p[i] = -(2.0 * D_eff_p[i] * D_eff_p[i + 1] / (D_eff_p[i] + D_eff_p[i + 1])) * (c_e_p[i + 1] - c_e_p[i]) / dx_p;
    end for;

    // 3. Conservation equations: der(eps_c_e) = -div(N_diff + t_plus * i_e / F) + a_j / F
    // Negative electrode (left boundary i_e = 0)
    der(eps_c_e_n[1]) = -(N_diff_n[1] - 0.0) / dx_n + (a_j_n[1] - P.t_plus * (i_e_n_right[1] - 0.0) / dx_n) / P.F;
    for i in 2:(Nn - 1) loop
      der(eps_c_e_n[i]) = -(N_diff_n[i] - N_diff_n[i - 1]) / dx_n + (a_j_n[i] - P.t_plus * (i_e_n_right[i] - i_e_n_right[i - 1]) / dx_n) / P.F;
    end for;
    der(eps_c_e_n[Nn]) = -(N_diff_ns - N_diff_n[Nn - 1]) / dx_n + (a_j_n[Nn] - P.t_plus * (i_e_n_right[Nn] - i_e_n_right[Nn - 1]) / dx_n) / P.F;

    // Separator (zero reaction source, uniform i_e across separator faces so div(i_e) = 0)
    der(eps_c_e_s[1]) = -(N_diff_s[1] - N_diff_ns) / dx_s;
    for i in 2:(Ns - 1) loop
      der(eps_c_e_s[i]) = -(N_diff_s[i] - N_diff_s[i - 1]) / dx_s;
    end for;
    der(eps_c_e_s[Ns]) = -(N_diff_sp - N_diff_s[Ns - 1]) / dx_s;

    // Positive electrode (right boundary i_e = 0)
    der(eps_c_e_p[1]) = -(N_diff_p[1] - N_diff_sp) / dx_p + (a_j_p[1] - P.t_plus * (i_e_p_left[2] - i_e_p_left[1]) / dx_p) / P.F;
    for i in 2:(Np - 1) loop
      der(eps_c_e_p[i]) = -(N_diff_p[i] - N_diff_p[i - 1]) / dx_p + (a_j_p[i] - P.t_plus * (i_e_p_left[i + 1] - i_e_p_left[i]) / dx_p) / P.F;
    end for;
    der(eps_c_e_p[Np]) = -(0.0 - N_diff_p[Np - 1]) / dx_p + (a_j_p[Np] - P.t_plus * (0.0 - i_e_p_left[Np]) / dx_p) / P.F;

    // 4. Domain averages
    c_e_n_avg = sum(c_e_n) / Nn;
    c_e_s_avg = sum(c_e_s) / Ns;
    c_e_p_avg = sum(c_e_p) / Np;
    c_e_av = (
      (P.L_n / P.eps_e_n + P.L_s / P.eps_e_s + P.L_p / P.eps_e_p) / (P.L_n + P.L_s + P.L_p)
    ) * (
      dx_n * sum(eps_c_e_n) + dx_s * sum(eps_c_e_s) + dx_p * sum(eps_c_e_p)
    ) / (P.L_n + P.L_s + P.L_p);
  end Electrolyte1D;

end Submodels;
