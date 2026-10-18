within RuBaMM;
package Cells
  "Complete Lithium-Ion Electrochemical Cell Models with Acausal Electrical Terminals"

  model SPM
    "Single Particle Model (SPM) of a Lithium-Ion Cell (matching pybamm.lithium_ion.SPM)"
    extends Interfaces.TwoPin;
    import P = RuBaMM.Parameters.Marquis2019;

    parameter Integer Nr = 10 "Number of radial shells per electrode particle";
    parameter Real T = P.T_ref "Cell temperature [K]";
    parameter Real c_n_init = P.c_n_init "Initial negative particle concentration [mol/m^3]";
    parameter Real c_p_init = P.c_p_init "Initial positive particle concentration [mol/m^3]";

    // Representative negative and positive spherical particles
    Submodels.SphericalParticle neg_particle(
      Nr = Nr,
      R_p = P.R_n,
      D_s = P.D_s_n_ref,
      c_init = c_n_init,
      c_max = P.c_n_max
    );

    Submodels.SphericalParticle pos_particle(
      Nr = Nr,
      R_p = P.R_p,
      D_s = P.D_s_p_ref,
      c_init = c_p_init,
      c_max = P.c_p_max
    );

    // Interfacial current densities [A/m^2]
    Real j_n(start = 0.0) "Negative electrode interfacial current density [A/m^2]";
    Real j_p(start = 0.0) "Positive electrode interfacial current density [A/m^2]";

    // Exchange current densities [A/m^2]
    Real j0_n(start = 6.0) "Negative electrode exchange current density [A/m^2]";
    Real j0_p(start = 0.5) "Positive electrode exchange current density [A/m^2]";

    // Open-circuit potentials [V]
    Real U_n(start = 0.18) "Negative electrode open-circuit potential [V]";
    Real U_p(start = 4.01) "Positive electrode open-circuit potential [V]";

    // Reaction overpotentials [V]
    Real eta_n(start = 0.0) "Negative electrode reaction overpotential [V]";
    Real eta_p(start = 0.0) "Positive electrode reaction overpotential [V]";

    // Discharge capacity [A.h]
    Real Q_dis_Ah(start = 0.0) "Discharge capacity [A.h]";

  equation
    // Discharge capacity integration
    der(Q_dis_Ah) = i_cell / 3600.0;

    // Uniform interfacial current densities from total cell current i_cell
    j_n = i_cell / (P.a_n * P.L_n * P.A_cc);
    j_p = -i_cell / (P.a_p * P.L_p * P.A_cc);

    // Couple molar reaction fluxes into the solid particles
    neg_particle.j_flux = j_n / P.F;
    pos_particle.j_flux = j_p / P.F;

    // Exchange current densities (electrolyte concentration is uniform c_e_init in SPM)
    j0_n = Submodels.j0_butler_volmer(P.m_ref_n, P.c_e_init, neg_particle.c_s_surf, P.c_n_max);
    j0_p = Submodels.j0_butler_volmer(P.m_ref_p, P.c_e_init, pos_particle.c_s_surf, P.c_p_max);

    // Open-circuit potentials at particle surfaces
    U_n = P.U_n(neg_particle.sto_surf);
    U_p = P.U_p(pos_particle.sto_surf);

    // Symmetric Butler-Volmer reaction overpotentials
    eta_n = Submodels.eta_butler_volmer(j_n, j0_n, T);
    eta_p = Submodels.eta_butler_volmer(j_p, j0_p, T);

    // Terminal voltage across positive and negative pins
    v = (U_p + eta_p) - (U_n + eta_n);
  end SPM;

  model SPMe
    "Single Particle Model with Electrolyte (SPMe) of a Lithium-Ion Cell (matching pybamm.lithium_ion.SPMe)"
    extends Interfaces.TwoPin;
    import P = RuBaMM.Parameters.Marquis2019;

    parameter Integer Nr = 10 "Number of radial shells per electrode particle";
    parameter Integer Nn = 4 "Number of FVM control volumes in negative electrode";
    parameter Integer Ns = 3 "Number of FVM control volumes in separator";
    parameter Integer Np = 4 "Number of FVM control volumes in positive electrode";

    parameter Real T = P.T_ref "Cell temperature [K]";
    parameter Real c_n_init = P.c_n_init "Initial negative particle concentration [mol/m^3]";
    parameter Real c_p_init = P.c_p_init "Initial positive particle concentration [mol/m^3]";

    // Effective solid-phase conductivities
    parameter Real sigma_eff_n = P.sigma_n * ((1.0 - P.eps_e_n) ^ P.b_e_n);
    parameter Real sigma_eff_p = P.sigma_p * ((1.0 - P.eps_e_p) ^ P.b_e_p);

    // Representative negative and positive spherical particles
    Submodels.SphericalParticle neg_particle(
      Nr = Nr,
      R_p = P.R_n,
      D_s = P.D_s_n_ref,
      c_init = c_n_init,
      c_max = P.c_n_max
    );

    Submodels.SphericalParticle pos_particle(
      Nr = Nr,
      R_p = P.R_p,
      D_s = P.D_s_p_ref,
      c_init = c_p_init,
      c_max = P.c_p_max
    );

    // 1D Electrolyte FVM transport submodel
    Submodels.Electrolyte1D electrolyte(
      Nn = Nn,
      Ns = Ns,
      Np = Np
    );

    // Interfacial current densities [A/m^2]
    Real j_n(start = 0.0) "Negative electrode interfacial current density [A/m^2]";
    Real j_p(start = 0.0) "Positive electrode interfacial current density [A/m^2]";

    // Spatially resolved reaction overpotentials across electrodes [V]
    Real eta_n_x[Nn](each start = 0.0);
    Real eta_p_x[Np](each start = 0.0);
    Real eta_n(start = 0.0) "X-averaged negative electrode reaction overpotential [V]";
    Real eta_p(start = 0.0) "X-averaged positive electrode reaction overpotential [V]";

    // Open-circuit potentials [V]
    Real U_n(start = 0.18) "Negative electrode open-circuit potential [V]";
    Real U_p(start = 4.01) "Positive electrode open-circuit potential [V]";

    // Electrolyte & Solid-phase overpotential components [V]
    Real log_c_e_n[Nn](each start = 6.9);
    Real log_c_e_p[Np](each start = 6.9);
    Real eta_conc(start = 0.0) "X-averaged electrolyte concentration overpotential [V]";
    Real dphi_e_ohm(start = 0.0) "X-averaged electrolyte ohmic losses [V]";
    Real dphi_s_ohm(start = 0.0) "X-averaged solid phase ohmic losses [V]";

  equation
    // 1. Uniform interfacial current densities
    j_n = i_cell / (P.a_n * P.L_n * P.A_cc);
    j_p = -i_cell / (P.a_p * P.L_p * P.A_cc);

    // 2. Couple molar fluxes to solid particles
    neg_particle.j_flux = j_n / P.F;
    pos_particle.j_flux = j_p / P.F;

    // 3. Couple volumetric reaction currents and linear electrolyte current to 1D electrolyte
    for i in 1:Nn loop
      electrolyte.a_j_n[i] = P.a_n * j_n;
      electrolyte.i_e_n_right[i] = (i_cell / P.A_cc) * (i * 1.0 / Nn);
      eta_n_x[i] = Submodels.eta_butler_volmer(
        j_n,
        Submodels.j0_butler_volmer(P.m_ref_n, electrolyte.c_e_n[i], neg_particle.c_s_surf, P.c_n_max),
        T
      );
      log_c_e_n[i] = log(electrolyte.c_e_n[i]);
    end for;

    for i in 1:Np loop
      electrolyte.a_j_p[i] = P.a_p * j_p;
      electrolyte.i_e_p_left[i] = (i_cell / P.A_cc) * (1.0 - (i - 1.0) / Np);
      eta_p_x[i] = Submodels.eta_butler_volmer(
        j_p,
        Submodels.j0_butler_volmer(P.m_ref_p, electrolyte.c_e_p[i], pos_particle.c_s_surf, P.c_p_max),
        T
      );
      log_c_e_p[i] = log(electrolyte.c_e_p[i]);
    end for;

    // 4. Open-circuit potentials and X-averaged reaction overpotentials
    U_n = P.U_n(neg_particle.sto_surf);
    U_p = P.U_p(pos_particle.sto_surf);
    eta_n = sum(eta_n_x) / Nn;
    eta_p = sum(eta_p_x) / Np;

    // 5. Electrolyte concentration overpotential and ohmic drops
    eta_conc = (2.0 * (1.0 - P.t_plus) * P.R_gas * T / P.F) * (
      (sum(log_c_e_p) / Np) - (sum(log_c_e_n) / Nn)
    );

    dphi_e_ohm = -(i_cell / P.A_cc) / P.kappa_e(electrolyte.c_e_av) * (
      P.L_n / (3.0 * electrolyte.b_n)
      + P.L_s / electrolyte.b_s
      + P.L_p / (3.0 * electrolyte.b_p)
    );

    dphi_s_ohm = -(i_cell / P.A_cc) * (
      P.L_n / (3.0 * sigma_eff_n)
      + P.L_p / (3.0 * sigma_eff_p)
    );

    // 6. Terminal voltage
    v = (U_p - U_n) + (eta_p - eta_n) + eta_conc + dphi_e_ohm + dphi_s_ohm;
  end SPMe;

  model DFN
    "Full Doyle-Fuller-Newman (DFN / P2D) Lithium-Ion Cell Model (matching pybamm.lithium_ion.DFN)"
    extends Interfaces.TwoPin;
    import P = RuBaMM.Parameters.Marquis2019;

    parameter Integer Nr = 8 "Number of radial shells per electrode particle";
    parameter Integer Nn = 4 "Number of FVM control volumes in negative electrode";
    parameter Integer Ns = 3 "Number of FVM control volumes in separator";
    parameter Integer Np = 4 "Number of FVM control volumes in positive electrode";

    parameter Real T = P.T_ref "Cell temperature [K]";
    parameter Real c_n_init = P.c_n_init "Initial negative particle concentration [mol/m^3]";
    parameter Real c_p_init = P.c_p_init "Initial positive particle concentration [mol/m^3]";

    // Effective solid-phase conductivities [S/m]
    parameter Real sigma_eff_n = P.sigma_n * ((1.0 - P.eps_e_n) ^ P.b_e_n);
    parameter Real sigma_eff_p = P.sigma_p * ((1.0 - P.eps_e_p) ^ P.b_e_p);

    // Initial open-circuit potentials for consistent DAE initialization
    parameter Real U_n_0 = 0.175193184028335;
    parameter Real U_p_0 = 4.027013847342061;

    // 1. Spatially distributed spherical particles across negative (Nn) and positive (Np) electrodes
    Submodels.SphericalParticle neg_particles[Nn](
      each Nr = Nr,
      each R_p = P.R_n,
      each D_s = P.D_s_n_ref,
      each c_init = c_n_init,
      each c_max = P.c_n_max
    );

    Submodels.SphericalParticle pos_particles[Np](
      each Nr = Nr,
      each R_p = P.R_p,
      each D_s = P.D_s_p_ref,
      each c_init = c_p_init,
      each c_max = P.c_p_max
    );

    // 2. 1D Electrolyte FVM mass transport
    Submodels.Electrolyte1D electrolyte(
      Nn = Nn,
      Ns = Ns,
      Np = Np
    );

    // 3. Algebraic potential fields [V]
    Real phi_s_n[Nn](each start = 0.0) "Negative electrode solid potential [V]";
    Real phi_s_p[Np](each start = U_p_0 - U_n_0) "Positive electrode solid potential [V]";
    Real phi_e_n[Nn](each start = -U_n_0) "Negative electrode electrolyte potential [V]";
    Real phi_e_s[Ns](each start = -U_n_0) "Separator electrolyte potential [V]";
    Real phi_e_p[Np](each start = -U_n_0) "Positive electrode electrolyte potential [V]";

    // 4. Local reaction kinetics variables
    Real U_n_x[Nn](each start = U_n_0) "Local negative electrode OCP [V]";
    Real U_p_x[Np](each start = U_p_0) "Local positive electrode OCP [V]";
    Real j0_n_x[Nn](each start = 6.7) "Local negative electrode exchange current density [A/m^2]";
    Real j0_p_x[Np](each start = 0.47) "Local positive electrode exchange current density [A/m^2]";
    Real eta_n_x[Nn](each start = 0.0) "Local negative electrode reaction overpotential [V]";
    Real eta_p_x[Np](each start = 0.0) "Local positive electrode reaction overpotential [V]";
    Real j_n[Nn](each start = 0.0) "Local negative electrode interfacial current density [A/m^2]";
    Real j_p[Np](each start = 0.0) "Local positive electrode interfacial current density [A/m^2]";

    // 5. Electrolyte ionic conductivity and ionic current density across faces
    Real kappa_eff_n[Nn](each start = 0.16);
    Real kappa_eff_s[Ns](each start = 1.0);
    Real kappa_eff_p[Np](each start = 0.16);
    Real kd_n[Nn](each start = 0.00003);
    Real kd_s[Ns](each start = 0.00003);
    Real kd_p[Np](each start = 0.00003);

    Real i_e_n[Nn - 1](each start = 0.0) "Interior electrolyte current density in negative electrode [A/m^2]";
    Real i_e_ns(start = 0.0) "Electrolyte current density at negative/separator interface [A/m^2]";
    Real i_e_s[Ns - 1](each start = 0.0) "Interior electrolyte current density in separator [A/m^2]";
    Real i_e_sp(start = 0.0) "Electrolyte current density at separator/positive interface [A/m^2]";
    Real i_e_p[Np - 1](each start = 0.0) "Interior electrolyte current density in positive electrode [A/m^2]";

    // 6. Summary X-averaged observables
    Real c_s_n_surf_avg(start = c_n_init) "X-averaged negative particle surface concentration [mol/m^3]";
    Real c_s_p_surf_avg(start = c_p_init) "X-averaged positive particle surface concentration [mol/m^3]";
    Real c_s_n_surf_x[Nn](each start = c_n_init);
    Real c_s_p_surf_x[Np](each start = c_p_init);
    Real eta_n_avg(start = 0.0) "X-averaged negative reaction overpotential [V]";
    Real eta_p_avg(start = 0.0) "X-averaged positive reaction overpotential [V]";
    Real phi_s_p_cc(start = U_p_0 - U_n_0) "Positive current collector solid potential at x = L [V]";

  equation
    // A. Local Butler-Volmer kinetics and particle coupling in Negative Electrode
    for i in 1:Nn loop
      c_s_n_surf_x[i] = neg_particles[i].c_s_surf;
      U_n_x[i] = P.U_n(neg_particles[i].sto_surf);
      j0_n_x[i] = Submodels.j0_butler_volmer(P.m_ref_n, electrolyte.c_e_n[i], neg_particles[i].c_s_surf, P.c_n_max);
      eta_n_x[i] = phi_s_n[i] - phi_e_n[i] - U_n_x[i];
      j_n[i] = 2.0 * j0_n_x[i] * sinh((0.5 * P.F / (P.R_gas * T)) * eta_n_x[i]);
      neg_particles[i].j_flux = j_n[i] / P.F;
      electrolyte.a_j_n[i] = P.a_n * j_n[i];
      kappa_eff_n[i] = electrolyte.b_n * P.kappa_e(electrolyte.c_e_n[i]);
      kd_n[i] = (2.0 * (1.0 - P.t_plus) * P.R_gas * T / P.F) / electrolyte.c_e_n[i];
    end for;

    for i in 1:Ns loop
      kappa_eff_s[i] = electrolyte.b_s * P.kappa_e(electrolyte.c_e_s[i]);
      kd_s[i] = (2.0 * (1.0 - P.t_plus) * P.R_gas * T / P.F) / electrolyte.c_e_s[i];
    end for;

    // B. Local Butler-Volmer kinetics and particle coupling in Positive Electrode
    for i in 1:Np loop
      c_s_p_surf_x[i] = pos_particles[i].c_s_surf;
      U_p_x[i] = P.U_p(pos_particles[i].sto_surf);
      j0_p_x[i] = Submodels.j0_butler_volmer(P.m_ref_p, electrolyte.c_e_p[i], pos_particles[i].c_s_surf, P.c_p_max);
      eta_p_x[i] = phi_s_p[i] - phi_e_p[i] - U_p_x[i];
      j_p[i] = 2.0 * j0_p_x[i] * sinh((0.5 * P.F / (P.R_gas * T)) * eta_p_x[i]);
      pos_particles[i].j_flux = j_p[i] / P.F;
      electrolyte.a_j_p[i] = P.a_p * j_p[i];
      kappa_eff_p[i] = electrolyte.b_p * P.kappa_e(electrolyte.c_e_p[i]);
      kd_p[i] = (2.0 * (1.0 - P.t_plus) * P.R_gas * T / P.F) / electrolyte.c_e_p[i];
    end for;

    // C. Solid-phase charge conservation in Negative Electrode (Dirichlet phi_s_n(0) = 0, Neumann at x = L_n)
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * (
      P.a_n * j_n[1] - (sigma_eff_n / (electrolyte.dx_n ^ 2)) * (phi_s_n[2] - 3.0 * phi_s_n[1])
    );
    for i in 2:(Nn - 1) loop
      0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * (
        P.a_n * j_n[i] - (sigma_eff_n / (electrolyte.dx_n ^ 2)) * (phi_s_n[i + 1] - 2.0 * phi_s_n[i] + phi_s_n[i - 1])
      );
    end for;
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * (
      P.a_n * j_n[Nn] - (sigma_eff_n / (electrolyte.dx_n ^ 2)) * (-phi_s_n[Nn] + phi_s_n[Nn - 1])
    );

    // D. Solid-phase charge conservation in Positive Electrode (Neumann 0 at x = L_n+L_s, current i_cell at x = L)
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * (
      P.a_p * j_p[1] - (sigma_eff_p / (electrolyte.dx_p ^ 2)) * (phi_s_p[2] - phi_s_p[1])
    );
    for i in 2:(Np - 1) loop
      0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * (
        P.a_p * j_p[i] - (sigma_eff_p / (electrolyte.dx_p ^ 2)) * (phi_s_p[i + 1] - 2.0 * phi_s_p[i] + phi_s_p[i - 1])
      );
    end for;
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * (
      P.a_p * j_p[Np] - (
        -(i_cell / P.A_cc) / electrolyte.dx_p
        - (sigma_eff_p / (electrolyte.dx_p ^ 2)) * (phi_s_p[Np] - phi_s_p[Np - 1])
      )
    );

    // E. Electrolyte ionic current density at internal faces (matching PyBaMM arithmetic kappa_eff & harmonic 1/c_e)
    for i in 1:(Nn - 1) loop
      i_e_n[i] = (0.5 * (kappa_eff_n[i] + kappa_eff_n[i + 1])) * (
        (2.0 * kd_n[i] * kd_n[i + 1] / (kd_n[i] + kd_n[i + 1])) * (electrolyte.c_e_n[i + 1] - electrolyte.c_e_n[i]) / electrolyte.dx_n
        - (phi_e_n[i + 1] - phi_e_n[i]) / electrolyte.dx_n
      );
      electrolyte.i_e_n_right[i] = i_e_n[i];
    end for;

    i_e_ns = (electrolyte.w_ns_s * kappa_eff_n[Nn] + electrolyte.w_ns_n * kappa_eff_s[1]) * (
      (1.0 / (electrolyte.w_ns_n / kd_n[Nn] + electrolyte.w_ns_s / kd_s[1])) * (electrolyte.c_e_s[1] - electrolyte.c_e_n[Nn]) / electrolyte.dx_ns
      - (phi_e_s[1] - phi_e_n[Nn]) / electrolyte.dx_ns
    );
    electrolyte.i_e_n_right[Nn] = i_e_ns;

    for i in 1:(Ns - 1) loop
      i_e_s[i] = (0.5 * (kappa_eff_s[i] + kappa_eff_s[i + 1])) * (
        (2.0 * kd_s[i] * kd_s[i + 1] / (kd_s[i] + kd_s[i + 1])) * (electrolyte.c_e_s[i + 1] - electrolyte.c_e_s[i]) / electrolyte.dx_s
        - (phi_e_s[i + 1] - phi_e_s[i]) / electrolyte.dx_s
      );
    end for;

    i_e_sp = (electrolyte.w_sp_p * kappa_eff_s[Ns] + electrolyte.w_sp_s * kappa_eff_p[1]) * (
      (1.0 / (electrolyte.w_sp_s / kd_s[Ns] + electrolyte.w_sp_p / kd_p[1])) * (electrolyte.c_e_p[1] - electrolyte.c_e_s[Ns]) / electrolyte.dx_sp
      - (phi_e_p[1] - phi_e_s[Ns]) / electrolyte.dx_sp
    );
    electrolyte.i_e_p_left[1] = i_e_sp;

    for i in 1:(Np - 1) loop
      i_e_p[i] = (0.5 * (kappa_eff_p[i] + kappa_eff_p[i + 1])) * (
        (2.0 * kd_p[i] * kd_p[i + 1] / (kd_p[i] + kd_p[i + 1])) * (electrolyte.c_e_p[i + 1] - electrolyte.c_e_p[i]) / electrolyte.dx_p
        - (phi_e_p[i + 1] - phi_e_p[i]) / electrolyte.dx_p
      );
      electrolyte.i_e_p_left[i + 1] = i_e_p[i];
    end for;

    // F. Electrolyte charge conservation: div(i_e) = a_k * j_k across Nn + Ns + Np control volumes
    // Negative electrode
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_n[1] - 0.0) / electrolyte.dx_n - P.a_n * j_n[1]);
    for i in 2:(Nn - 1) loop
      0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_n[i] - i_e_n[i - 1]) / electrolyte.dx_n - P.a_n * j_n[i]);
    end for;
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_ns - i_e_n[Nn - 1]) / electrolyte.dx_n - P.a_n * j_n[Nn]);

    // Separator
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_s[1] - i_e_ns) / electrolyte.dx_s);
    for i in 2:(Ns - 1) loop
      0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_s[i] - i_e_s[i - 1]) / electrolyte.dx_s);
    end for;
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_sp - i_e_s[Ns - 1]) / electrolyte.dx_s);

    // Positive electrode
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_p[1] - i_e_sp) / electrolyte.dx_p - P.a_p * j_p[1]);
    for i in 2:(Np - 1) loop
      0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((i_e_p[i] - i_e_p[i - 1]) / electrolyte.dx_p - P.a_p * j_p[i]);
    end for;
    0.0 = ((P.L_n + P.L_s + P.L_p) ^ 2) * ((0.0 - i_e_p[Np - 1]) / electrolyte.dx_p - P.a_p * j_p[Np]);

    // G. Terminal voltage (boundary extrapolation at x = L matching PyBaMM FiniteVolume Neumann boundary_value)
    phi_s_p_cc = 1.5 * phi_s_p[Np] - 0.5 * phi_s_p[Np - 1]
      + (3.0 * electrolyte.dx_p / 8.0) * (-(i_cell / P.A_cc) / sigma_eff_p);
    v = phi_s_p_cc;

    // H. Summary X-averaged observables
    c_s_n_surf_avg = sum(c_s_n_surf_x) / Nn;
    c_s_p_surf_avg = sum(c_s_p_surf_x) / Np;
    eta_n_avg = sum(eta_n_x) / Nn;
    eta_p_avg = sum(eta_p_x) / Np;
  end DFN;

end Cells;
