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

    // 3. Couple volumetric reaction currents to 1D electrolyte
    for i in 1:Nn loop
      electrolyte.a_j_n[i] = P.a_n * j_n;
      eta_n_x[i] = Submodels.eta_butler_volmer(
        j_n,
        Submodels.j0_butler_volmer(P.m_ref_n, electrolyte.c_e_n[i], neg_particle.c_s_surf, P.c_n_max),
        T
      );
      log_c_e_n[i] = log(electrolyte.c_e_n[i]);
    end for;

    for i in 1:Np loop
      electrolyte.a_j_p[i] = P.a_p * j_p;
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

end Cells;
