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

end Cells;
