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

end Submodels;
