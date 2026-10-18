# RuBaMM

**RuBaMM** is a ground-up, native Modelica (`.mo`) implementation of [PyBaMM](https://pybamm.org/) battery models designed for compilation, symbolic manipulation, and acausal multi-subsystem simulation with [Rumoca](https://github.com/cognipilot/rumoca).

## Package Structure

* [`models/RuBaMM/package.mo`](models/RuBaMM/package.mo) — Root `RuBaMM` Modelica package.
* [`models/RuBaMM/Interfaces.mo`](models/RuBaMM/Interfaces.mo) — Acausal electrical connectors (`Pin`, `TwoPin`) and circuit components (`Ground`, `CurrentSource`, `PulseCurrentSource`, `VoltageSource`, `Resistor`).
* [`models/RuBaMM/EquivalentCircuit.mo`](models/RuBaMM/EquivalentCircuit.mo) — 1-RC `Thevenin` Equivalent Circuit Model with coupled lumped cell/jig thermal dynamics (`pybamm.equivalent_circuit.Thevenin`).
* [`models/RuBaMM/Parameters.mo`](models/RuBaMM/Parameters.mo) — Standard `Marquis2019` Graphite/LiCoO2 physical constants, analytical Open-Circuit Potentials (`U_n`, `U_p`), and Capiglia electrolyte diffusivity/conductivity (`D_e`, `kappa_e`).
* [`models/RuBaMM/Submodels.mo`](models/RuBaMM/Submodels.mo) — Reusable Finite Volume Method (FVM) electrochemical submodels:
  * `SphericalParticle` — 1D spherical FVM Fickian solid diffusion with linear surface extrapolation.
  * `j0_butler_volmer` & `eta_butler_volmer` — Symmetric Butler-Volmer exchange current density and overpotential functions.
  * `Electrolyte1D` — 1D planar FVM electrolyte mass conservation across Negative Electrode, Separator, and Positive Electrode with harmonic-mean interface diffusivities.
* [`models/RuBaMM/Cells.mo`](models/RuBaMM/Cells.mo) — Full acausal lithium-ion cell models extending `Interfaces.TwoPin`:
  * `SPM` — Single Particle Model (`pybamm.lithium_ion.SPM`).
  * `SPMe` — Single Particle Model with Electrolyte (`pybamm.lithium_ion.SPMe`).
  * `DFN` — Full Doyle-Fuller-Newman / P2D nonlinear DAE model (`pybamm.lithium_ion.DFN`).

## Running Equivalence Tests

Activate the virtual environment and run the checkpoint `pytest` suite:

```powershell
.\.venv\Scripts\pytest -v
```

### Checkpoints

1. **Checkpoint 1** (`tests/test_cp1_interfaces_ecm.py`): Acausal electrical interfaces and `Thevenin` ECM vs `pybamm.equivalent_circuit.Thevenin`.
2. **Checkpoint 2** (`tests/test_cp2_particle_ocp.py`): `Marquis2019` OCP/electrolyte functions and `SphericalParticle` FVM radial shells vs PyBaMM.
3. **Checkpoint 3** (`tests/test_cp3_spm.py`): `RuBaMM.Cells.SPM` vs `pybamm.lithium_ion.SPM()` under constant-current and sinusoidal dynamic loads.
4. **Checkpoint 4** (`tests/test_cp4_spme.py`): `RuBaMM.Cells.SPMe` and `Electrolyte1D` vs `pybamm.lithium_ion.SPMe()`.
5. **Checkpoint 5** (`tests/test_cp5_dfn.py`): `RuBaMM.Cells.DFN` (coupled P2D nonlinear DAE) vs `pybamm.lithium_ion.DFN()`.
6. **Checkpoint 6** (`tests/test_cp6_acausal_coupling.py`): Acausal subsystem coupling of `DFN` and `SPM` cells with external `VoltageSource`, `Resistor` loads, and parallel multi-cell pack balancing.
