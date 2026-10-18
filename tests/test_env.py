import numpy as np
import pybamm
import rumoca as rm


def test_rumoca_environment():
    """Verify Rumoca can compile and simulate a basic ODE model."""
    session = rm.Session()
    model = session.loads(
        "model Decay Real x(start=1.0); equation der(x) = -x; end Decay;",
        model="Decay",
    )
    res = model.simulate(t=(0.0, 1.0), dt=0.1)
    assert np.isclose(res["x"][-1], np.exp(-1.0), rtol=1e-3)


def test_pybamm_environment():
    """Verify PyBaMM can instantiate and solve a Single Particle Model (SPM)."""
    model = pybamm.lithium_ion.SPM()
    sim = pybamm.Simulation(model)
    sol = sim.solve([0, 100])
    voltage = sol["Terminal voltage [V]"].entries
    assert len(voltage) > 1
    assert 3.0 < voltage[-1] < 4.5
