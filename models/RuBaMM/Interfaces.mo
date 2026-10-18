within RuBaMM;
package Interfaces
  "Acausal electrical connectors and standard circuit components for RuBaMM"

  connector Pin
    "Electrical pin with potential v [V] and current i [A] flowing into the pin"
    Real v(start = 0.0);
    flow Real i(start = 0.0);
  end Pin;

  partial model TwoPin
    "Base class for a two-terminal electrical component (positive pin p, negative pin n)"
    Pin p;
    Pin n;
    Real v(start = 0.0) "Terminal voltage v = p.v - n.v [V]";
    Real i(start = 0.0) "Current flowing into positive pin p -> n (discharge > 0 when i_discharge = -p.i)";
    Real i_cell(start = 0.0) "PyBaMM convention: discharge current > 0 (leaving positive pin p, so i_cell = -p.i) [A]";
  equation
    v = p.v - n.v;
    0 = p.i + n.i;
    i = p.i;
    i_cell = -p.i;
  end TwoPin;

  model Ground
    "Zero-potential electrical ground reference"
    Pin p;
  equation
    p.v = 0.0;
  end Ground;

  model CurrentSource
    "Constant or parameter-driven current load (draws discharge current I_load [A] from p to n)"
    extends TwoPin;
    parameter Real I_load = 1.0 "Discharge current drawn from positive terminal p to n [A]";
  equation
    p.i = I_load;
  end CurrentSource;

  model PulseCurrentSource
    "Sinusoidal + DC current load for dynamic testing"
    extends TwoPin;
    parameter Real I_dc = 1.0 "DC discharge current [A]";
    parameter Real I_amp = 0.5 "AC current amplitude [A]";
    parameter Real omega = 0.05 "Angular frequency [rad/s]";
  equation
    p.i = I_dc + I_amp * sin(omega * time);
  end PulseCurrentSource;

  model VoltageSource
    "Constant voltage source holding v = V_set [V] across p and n"
    extends TwoPin;
    parameter Real V_set = 4.0 "Source voltage [V]";
  equation
    v = V_set;
  end VoltageSource;

  model Resistor
    "Linear electrical resistor following Ohm's law"
    extends TwoPin;
    parameter Real R = 1.0 "Resistance [Ohm]";
  equation
    v = R * i;
  end Resistor;

end Interfaces;
