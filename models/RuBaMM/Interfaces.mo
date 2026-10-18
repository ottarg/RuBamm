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

  model CCCVCycler
    "Acausal 5-stage Battery Cycler: (1) CC Discharge -> (2) Rest -> (3) CC Charge -> (4) CV Hold -> (5) Rest"
    extends TwoPin;
    parameter Real I_dis = 0.680616 "Stage 1: Constant discharge current [A]";
    parameter Real V_min = 3.6 "Stage 1 cutoff: Minimum discharge voltage [V]";
    parameter Real t_rest1 = 300.0 "Stage 2: Rest duration after discharge [s]";
    parameter Real I_chg = 0.680616 "Stage 3: Constant charge current [A]";
    parameter Real V_max = 4.0 "Stage 3/4: Charge target & CV hold voltage [V]";
    parameter Real t_cv = 600.0 "Stage 4: CV hold duration (or until |i| <= I_cut) [s]";
    parameter Real I_cut = 0.05 "Stage 4 cutoff: Minimum CV taper current magnitude [A]";

    Integer stage(start = 1) "Active experiment stage (1..5)";
    Real t_stage(start = 0.0) "Timestamp when current stage started [s]";
  equation
    when pre(stage) == 1 and v <= (if pre(stage) == 1 then V_min else -100.0) then
      stage = 2;
      t_stage = time;
    elsewhen pre(stage) == 2 and (time - pre(t_stage)) >= (if pre(stage) == 2 then t_rest1 else 1.0e12) then
      stage = 3;
      t_stage = time;
    elsewhen pre(stage) == 3 and v >= (if pre(stage) == 3 then V_max else 100.0) then
      stage = 4;
      t_stage = time;
    elsewhen pre(stage) == 4 and ((time - pre(t_stage)) >= (if pre(stage) == 4 then t_cv else 1.0e12) or (-p.i) <= (if pre(stage) == 4 and (time - pre(t_stage)) >= 1.0 then I_cut else -100.0)) then
      stage = 5;
      t_stage = time;
    end when;

    0.0 = if stage == 1 then (p.i - I_dis)
          else if stage == 2 then (p.i - 0.0)
          else if stage == 3 then (p.i - (-I_chg))
          else if stage == 4 then (v - V_max)
          else (p.i - 0.0);
  end CCCVCycler;

end Interfaces;
