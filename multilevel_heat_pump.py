from ahuora_builder.custom.thermal_utility_systems.simple_heat_pump import (
    SimpleHeatPump,
)
from ahuora_property_packages.build_package import build_package
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Compressor, HeatExchanger, Valve
from idaes.models.unit_models.heat_exchanger import HeatExchangerFlowPattern
from pyomo.environ import (
    ConcreteModel,
    SolverFactory,
    TransformationFactory,
    units as pyunits,
    value,
    Block,
    Var,
    Constraint,
    Expression
)
from pyomo.network import Arc, SequentialDecomposition
from src.hierarchy.construction import activate_hifi, activate_lofi
from src.hierarchy.connection import connect_block


def fix_water_state(port, property_package, temperature, pressure, flow_mol):
    port.flow_mol.fix(flow_mol)
    port.pressure.fix(pressure)
    port.enth_mol.fix(property_package.htpx(temperature, pressure))


m = ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.water = build_package("helmholtz", ["water"], ["Liq", "Vap"])
m.fs.butane = build_package("helmholtz", ["n-butane"], ["Liq", "Vap"])



water_side = {
    "property_package": m.fs.water,
    "has_pressure_change": False,
}
butane_side = {
    "property_package": m.fs.butane,
    "has_pressure_change": False,
}

m.fs.heat_pump = Block()
m.fs.heat_pump.lofi = Block()
m.fs.heat_pump.hifi = Block()

# IDAES unit-model heat pump using n-butane as the closed-loop working fluid:
# evaporator -> compressor -> condenser -> expansion valve -> evaporator.
m.fs.heat_pump.hifi.butane_evaporator = HeatExchanger(
    hot_side=water_side,
    cold_side=butane_side,
    flow_pattern=HeatExchangerFlowPattern.countercurrent,
)
m.fs.heat_pump.hifi.butane_compressor = Compressor(property_package=m.fs.butane)
m.fs.heat_pump.hifi.butane_condenser = HeatExchanger(
    hot_side=butane_side,
    cold_side=water_side,
    flow_pattern=HeatExchangerFlowPattern.countercurrent,
)
m.fs.heat_pump.hifi.butane_valve = Valve(property_package=m.fs.butane)

m.fs.heat_pump.hifi.butane_evaporator_to_compressor = Arc(
    source=m.fs.heat_pump.hifi.butane_evaporator.cold_side_outlet,
    destination=m.fs.heat_pump.hifi.butane_compressor.inlet,
)
m.fs.heat_pump.hifi.butane_compressor_to_condenser = Arc(
    source=m.fs.heat_pump.hifi.butane_compressor.outlet,
    destination=m.fs.heat_pump.hifi.butane_condenser.hot_side_inlet,
)
m.fs.heat_pump.hifi.butane_condenser_to_valve = Arc(
    source=m.fs.heat_pump.hifi.butane_condenser.hot_side_outlet,
    destination=m.fs.heat_pump.hifi.butane_valve.inlet,
)
m.fs.heat_pump.hifi.butane_valve_to_evaporator = Arc(
    source=m.fs.heat_pump.hifi.butane_valve.outlet,
    destination=m.fs.heat_pump.hifi.butane_evaporator.cold_side_inlet,
)

# SimpleHeatPump model operating between the same 20 C source water and
# 30 C sink water levels.
m.fs.heat_pump.lofi = SimpleHeatPump(
    source=water_side,
    sink=water_side,
)

TransformationFactory("network.expand_arcs").apply_to(m)

source_temperature = 293.15 * pyunits.K
sink_temperature = 303.15 * pyunits.K
water_pressure = 101325 * pyunits.Pa
source_water_flow = 100 * pyunits.mol / pyunits.s
sink_water_flow = 100 * pyunits.mol / pyunits.s

butane_evaporating_temperature = 288.15 * pyunits.K
butane_condensing_temperature = 308.15 * pyunits.K
butane_evaporating_pressure = 2.0e5 * pyunits.Pa
butane_condensing_pressure = 3.2e5 * pyunits.Pa
butane_flow = 10 * pyunits.mol / pyunits.s

fix_water_state(
    m.fs.heat_pump.hifi.butane_evaporator.hot_side_inlet,
    m.fs.water,
    source_temperature,
    water_pressure,
    source_water_flow,
)
fix_water_state(
    m.fs.heat_pump.hifi.butane_condenser.cold_side_inlet,
    m.fs.water,
    sink_temperature,
    water_pressure,
    sink_water_flow,
)

m.fs.heat_pump.hifi.butane_evaporator.cold_side_inlet.flow_mol[0].set_value(butane_flow)
m.fs.heat_pump.hifi.butane_evaporator.cold_side_inlet.pressure[0].set_value(
    butane_evaporating_pressure
)
m.fs.heat_pump.hifi.butane_evaporator.cold_side_inlet.enth_mol[0].set_value(
    m.fs.butane.htpx(
        butane_evaporating_temperature,
        butane_evaporating_pressure,
    )
)
m.fs.heat_pump.hifi.butane_evaporator.area.fix(10)
m.fs.heat_pump.hifi.butane_evaporator.overall_heat_transfer_coefficient[0].fix(500)
m.fs.heat_pump.hifi.butane_compressor.ratioP[0].fix(
    value(butane_condensing_pressure / butane_evaporating_pressure)
)
m.fs.heat_pump.hifi.butane_compressor.efficiency_isentropic[0].fix(0.75)
m.fs.heat_pump.hifi.butane_condenser.hot_side_outlet.enth_mol[0].set_value(
    m.fs.butane.htpx(
        butane_condensing_temperature,
        butane_condensing_pressure,
    )
)
m.fs.heat_pump.hifi.butane_condenser.area.fix(10)
m.fs.heat_pump.hifi.butane_condenser.overall_heat_transfer_coefficient[0].fix(500)

fix_water_state(
    m.fs.heat_pump.lofi.source_inlet,
    m.fs.water,
    source_temperature,
    water_pressure,
    source_water_flow,
)
fix_water_state(
    m.fs.heat_pump.lofi.sink_inlet,
    m.fs.water,
    sink_temperature,
    water_pressure,
    sink_water_flow,
)
m.fs.heat_pump.lofi.approach_temperature.fix(5 * pyunits.K)
m.fs.heat_pump.lofi.efficiency.fix(0.45)
m.fs.heat_pump.lofi.work_mechanical[0].fix(20_000 * pyunits.W)

assert degrees_of_freedom(m) == 0







seq = SequentialDecomposition()
seq.options.select_tear_method = "heuristic"
seq.options.tear_method = "Wegstein"
seq.options.iterLim = 5
seq.options.tear_set = [m.fs.heat_pump.hifi.butane_valve_to_evaporator]

tear_guesses = {
    "flow_mol": {0: value(butane_flow)},
    "pressure": {0: value(butane_evaporating_pressure)},
    "enth_mol": {
        0: value(
            m.fs.butane.htpx(
                butane_evaporating_temperature,
                butane_evaporating_pressure,
            )
        )
    },
}
seq.set_guesses_for(m.fs.heat_pump.hifi.butane_evaporator.cold_side_inlet, tear_guesses)


seq.run(m, lambda unit: unit.initialize(solver="ipopt"))
m.fs.heat_pump.lofi.initialize(solver="ipopt")

assert degrees_of_freedom(m) == 0

solver = SolverFactory("ipopt")
res = solver.solve(m, tee=True)

m.fs.heat_pump.lofi.report()
m.fs.heat_pump.hifi.butane_evaporator.report()
m.fs.heat_pump.hifi.butane_condenser.report()





# Add the global interface
hp = m.fs.heat_pump
t0 = m.fs.time.first()

hp.source_inlet_flow_mol = Var(
    initialize=value(source_water_flow), units=pyunits.mol / pyunits.s
)
hp.source_inlet_pressure = Var(initialize=value(water_pressure), units=pyunits.Pa)
hp.source_inlet_enth_mol = Var(
    initialize=value(m.fs.water.htpx(source_temperature, water_pressure)),
    units=pyunits.J / pyunits.mol,
)
hp.source_outlet_flow_mol = Var(
    initialize=value(source_water_flow), units=pyunits.mol / pyunits.s
)
hp.source_outlet_pressure = Var(initialize=value(water_pressure), units=pyunits.Pa)
hp.source_outlet_enth_mol = Var(
    initialize=value(hp.lofi.source_outlet.enth_mol[t0]),
    units=pyunits.J / pyunits.mol,
)
hp.sink_inlet_flow_mol = Var(
    initialize=value(sink_water_flow), units=pyunits.mol / pyunits.s
)
hp.sink_inlet_pressure = Var(initialize=value(water_pressure), units=pyunits.Pa)
hp.sink_inlet_enth_mol = Var(
    initialize=value(m.fs.water.htpx(sink_temperature, water_pressure)),
    units=pyunits.J / pyunits.mol,
)
hp.sink_outlet_flow_mol = Var(
    initialize=value(sink_water_flow), units=pyunits.mol / pyunits.s
)
hp.sink_outlet_pressure = Var(initialize=value(water_pressure), units=pyunits.Pa)
hp.sink_outlet_enth_mol = Var(
    initialize=value(hp.lofi.sink_outlet.enth_mol[t0]),
    units=pyunits.J / pyunits.mol,
)
hp.coefficient_of_performance = Var(initialize=2.0, units=pyunits.dimensionless)
hp.work_mechanical = Var(initialize=20_000, units=pyunits.W)

hp.source_inlet_flow_mol.fix(source_water_flow)
hp.source_inlet_pressure.fix(water_pressure)
hp.source_inlet_enth_mol.fix(m.fs.water.htpx(source_temperature, water_pressure))
hp.sink_inlet_flow_mol.fix(sink_water_flow)
hp.sink_inlet_pressure.fix(water_pressure)
hp.sink_inlet_enth_mol.fix(m.fs.water.htpx(sink_temperature, water_pressure))
hp.work_mechanical.fix(20_000 * pyunits.W)

hp.hifi.butane_evaporator.hot_side_inlet.flow_mol.unfix()
hp.hifi.butane_evaporator.hot_side_inlet.pressure.unfix()
hp.hifi.butane_evaporator.hot_side_inlet.enth_mol.unfix()
hp.hifi.butane_condenser.cold_side_inlet.flow_mol.unfix()
hp.hifi.butane_condenser.cold_side_inlet.pressure.unfix()
hp.hifi.butane_condenser.cold_side_inlet.enth_mol.unfix()
hp.lofi.source_inlet.flow_mol.unfix()
hp.lofi.source_inlet.pressure.unfix()
hp.lofi.source_inlet.enth_mol.unfix()
hp.lofi.sink_inlet.flow_mol.unfix()
hp.lofi.sink_inlet.pressure.unfix()
hp.lofi.sink_inlet.enth_mol.unfix()
hp.lofi.work_mechanical.unfix()
hp.hifi.butane_condenser.overall_heat_transfer_coefficient.unfix()

hp.hifi_source_inlet_flow_mol_eq = Constraint(
    expr=hp.hifi.butane_evaporator.hot_side_inlet.flow_mol[t0]
    == hp.source_inlet_flow_mol
)
hp.hifi_source_inlet_pressure_eq = Constraint(
    expr=hp.hifi.butane_evaporator.hot_side_inlet.pressure[t0]
    == hp.source_inlet_pressure
)
hp.hifi_source_inlet_enth_mol_eq = Constraint(
    expr=hp.hifi.butane_evaporator.hot_side_inlet.enth_mol[t0]
    == hp.source_inlet_enth_mol
)
hp.hifi_source_outlet_flow_mol_eq = Constraint(
    expr=hp.hifi.butane_evaporator.hot_side_outlet.flow_mol[t0]
    == hp.source_outlet_flow_mol
)
hp.hifi_source_outlet_pressure_eq = Constraint(
    expr=hp.hifi.butane_evaporator.hot_side_outlet.pressure[t0]
    == hp.source_outlet_pressure
)
hp.hifi_source_outlet_enth_mol_eq = Constraint(
    expr=hp.hifi.butane_evaporator.hot_side_outlet.enth_mol[t0]
    == hp.source_outlet_enth_mol
)
hp.hifi_sink_inlet_flow_mol_eq = Constraint(
    expr=hp.hifi.butane_condenser.cold_side_inlet.flow_mol[t0]
    == hp.sink_inlet_flow_mol
)
hp.hifi_sink_inlet_pressure_eq = Constraint(
    expr=hp.hifi.butane_condenser.cold_side_inlet.pressure[t0]
    == hp.sink_inlet_pressure
)
hp.hifi_sink_inlet_enth_mol_eq = Constraint(
    expr=hp.hifi.butane_condenser.cold_side_inlet.enth_mol[t0]
    == hp.sink_inlet_enth_mol
)
hp.hifi_sink_outlet_flow_mol_eq = Constraint(
    expr=hp.hifi.butane_condenser.cold_side_outlet.flow_mol[t0]
    == hp.sink_outlet_flow_mol
)
hp.hifi_sink_outlet_pressure_eq = Constraint(
    expr=hp.hifi.butane_condenser.cold_side_outlet.pressure[t0]
    == hp.sink_outlet_pressure
)
hp.hifi_sink_outlet_enth_mol_eq = Constraint(
    expr=hp.hifi.butane_condenser.cold_side_outlet.enth_mol[t0]
    == hp.sink_outlet_enth_mol
)
hp.hifi_work_mechanical_eq = Constraint(
    expr=hp.hifi.butane_compressor.work_mechanical[t0] == hp.work_mechanical
)
hp.hifi_cop_eq = Constraint(
    expr=hp.hifi.butane_condenser.cold_side.heat[t0]
    == hp.coefficient_of_performance * hp.work_mechanical
)

hp.lofi_source_inlet_flow_mol_eq = Constraint(
    expr=hp.lofi.source_inlet.flow_mol[t0] == hp.source_inlet_flow_mol
)
hp.lofi_source_inlet_pressure_eq = Constraint(
    expr=hp.lofi.source_inlet.pressure[t0] == hp.source_inlet_pressure
)
hp.lofi_source_inlet_enth_mol_eq = Constraint(
    expr=hp.lofi.source_inlet.enth_mol[t0] == hp.source_inlet_enth_mol
)
hp.lofi_source_outlet_flow_mol_eq = Constraint(
    expr=hp.lofi.source_outlet.flow_mol[t0] == hp.source_outlet_flow_mol
)
hp.lofi_source_outlet_pressure_eq = Constraint(
    expr=hp.lofi.source_outlet.pressure[t0] == hp.source_outlet_pressure
)
hp.lofi_source_outlet_enth_mol_eq = Constraint(
    expr=hp.lofi.source_outlet.enth_mol[t0] == hp.source_outlet_enth_mol
)
hp.lofi_sink_inlet_flow_mol_eq = Constraint(
    expr=hp.lofi.sink_inlet.flow_mol[t0] == hp.sink_inlet_flow_mol
)
hp.lofi_sink_inlet_pressure_eq = Constraint(
    expr=hp.lofi.sink_inlet.pressure[t0] == hp.sink_inlet_pressure
)
hp.lofi_sink_inlet_enth_mol_eq = Constraint(
    expr=hp.lofi.sink_inlet.enth_mol[t0] == hp.sink_inlet_enth_mol
)
hp.lofi_sink_outlet_flow_mol_eq = Constraint(
    expr=hp.lofi.sink_outlet.flow_mol[t0] == hp.sink_outlet_flow_mol
)
hp.lofi_sink_outlet_pressure_eq = Constraint(
    expr=hp.lofi.sink_outlet.pressure[t0] == hp.sink_outlet_pressure
)
hp.lofi_sink_outlet_enth_mol_eq = Constraint(
    expr=hp.lofi.sink_outlet.enth_mol[t0] == hp.sink_outlet_enth_mol
)
hp.lofi_work_mechanical_eq = Constraint(
    expr=hp.lofi.work_mechanical[t0] == hp.work_mechanical
)
hp.lofi_cop_eq = Constraint(
    expr=hp.lofi.coefficient_of_performance == hp.coefficient_of_performance
)

activate_hifi(m.fs.heat_pump)
assert degrees_of_freedom(m) == 0
res = solver.solve(m, tee=True)

activate_lofi(m.fs.heat_pump)
assert degrees_of_freedom(m) == 0
res = solver.solve(m, tee=True)
