import pandas as pd
from pyomo.environ import Block, ConcreteModel, Constraint, Var, value, SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.opt.results import SolverResults

from src.surrogates.generation import build_surrogate


class CubeRootSolver:
    """Analytic stand-in for an NLP solver for the test equation y == x**3."""

    def __init__(self):
        self.options = {}

    def solve(self, blk, tee=False):
        blk.x.set_value(value(blk.y) ** (1 / 3))

        results = SolverResults()
        results.solver.status = SolverStatus.ok
        results.solver.termination_condition = TerminationCondition.optimal
        return results


def make_cube_block():
    model = ConcreteModel()
    model.unit = Block()
    model.unit.x = Var(bounds=(0, 2), initialize=1)
    model.unit.y = Var(bounds=(0, 8), initialize=1)
    model.unit.y.fix(1)
    model.unit.cube = Constraint(expr=model.unit.y == model.unit.x**3)
    return model


def main():
    model = make_cube_block()
    surrogate_block = build_surrogate(
        model.unit,
        n_samples=20,
        solver=CubeRootSolver(),
        seed=1,
        backend="omlt",
        #basis_function="gaussian",
        #regularization=True,
    )

    m = ConcreteModel()
    m.unit = surrogate_block
    m.unit.y.unfix()
    m.unit.x.fix(1.5)

    solver = SolverFactory("ipopt")
    solver.solve(m)

    print(value(m.unit.y))
    print(value(m.unit.x))


if __name__ == "__main__":
    main()
