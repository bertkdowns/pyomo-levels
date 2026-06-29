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
        basis_function="gaussian",
        regularization=True,
    )

    print("Training data:")
    print(surrogate_block._training_data.round(4).to_string(index=False))

    test_inputs = pd.DataFrame({"y": [0.125, 1.0, 3.375, 8.0]})
    predictions = surrogate_block._pysmo_surrogate.evaluate_surrogate(test_inputs)

    print("\nSurrogate predictions for x = cbrt(y):")
    for y_value, predicted_x in zip(test_inputs["y"], predictions["x"], strict=True):
        actual_x = y_value ** (1 / 3)
        print(
            f"y={y_value:5.3f}  "
            f"predicted x={predicted_x:7.4f}  "
            f"actual x={actual_x:7.4f}"
        )

    m = ConcreteModel()
    m.unit = surrogate_block
    m.unit.x.fix(0.7)
    m.unit.y.unfix()

    solver = SolverFactory("ipopt")
    solver.solve(m)

    print(value(m.unit.y))


if __name__ == "__main__":
    main()
