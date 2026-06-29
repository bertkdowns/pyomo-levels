from __future__ import annotations

from pyomo.environ import Block, ConstraintList, Var


def connect_block(block_1: Block, block_2: Block) -> ConstraintList:
    """Build equality constraints connecting matching top-level variables.

    For every top-level variable data object in ``block_1``, this creates a
    constraint enforcing that the variable with the same component name and
    index in ``block_2`` equals the variable in ``block_1``.
    """
    connections = ConstraintList()

    for var_1_component in block_1.component_objects(Var, descend_into=False):
        var_2_component = block_2.find_component(var_1_component.local_name)
        if var_2_component is None:
            raise KeyError(
                f"Could not find top-level variable "
                f"{var_1_component.local_name!r} in block_2."
            )
        if not isinstance(var_2_component, Var):
            raise TypeError(
                f"Component {var_1_component.local_name!r} in block_2 is not a Var."
            )

        for index in var_1_component:
            if index not in var_2_component:
                raise KeyError(
                    f"Could not find index {index!r} for variable "
                    f"{var_1_component.local_name!r} in block_2."
                )
            connections.add(
                var_2_component[index] == var_1_component[index]
            )

    return connections