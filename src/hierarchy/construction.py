from __future__ import annotations

from pyomo.environ import Block, ConstraintList, Var, value
from .connection import connect_block


def build_hierarchical_block(hifi_block: Block, lofi_block: Block, base_block: Block) -> Block:
    base_block.hifi = hifi_block
    base_block.lofi = lofi_block
    base_block.hifi.connections = connect_block(base_block, hifi_block)
    base_block.lofi.connections = connect_block(base_block, lofi_block)

    return base_block

def activate_hifi(base_block: Block):
    base_block.hifi.activate()
    base_block.lofi.deactivate()

def activate_lofi(base_block: Block):
    base_block.hifi.deactivate()
    base_block.lofi.activate()


def copy_block_values(block_1: Block, block_2: Block):
    """Copy top-level variable values from ``block_1`` to matching vars in ``block_2``."""
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
            var_2_component[index].set_value(
                value(var_1_component[index], exception=False),
                skip_validation=True,
            )

    return block_2
