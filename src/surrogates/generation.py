from __future__ import annotations

from collections.abc import Mapping

from src.surrogates.common import (
    build_surrogate_block,
    collect_training_data,
    latin_hypercube_samples,
    scan_block_variables,
)
from src.surrogates.omlt import build_omlt_surrogate
from src.surrogates.pysmo import build_pysmo_surrogate


def build_surrogate(
    blk,
    n_samples: int,
    solver=None,
    input_bounds: Mapping | None = None,
    solve_target=None,
    tee: bool = False,
    seed: int | None = None,
    backend: str = "pysmo",
    solver_options: Mapping | None = None,
    **backend_options,
):
    """Build a surrogate replacement block from one Pyomo block.

    Top-level fixed variables on ``blk`` are treated as surrogate inputs, and
    top-level unfixed variables are treated as surrogate outputs. Sub-blocks are
    not scanned for variables.

    Args:
        blk: Pyomo block to sample.
        n_samples: Number of Latin hypercube samples to solve.
        solver: Pyomo solver object or solver name. Defaults to ``"ipopt"``.
        input_bounds: Optional mapping from input labels (for example ``"x"``
            or ``"x[1]"``) or input VarData objects to ``(lower, upper)``.
        solve_target: Optional Pyomo model/block passed to the solver. Defaults
            to ``blk``.
        tee: Forward solver output.
        seed: Optional Latin hypercube random seed.
        backend: Surrogate implementation. Supported values are ``"pysmo"`` and
            ``"omlt"``.
        solver_options: Optional mapping copied to the solver options.
        backend_options: Extra options forwarded to the selected backend.

    Returns:
        A new Pyomo ``Block`` with cloned top-level variables from ``blk`` and
        backend-specific surrogate constraints that compute the original
        unfixed variables from the original fixed variables.
    """
    metadata = scan_block_variables(blk, input_bounds=input_bounds)
    samples = latin_hypercube_samples(
        metadata.input_labels,
        metadata.input_bounds,
        n_samples,
        seed,
        tee=backend_options.pop("sampling_tee", False),
    )
    training_data = collect_training_data(
        blk=blk,
        input_vars=metadata.input_vars,
        input_labels=metadata.input_labels,
        output_vars=metadata.output_vars,
        output_labels=metadata.output_labels,
        samples=samples,
        solver=solver,
        solve_target=solve_target,
        tee=tee,
        solver_options=solver_options,
    )

    backend_key = backend.lower()
    if backend_key == "pysmo":
        surrogate_builder = build_pysmo_surrogate(
            input_labels=metadata.input_labels,
            output_labels=metadata.output_labels,
            input_bounds=metadata.input_bounds,
            training_data=training_data,
            **backend_options,
        )
    elif backend_key == "omlt":
        surrogate_builder = build_omlt_surrogate(
            input_labels=metadata.input_labels,
            output_labels=metadata.output_labels,
            input_bounds=metadata.input_bounds,
            training_data=training_data,
            **backend_options,
        )
    else:
        raise ValueError(
            f"Unknown surrogate backend {backend!r}. Expected 'pysmo' or 'omlt'."
        )

    surrogate_block = build_surrogate_block(
        source_block=blk,
        input_labels=metadata.input_labels,
        output_labels=metadata.output_labels,
        populate_surrogate=surrogate_builder.populate_block,
    )
    surrogate_block._training_data = training_data
    surrogate_block._surrogate = surrogate_builder
    if backend_key == "pysmo":
        surrogate_block._pysmo_surrogate = surrogate_builder.surrogate
    elif backend_key == "omlt":
        surrogate_block._omlt_surrogate = surrogate_builder
    return surrogate_block
