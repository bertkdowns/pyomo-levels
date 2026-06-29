from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from collections.abc import Callable, Mapping

import pandas as pd
from idaes.core.surrogate.pysmo.sampling import LatinHypercubeSampling
from pyomo.common.modeling import unique_component_name
from pyomo.environ import Block, SolverFactory, Var, value
from pyomo.opt import SolverStatus, TerminationCondition


@dataclass(frozen=True)
class BlockSurrogateMetadata:
    input_vars: list
    input_labels: list[str]
    output_vars: list
    output_labels: list[str]
    input_bounds: dict[str, tuple[float, float]]


def scan_block_variables(blk, input_bounds: Mapping | None = None):
    top_level_vars = list(blk.component_data_objects(Var, descend_into=False))
    input_vars = [var for var in top_level_vars if var.fixed]
    output_vars = [var for var in top_level_vars if not var.fixed]

    if not input_vars:
        raise ValueError("No top-level fixed variables found to use as inputs.")
    if not output_vars:
        raise ValueError("No top-level unfixed variables found to use as outputs.")

    input_labels = [_local_name(var, blk) for var in input_vars]
    output_labels = [_local_name(var, blk) for var in output_vars]
    bounds = {
        label: _input_bounds_for(var, label, input_bounds)
        for var, label in zip(input_vars, input_labels, strict=True)
    }

    return BlockSurrogateMetadata(
        input_vars=input_vars,
        input_labels=input_labels,
        output_vars=output_vars,
        output_labels=output_labels,
        input_bounds=bounds,
    )


def latin_hypercube_samples(input_labels, input_bounds, n_samples, seed, tee=False):
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")

    lower_bounds = [input_bounds[label][0] for label in input_labels]
    upper_bounds = [input_bounds[label][1] for label in input_labels]

    with _maybe_suppress_stdout(tee):
        sampler = LatinHypercubeSampling(
            [lower_bounds, upper_bounds],
            number_of_samples=n_samples,
            sampling_type="creation",
            rand_seed=seed,
        )
        samples = sampler.sample_points()

    return pd.DataFrame(samples, columns=input_labels)


def collect_training_data(
    blk,
    input_vars,
    input_labels,
    output_vars,
    output_labels,
    samples,
    solver,
    solve_target,
    tee,
    solver_options,
):
    solve_target = blk if solve_target is None else solve_target
    solver = SolverFactory("ipopt") if solver is None else solver
    if isinstance(solver, str):
        solver = SolverFactory(solver)
    if solver_options is not None:
        solver.options.update(dict(solver_options))

    original_input_state = [
        (var, value(var, exception=False), var.fixed) for var in input_vars
    ]
    rows = []
    try:
        for sample_number, sample in samples.iterrows():
            for var, label in zip(input_vars, input_labels, strict=True):
                var.fix(float(sample[label]))

            result = solver.solve(solve_target, tee=tee)
            _raise_for_failed_solve(result, sample_number)

            row = {label: float(sample[label]) for label in input_labels}
            for var, label in zip(output_vars, output_labels, strict=True):
                row[label] = value(var)
            rows.append(row)
    finally:
        for var, original_value, was_fixed in original_input_state:
            if original_value is not None:
                var.set_value(original_value, skip_validation=True)
            if was_fixed:
                var.fix()
            else:
                var.unfix()

    return pd.DataFrame(rows, columns=[*input_labels, *output_labels])


def build_surrogate_block(
    source_block,
    input_labels,
    output_labels,
    populate_surrogate: Callable,
):
    new_block = Block(concrete=True)
    for var_component in source_block.component_objects(Var, descend_into=False):
        new_block.add_component(
            var_component.local_name,
            _copy_var_component(var_component),
        )

    new_vars_by_label = {
        _local_name(var, new_block): var
        for var in new_block.component_data_objects(Var, descend_into=False)
    }
    input_vars = [new_vars_by_label[label] for label in input_labels]
    output_vars = [new_vars_by_label[label] for label in output_labels]

    surrogate_component_name = unique_component_name(new_block, "surrogate")
    populate_surrogate(new_block, surrogate_component_name, input_vars, output_vars)
    return new_block


def _copy_var_component(var_component):
    if var_component.is_indexed():
        copied = Var(list(var_component.keys()))
    else:
        copied = Var()
    copied.construct()

    for index in var_component:
        source_var = var_component[index]
        target_var = copied[index]
        target_var.domain = source_var.domain
        target_var.setlb(value(source_var.lb, exception=False))
        target_var.setub(value(source_var.ub, exception=False))
        source_value = value(source_var, exception=False)
        if source_value is not None:
            target_var.set_value(source_value, skip_validation=True)
        if source_var.fixed:
            target_var.fix()

    return copied


def _input_bounds_for(var, label, input_bounds):
    bounds = None
    if input_bounds is not None:
        try:
            bounds = input_bounds[var]
        except (KeyError, TypeError):
            bounds = input_bounds.get(label)

    if bounds is None:
        bounds = (
            value(var.lb, exception=False),
            value(var.ub, exception=False),
        )

    if bounds[0] is None or bounds[1] is None:
        raise ValueError(
            f"Input variable {label!r} needs finite lower and upper bounds."
        )
    if bounds[0] >= bounds[1]:
        raise ValueError(
            f"Input variable {label!r} needs lower bound < upper bound; "
            f"got {bounds}."
        )
    return float(bounds[0]), float(bounds[1])


def _raise_for_failed_solve(result, sample_number):
    status = result.solver.status
    termination = result.solver.termination_condition
    acceptable = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
    }
    if status != SolverStatus.ok or termination not in acceptable:
        raise RuntimeError(
            "Sample solve failed at sample "
            f"{sample_number}: status={status}, termination={termination}."
        )


def _maybe_suppress_stdout(enabled):
    if enabled:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def _local_name(var, block):
    return var.getname(fully_qualified=False, relative_to=block)
