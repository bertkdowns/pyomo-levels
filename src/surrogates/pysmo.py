from __future__ import annotations

import contextlib
import io
import os
import tempfile
from dataclasses import dataclass

from idaes.core.surrogate.pysmo import radial_basis_function as rbf
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock


@dataclass
class PysmoSurrogateBuilder:
    surrogate: PysmoSurrogate

    def populate_block(self, block, component_name, input_vars, output_vars):
        block.add_component(component_name, SurrogateBlock())
        getattr(block, component_name).build_model(
            self.surrogate,
            input_vars=input_vars,
            output_vars=output_vars,
        )


def build_pysmo_surrogate(
    input_labels,
    output_labels,
    input_bounds,
    training_data,
    basis_function: str = "gaussian",
    solution_method: str = "algebraic",
    regularization: bool = True,
    pysmo_tee: bool = False,
):
    with tempfile.TemporaryDirectory(prefix="pysmo-rbf-") as output_directory:
        with _maybe_suppress_stdout(pysmo_tee):
            trained = _TemporaryFilePysmoRBFTrainer(
                input_labels=input_labels,
                output_labels=output_labels,
                training_dataframe=training_data,
                input_bounds=input_bounds,
                basis_function=basis_function,
                solution_method=solution_method,
                regularization=regularization,
                output_directory=output_directory,
            ).train_surrogate()

    surrogate = PysmoSurrogate(
        trained_surrogates=trained,
        input_labels=input_labels,
        output_labels=output_labels,
        input_bounds=input_bounds,
    )
    return PysmoSurrogateBuilder(surrogate=surrogate)


class _TemporaryFilePysmoRBFTrainer(PysmoRBFTrainer):
    def __init__(self, *args, output_directory, **settings):
        self._output_directory = output_directory
        super().__init__(*args, **settings)

    def _create_model(self, pysmo_input, output_label):
        safe_label = "".join(
            character if character.isalnum() else "_"
            for character in str(output_label)
        )
        model = rbf.RadialBasisFunctions(
            pysmo_input,
            basis_function=self.config.basis_function,
            solution_method=self.config.solution_method,
            regularization=self.config.regularization,
            fname=os.path.join(self._output_directory, f"{safe_label}.pickle"),
            overwrite=True,
        )
        model.get_feature_vector()
        return model


def _maybe_suppress_stdout(enabled):
    if enabled:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())
