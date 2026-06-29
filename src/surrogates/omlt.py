from __future__ import annotations

import tempfile
from dataclasses import dataclass

import pandas as pd
from pyomo.environ import Constraint


@dataclass
class OmltSurrogateBuilder:
    torch_model: object
    network_definition: object
    input_labels: list[str]
    output_labels: list[str]
    input_bounds: dict[str, tuple[float, float]]

    def populate_block(self, block, component_name, input_vars, output_vars):
        try:
            from omlt import OmltBlock
            from omlt.neuralnet import FullSpaceNNFormulation
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The OMLT backend requires the optional 'omlt' dependency."
            ) from exc

        block.add_component(component_name, OmltBlock())
        omlt_block = getattr(block, component_name)
        omlt_block.build_formulation(FullSpaceNNFormulation(self.network_definition))

        input_link_name = f"{component_name}_input_link"
        output_link_name = f"{component_name}_output_link"

        def input_link_rule(_, index):
            return omlt_block.inputs[index] == input_vars[index]

        def output_link_rule(_, index):
            return output_vars[index] == omlt_block.outputs[index]

        block.add_component(
            input_link_name,
            Constraint(range(len(input_vars)), rule=input_link_rule),
        )
        block.add_component(
            output_link_name,
            Constraint(range(len(output_vars)), rule=output_link_rule),
        )

    def evaluate_surrogate(self, inputs: pd.DataFrame):
        torch = _import_torch()
        with torch.no_grad():
            x = torch.as_tensor(
                inputs[self.input_labels].to_numpy(),
                dtype=torch.float64,
            )
            y = self.torch_model(x).detach().numpy()
        return pd.DataFrame(y, index=inputs.index, columns=self.output_labels)


def build_omlt_surrogate(
    input_labels,
    output_labels,
    input_bounds,
    training_data,
    seed: int | None = None,
    training_method: str = "lstsq",
    learning_rate: float = 1e-2,
    epochs: int = 2000,
):
    torch = _import_torch()
    network_definition_loader = _import_onnx_loader()

    x_data = torch.as_tensor(
        training_data[input_labels].to_numpy(),
        dtype=torch.float64,
    )
    y_data = torch.as_tensor(
        training_data[output_labels].to_numpy(),
        dtype=torch.float64,
    )

    if seed is not None:
        torch.manual_seed(seed)

    torch_model = torch.nn.Linear(len(input_labels), len(output_labels)).double()
    if training_method == "lstsq":
        _fit_linear_lstsq(torch_model, x_data, y_data)
    elif training_method == "sgd":
        _fit_linear_sgd(torch_model, x_data, y_data, learning_rate, epochs)
    else:
        raise ValueError("training_method must be 'lstsq' or 'sgd'.")

    network_definition = _load_omlt_network_definition(
        torch=torch,
        torch_model=torch_model,
        network_definition_loader=network_definition_loader,
        n_inputs=len(input_labels),
        input_bounds=input_bounds,
        input_labels=input_labels,
    )

    return OmltSurrogateBuilder(
        torch_model=torch_model,
        network_definition=network_definition,
        input_labels=list(input_labels),
        output_labels=list(output_labels),
        input_bounds=dict(input_bounds),
    )


def _fit_linear_lstsq(model, x_data, y_data):
    torch = _import_torch()
    ones = torch.ones((x_data.shape[0], 1), dtype=x_data.dtype)
    design_matrix = torch.cat((x_data, ones), dim=1)
    solution = torch.linalg.lstsq(design_matrix, y_data).solution
    weight = solution[:-1, :].T
    bias = solution[-1, :]

    with torch.no_grad():
        model.weight.copy_(weight)
        model.bias.copy_(bias)


def _fit_linear_sgd(model, x_data, y_data, learning_rate, epochs):
    torch = _import_torch()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_data), y_data)
        loss.backward()
        optimizer.step()


def _load_omlt_network_definition(
    torch,
    torch_model,
    network_definition_loader,
    n_inputs,
    input_bounds,
    input_labels,
):
    dummy_input = torch.zeros(n_inputs, dtype=torch.float64)
    scaled_input_bounds = {
        index: input_bounds[label]
        for index, label in enumerate(input_labels)
    }

    with tempfile.NamedTemporaryFile(suffix=".onnx") as onnx_file:
        torch.onnx.export(
            torch_model,
            dummy_input,
            onnx_file.name,
            input_names=["input"],
            output_names=["output"],
        )
        return network_definition_loader(
            onnx_file.name,
            input_bounds=scaled_input_bounds,
        )


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The OMLT backend requires the optional 'torch' dependency."
        ) from exc
    return torch


def _import_onnx_loader():
    try:
        from omlt.io import load_onnx_neural_network
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The OMLT backend requires the optional 'omlt' dependency."
        ) from exc
    return load_onnx_neural_network
