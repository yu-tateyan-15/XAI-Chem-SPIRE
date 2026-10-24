"""Model for notebook-based atom-contribution experiments."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Final
import torch
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, GCNConv, GraphNorm
from torch import Tensor
from numbers import Real
from torch_geometric.nn import global_add_pool


@dataclass(frozen=True)
class _BackboneSpec:
    """Small compatibility envelope for one backbone."""

    constructor: type[nn.Module]
    supports_attention: bool
    supports_edge_features: bool
    requires_edge_features: bool
    in_channel_arg: str
    out_channel_arg: str
    heads_arg: str | None
    concat_arg: str | None
    edge_dim_arg: str | None


_FIXED_BACKBONE_SPECS: dict[str, _BackboneSpec] = {
    "gatv2": _BackboneSpec(
        constructor=GATv2Conv,
        supports_attention=True,
        supports_edge_features=True,
        requires_edge_features=False,
        in_channel_arg="in_channels",
        out_channel_arg="out_channels",
        heads_arg="heads",
        concat_arg="concat",
        edge_dim_arg="edge_dim",
    ),
    "gcn": _BackboneSpec(
        constructor=GCNConv,
        supports_attention=False,
        supports_edge_features=False,
        requires_edge_features=False,
        in_channel_arg="in_channels",
        out_channel_arg="out_channels",
        heads_arg=None,
        concat_arg=None,
        edge_dim_arg=None,
    ),
}


_BACKBONE_SPECS: dict[str, _BackboneSpec] = dict(_FIXED_BACKBONE_SPECS)


BACKBONE_NAMES: Final[tuple[str, ...]] = tuple(_BACKBONE_SPECS)


NORMALIZATION_NAMES: Final[tuple[str, ...]] = ("layernorm", "graphnorm")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _get_backbone_spec(name: str) -> _BackboneSpec:
    spec = _BACKBONE_SPECS.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown backbone {name!r}; supported backbones: {', '.join(BACKBONE_NAMES)}"
        )
    return spec


@dataclass(frozen=True)
class BackboneConfig:
    """Architecture contract for a registered atom-level PyG backbone.

    ``hidden_dim`` is the total output width. Attention backbones concatenate
    ``heads`` heads, so ``hidden_dim`` must be divisible by the effective head
    count. When ``heads`` is omitted, attention backbones use four heads.
    """

    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.0
    name: str = "gatv2"
    heads: int | None = None
    use_edge_features: bool = False
    edge_dim: int | None = None
    residual: bool = False
    normalization: str = "layernorm"
    gated_residual: bool = True
    minimum_residual_alpha: float = 0.25
    initial_residual_alpha: float = 0.5

    def __post_init__(self) -> None:
        _require_positive_int(self.input_dim, "input_dim")
        _require_positive_int(self.hidden_dim, "hidden_dim")
        _require_positive_int(self.num_layers, "num_layers")
        if self.name not in _BACKBONE_SPECS:
            raise ValueError(
                f"Unknown backbone {self.name!r}; supported backbones: {', '.join(BACKBONE_NAMES)}"
            )
        if not isinstance(self.dropout, (int, float)) or isinstance(
            self.dropout, bool
        ):
            raise TypeError("dropout must be a real number")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not isinstance(self.use_edge_features, bool):
            raise TypeError("use_edge_features must be boolean")
        if not isinstance(self.residual, bool):
            raise TypeError("residual must be boolean")
        if self.residual and self.name not in {"gat", "gatv2"}:
            raise ValueError(
                "residual is currently supported only for gat and gatv2"
            )
        if self.normalization not in NORMALIZATION_NAMES:
            raise ValueError(
                "normalization must be one of: "
                + ", ".join(NORMALIZATION_NAMES)
            )
        if not isinstance(self.gated_residual, bool):
            raise TypeError("gated_residual must be boolean")
        for value, field_name in (
            (self.minimum_residual_alpha, "minimum_residual_alpha"),
            (self.initial_residual_alpha, "initial_residual_alpha"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 < float(self.minimum_residual_alpha) < 1.0:
            raise ValueError(
                "minimum_residual_alpha must satisfy 0 < value < 1"
            )
        if not (
            float(self.minimum_residual_alpha)
            <= float(self.initial_residual_alpha)
            < 1.0
        ):
            raise ValueError(
                "initial_residual_alpha must satisfy "
                "minimum_residual_alpha <= value < 1"
            )

        spec = _get_backbone_spec(self.name)
        if self.use_edge_features:
            if not spec.supports_edge_features:
                raise ValueError(
                    f"Backbone {self.name!r} does not support edge features"
                )
            if self.edge_dim is None:
                raise ValueError(
                    "edge_dim is required when use_edge_features is true"
                )
            _require_positive_int(self.edge_dim, "edge_dim")
        elif self.edge_dim is not None:
            raise ValueError("edge_dim must be None when edge features are disabled")
        elif spec.requires_edge_features:
            raise ValueError(
                f"Backbone {self.name!r} requires use_edge_features=True"
            )

        if self.name in BACKBONE_NAMES:
            if spec.supports_attention:
                effective_heads = self.effective_heads
                if self.hidden_dim % effective_heads != 0:
                    raise ValueError(
                        "hidden_dim must be divisible by the attention head count"
                    )
            elif self.heads is not None:
                raise ValueError(
                    "heads is only valid for an attention backbone"
                )

    @property
    def effective_heads(self) -> int:
        spec = _get_backbone_spec(self.name)
        if not spec.supports_attention:
            return 1
        heads = 4 if self.heads is None else self.heads
        _require_positive_int(heads, "heads")
        return heads


def _message_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _resolve_conv_args(
    config: BackboneConfig,
    spec: _BackboneSpec,
    input_dim: int,
) -> dict[str, object]:
    if config.name in {"gin", "gine"}:
        kwargs: dict[str, object] = {
            "nn": _message_mlp(input_dim, config.hidden_dim),
        }
        if config.use_edge_features and spec.edge_dim_arg is not None:
            kwargs[spec.edge_dim_arg] = config.edge_dim
        return kwargs

    kwargs: dict[str, int | float | bool | None] = {
        spec.in_channel_arg: input_dim,
        spec.out_channel_arg: config.hidden_dim,
    }

    if spec.supports_attention and spec.heads_arg is not None:
        heads = config.effective_heads
        concat_default = True
        if spec.concat_arg is not None:
            concat_default = bool(
                inspect.signature(spec.constructor.__init__).parameters[
                    spec.concat_arg
                ].default
            )
        if concat_default:
            kwargs[spec.out_channel_arg] = config.hidden_dim // heads
        if spec.heads_arg is not None:
            kwargs[spec.heads_arg] = heads
        if spec.concat_arg is not None:
            kwargs[spec.concat_arg] = concat_default
    if config.use_edge_features and spec.edge_dim_arg is not None:
        kwargs[spec.edge_dim_arg] = config.edge_dim
    if config.residual:
        kwargs["residual"] = True
    return kwargs


def _build_conv(config: BackboneConfig, input_dim: int) -> nn.Module:
    spec = _get_backbone_spec(config.name)
    kwargs = _resolve_conv_args(config, spec, input_dim)
    # NOTE: CONTRACT in=kwargs from _resolve_conv_args; constructor errors fail-fast
    return spec.constructor(**kwargs)


def _validate_graph_inputs(
    config: BackboneConfig,
    x: Tensor,
    edge_index: Tensor,
    batch_index: Tensor,
    edge_attr: Tensor | None,
) -> None:
    if x.ndim != 2 or x.shape[1] != config.input_dim:
        raise ValueError(
            "cached_node_representations must have shape "
            f"[num_nodes, {config.input_dim}]"
        )
    if x.shape[0] == 0:
        raise ValueError("cached_node_representations requires at least one node")
    if not torch.is_floating_point(x):
        raise TypeError("cached_node_representations must be floating point")
    if not torch.isfinite(x).all():
        raise ValueError("cached_node_representations contains non-finite values")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long")
    if edge_index.device != x.device:
        raise ValueError("edge_index and node representations must share a device")
    if edge_index.numel():
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= x.shape[0]:
            raise ValueError("edge_index contains an out-of-range node index")

    if (
        batch_index.ndim != 1
        or batch_index.numel() != x.shape[0]
        or batch_index.dtype != torch.long
    ):
        raise ValueError(
            "batch_index must be a torch.long vector with one entry per node"
        )
    if batch_index.device != x.device:
        raise ValueError("batch_index and node representations must share a device")
    if torch.any(batch_index < 0):
        raise ValueError("batch_index cannot contain negative graph indices")
    graph_ids = torch.unique(batch_index, sorted=True)
    expected_ids = torch.arange(
        graph_ids.numel(),
        device=batch_index.device,
        dtype=batch_index.dtype,
    )
    if not torch.equal(graph_ids, expected_ids):
        raise ValueError("batch_index graph ids must be contiguous from zero")
    if edge_index.numel() and torch.any(
        batch_index[edge_index[0]] != batch_index[edge_index[1]]
    ):
        raise ValueError("edge_index cannot connect nodes from different graphs")

    if config.use_edge_features:
        if edge_attr is None:
            raise ValueError(
                "edge_attr is required when use_edge_features is true"
            )
        if edge_attr.ndim != 2 or edge_attr.shape != (
            edge_index.shape[1],
            config.edge_dim,
        ):
            raise ValueError(
                "edge_attr must have shape "
                f"[num_edges, {config.edge_dim}]"
            )
        if not torch.is_floating_point(edge_attr):
            raise TypeError("edge_attr must be floating point")
        if edge_attr.device != x.device:
            raise ValueError(
                "edge_attr and node representations must share a device"
            )
        if not torch.isfinite(edge_attr).all():
            raise ValueError("edge_attr contains non-finite values")


class PyGNodeBackbone(nn.Module):
    """Graph convolutions with mandatory gated skips for new training."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dim = config.hidden_dim
        input_dims = [config.input_dim] + [
            config.hidden_dim
        ] * (config.num_layers - 1)
        self.convolutions = nn.ModuleList(
            [_build_conv(config, input_dim) for input_dim in input_dims]
        )
        normalization_type = (
            nn.LayerNorm if config.normalization == "layernorm" else GraphNorm
        )
        self.normalizations = nn.ModuleList(
            [normalization_type(config.hidden_dim) for _ in input_dims]
        )
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(float(config.dropout))
        if config.gated_residual:
            self.skip_projections = nn.ModuleList(
                [
                    nn.Identity()
                    if input_dim == config.hidden_dim
                    else nn.Linear(input_dim, config.hidden_dim, bias=False)
                    for input_dim in input_dims
                ]
            )
            initial_alpha = float(config.initial_residual_alpha)
            initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
            self.residual_alpha_logits = nn.Parameter(
                torch.full((config.num_layers,), initial_logit)
            )
        else:
            # This branch exists only to reproduce pre-contract checkpoints.
            self.skip_projections = nn.ModuleList()
            self.register_parameter("residual_alpha_logits", None)

    @property
    def residual_alphas(self) -> Tensor:
        """Return the effective trainable skip weights for scientific check."""

        if self.residual_alpha_logits is None:
            return torch.empty(0)
        return torch.sigmoid(self.residual_alpha_logits).clamp_min(
            float(self.config.minimum_residual_alpha)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        _validate_graph_inputs(
            self.config,
            x,
            edge_index,
            batch_index,
            edge_attr,
        )
        messages = edge_attr if self.config.use_edge_features else None
        for layer_index, (convolution, normalization) in enumerate(zip(
            self.convolutions,
            self.normalizations,
        )):
            layer_input = x
            if self.config.use_edge_features:
                transformed = convolution(x, edge_index, messages)
            else:
                transformed = convolution(x, edge_index)
            if self.config.normalization == "graphnorm":
                transformed = normalization(transformed, batch_index)
            else:
                transformed = normalization(transformed)
            transformed = self.dropout(self.activation(transformed))
            if self.config.gated_residual:
                skip = self.skip_projections[layer_index](layer_input)
                alpha = self.residual_alphas[layer_index]
                x = alpha * skip + (1.0 - alpha) * transformed
            else:
                x = transformed
        return x


def build_backbone(config: BackboneConfig) -> PyGNodeBackbone:
    """Build one explicitly registered atom-level PyG backbone."""

    if not isinstance(config, BackboneConfig):
        raise TypeError("config must be a BackboneConfig")
    return PyGNodeBackbone(config)


@dataclass(frozen=True)
class PredictionOutput:
    """A graph prediction and its strict atom-level additive decomposition."""

    prediction: Tensor
    atom_contributions: Tensor
    node_embeddings: Tensor
    batch_index: Tensor


@dataclass(frozen=True)
class AffinePredictionOutput:
    """Raw-scale reconstruction with one graph-level affine baseline."""

    prediction: Tensor
    atom_contributions: Tensor
    atom_sum: Tensor
    baseline: Tensor
    additivity_residual: Tensor


ACTIVE_READOUT_ACTIVATION_CONTRACT = "tanh_output_leaky_hidden_v1"


LEGACY_READOUT_ACTIVATION_CONTRACT = "gelu_hidden_v1"


def _validate_batch_index(batch_index: Tensor, num_nodes: int) -> int:
    if batch_index.ndim != 1 or batch_index.numel() != num_nodes:
        raise ValueError(
            "batch_index must be one-dimensional with one entry per node"
        )
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must have dtype torch.long")
    if num_nodes == 0:
        raise ValueError("AtomSumReadout requires at least one node")
    if torch.any(batch_index < 0):
        raise ValueError("batch_index cannot contain negative graph indices")

    graph_ids = torch.unique(batch_index, sorted=True)
    expected_ids = torch.arange(
        graph_ids.numel(),
        device=batch_index.device,
        dtype=batch_index.dtype,
    )
    if not torch.equal(graph_ids, expected_ids):
        raise ValueError("batch_index graph ids must be contiguous from zero")
    return int(graph_ids.numel())


class AtomSumReadout(nn.Module):
    """Predict one scalar per atom and sum it exactly within each graph.

    The returned graph prediction is

        prediction[g] = sum(atom_contributions[i] for batch_index[i] == g).

    No graph-level context or residual term is accepted by this Interface.
    Atom contributions are predictive decomposition terms, not physical
    observables.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        bias: bool = True,
        activation_contract: str = ACTIVE_READOUT_ACTIVATION_CONTRACT,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int):
            raise TypeError("input_dim must be an integer")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim is not None:
            if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
                raise TypeError("hidden_dim must be an integer or None")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
        if num_layers is None:
            num_layers = 1 if hidden_dim is None else 2
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer or None")
        if num_layers not in {1, 2, 3}:
            raise ValueError("num_layers must be one of: 1, 2, 3")
        if num_layers == 1 and hidden_dim is not None:
            raise ValueError("hidden_dim must be None when num_layers is 1")
        if num_layers > 1 and hidden_dim is None:
            raise ValueError("hidden_dim is required when num_layers exceeds 1")
        self.num_layers = num_layers
        if activation_contract not in {
            ACTIVE_READOUT_ACTIVATION_CONTRACT,
            LEGACY_READOUT_ACTIVATION_CONTRACT,
        }:
            raise ValueError("Unsupported atom-readout activation contract")
        self.activation_contract = activation_contract

        if (
            num_layers == 1
            and activation_contract == LEGACY_READOUT_ACTIVATION_CONTRACT
        ):
            self.atom_head: nn.Module = nn.Linear(input_dim, 1, bias=bias)
        else:
            layers: list[nn.Module] = []
            in_dim = input_dim
            if num_layers == 1:
                # The final predictive projection always receives Tanh output.
                layers.append(nn.Tanh())
            else:
                for hidden_index in range(num_layers - 1):
                    layers.append(nn.Linear(in_dim, hidden_dim))
                    is_final_hidden = hidden_index == num_layers - 2
                    if (
                        activation_contract
                        == ACTIVE_READOUT_ACTIVATION_CONTRACT
                        and is_final_hidden
                    ):
                        layers.append(nn.Tanh())
                    elif (
                        activation_contract
                        == ACTIVE_READOUT_ACTIVATION_CONTRACT
                    ):
                        layers.append(nn.LeakyReLU())
                    else:
                        layers.append(nn.GELU())
                    in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1, bias=bias))
            self.atom_head = nn.Sequential(*layers)

    def initialize_output_layer_normal(self, std: Real) -> None:
        """Initialize only the final atom-contribution layer near zero."""

        if isinstance(std, bool) or not isinstance(std, Real):
            raise TypeError("std must be a real scalar")
        std_value = float(std)
        if not math.isfinite(std_value) or std_value <= 0.0:
            raise ValueError("std must be finite and positive")
        output_layer = tuple(
            module
            for module in self.atom_head.modules()
            if isinstance(module, nn.Linear)
        )[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("AtomSumReadout output layer must be linear")
        nn.init.normal_(output_layer.weight, mean=0.0, std=std_value)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        node_embeddings: Tensor,
        batch_index: Tensor,
    ) -> PredictionOutput:
        if node_embeddings.ndim != 2:
            raise ValueError("node_embeddings must have shape [num_nodes, dim]")
        if not torch.isfinite(node_embeddings).all():
            raise ValueError("node_embeddings contains non-finite values")
        if node_embeddings.device != batch_index.device:
            raise ValueError("node_embeddings and batch_index must share a device")

        num_graphs = _validate_batch_index(batch_index, node_embeddings.shape[0])
        atom_contributions = self.atom_head(node_embeddings).squeeze(-1)
        prediction = global_add_pool(
            atom_contributions,
            batch_index,
            size=num_graphs,
        )
        return PredictionOutput(
            prediction=prediction,
            atom_contributions=atom_contributions,
            node_embeddings=node_embeddings,
            batch_index=batch_index,
        )


def _as_finite_scalar(
    value: Real | Tensor,
    *,
    reference: Tensor,
    name: str,
) -> Tensor:
    scalar = torch.as_tensor(
        value,
        device=reference.device,
        dtype=reference.dtype,
    )
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not torch.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def reconstruct_affine(
    output: PredictionOutput,
    *,
    scale: Real | Tensor,
    offset: Real | Tensor,
) -> AffinePredictionOutput:
    """Map an additive standardized prediction to an affine raw target scale.

    For ``y_raw = offset + scale * y_scaled`` this returns

    ``atom_raw[i] = scale * atom_scaled[i]`` and
    ``prediction_raw[g] = offset + scale * prediction_scaled[g]``.

    The offset remains one graph-level baseline and is never divided among
    atoms. Non-affine inverse transforms must not use this function.
    """

    alpha = _as_finite_scalar(scale, reference=output.prediction, name="scale")
    beta = _as_finite_scalar(offset, reference=output.prediction, name="offset")
    if bool(alpha == 0):
        raise ValueError("scale must be non-zero")

    num_graphs = _validate_batch_index(
        output.batch_index,
        output.atom_contributions.numel(),
    )
    if output.prediction.ndim != 1 or output.prediction.numel() != num_graphs:
        raise ValueError("prediction must contain one scalar per graph")
    if output.atom_contributions.ndim != 1:
        raise ValueError("atom_contributions must be one-dimensional")
    if output.prediction.device != output.atom_contributions.device:
        raise ValueError("prediction and atom_contributions must share a device")
    if not torch.isfinite(output.prediction).all():
        raise ValueError("prediction contains non-finite values")
    if not torch.isfinite(output.atom_contributions).all():
        raise ValueError("atom_contributions contains non-finite values")

    atom_contributions = alpha * output.atom_contributions
    atom_sum = global_add_pool(
        atom_contributions,
        output.batch_index,
        size=num_graphs,
    )
    prediction = beta + alpha * output.prediction
    additivity_residual = prediction - (beta + atom_sum)
    return AffinePredictionOutput(
        prediction=prediction,
        atom_contributions=atom_contributions,
        atom_sum=atom_sum,
        baseline=beta,
        additivity_residual=additivity_residual,
    )


def validate_affine_additivity(
    output: AffinePredictionOutput,
    batch_index: Tensor,
    *,
    absolute_tolerance: Real,
    relative_tolerance: Real,
) -> None:
    """Validate raw-scale additivity with a cancellation-aware error bound.

    The relative term uses the magnitudes of the evaluated affine operands,
    rather than the possibly near-zero result after offset cancellation.
    This is a numerical validation rule only; it does not alter predictions
    or atom contributions.
    """

    num_graphs = _validate_batch_index(
        batch_index,
        output.atom_contributions.numel(),
    )
    if output.prediction.ndim != 1 or output.prediction.numel() != num_graphs:
        raise ValueError("prediction must contain one scalar per graph")
    if output.atom_contributions.ndim != 1:
        raise ValueError("atom_contributions must be one-dimensional")
    if output.prediction.device != output.atom_contributions.device:
        raise ValueError("prediction and atom_contributions must share a device")
    if output.prediction.device != batch_index.device:
        raise ValueError("prediction and batch_index must share a device")
    if not torch.isfinite(output.prediction).all():
        raise ValueError("prediction contains non-finite values")
    if not torch.isfinite(output.atom_contributions).all():
        raise ValueError("atom_contributions contains non-finite values")

    atol = _as_finite_scalar(
        absolute_tolerance,
        reference=output.prediction,
        name="absolute_tolerance",
    )
    rtol = _as_finite_scalar(
        relative_tolerance,
        reference=output.prediction,
        name="relative_tolerance",
    )
    if bool(atol < 0) or bool(rtol < 0):
        raise ValueError("Additivity tolerances cannot be negative")
    baseline = _as_finite_scalar(
        output.baseline,
        reference=output.prediction,
        name="baseline",
    )
    atom_sum = global_add_pool(
        output.atom_contributions,
        batch_index,
        size=num_graphs,
    )
    absolute_atom_sum = global_add_pool(
        output.atom_contributions.abs(),
        batch_index,
        size=num_graphs,
    )
    residual = output.prediction - (baseline + atom_sum)
    error_scale = (
        output.prediction.abs() + baseline.abs() + absolute_atom_sum
    )
    tolerance = atol + rtol * error_scale
    if torch.any(residual.abs() > tolerance):
        raise ValueError("Raw affine additivity validation failed")


class CachedRepresentationPredictor(nn.Module):
    """Apply a PyG backbone and strict atom-sum readout to cached node inputs.

    This model contains no foundation-model encoder and performs no cache
    lookup. The caller must provide an already validated tensor of cached atom
    representations.
    """

    def __init__(
        self,
        backbone_config: BackboneConfig,
        *,
        readout_hidden_dim: int | None = None,
        readout_num_layers: int | None = None,
        atom_bias: bool = True,
        readout_activation_contract: str = (
            ACTIVE_READOUT_ACTIVATION_CONTRACT
        ),
    ) -> None:
        super().__init__()
        self.backbone_config = backbone_config
        self.backbone: PyGNodeBackbone = build_backbone(backbone_config)
        self.readout = AtomSumReadout(
            self.backbone.output_dim,
            hidden_dim=readout_hidden_dim,
            num_layers=readout_num_layers,
            bias=atom_bias,
            activation_contract=readout_activation_contract,
        )

    def forward(
        self,
        cached_node_representations: Tensor,
        edge_index: Tensor,
        batch_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> PredictionOutput:
        node_embeddings = self.backbone(
            cached_node_representations,
            edge_index,
            batch_index,
            edge_attr,
        )
        return self.readout(node_embeddings, batch_index)
