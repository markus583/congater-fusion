import copy
import logging
from collections.abc import Collection, Mapping
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace
from typing import List, Optional, Union

from . import __version__
from .composition import AdapterCompositionBlock
from .utils import get_adapter_config_hash, resolve_adapter_config

logger = logging.getLogger(__name__)


class AdapterConfigBase(Mapping):
    """
    Base class for all adaptation methods. This class does not define specific configuration keys, but only provides
    some common helper methods.

    Args:
        architecture (str, optional): The type of adaptation method defined by the configuration.
    """

    architecture: Optional[str] = None

    def __init__(self):
        raise TypeError(
            "AdapterConfigBase is an abstract class and cannot be instantiated."
        )

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        # if name in self.__dict__:
        #     raise FrozenInstanceError()
        # else:
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """Converts the config class to a Python dict."""
        return asdict(self)

    def replace(self, **changes):
        """Returns a new instance of the config class with the specified changes applied."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        """Creates a config class from a Python dict."""
        if isinstance(config, AdapterConfigBase):
            return config

        # the constructor does not accept additional kwargs, so add them separately
        defined_kwargs, new_kwargs = {}, {}
        for k, v in config.items():
            if k in cls.__dataclass_fields__.keys():
                defined_kwargs[k] = v
            else:
                new_kwargs[k] = v
        obj = cls(**defined_kwargs)
        for k, v in new_kwargs.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _get_config_class(config_dict):
        """
        Returns the matching config class for the given config dict based on its "architecture" key.
        """
        architecture = config_dict.get("architecture", None)
        if architecture == "prefix_tuning":
            cls_new = PrefixTuningConfig
        elif architecture == "lora":
            cls_new = LoRAConfig
        elif architecture == "union":
            cls_new = ConfigUnion
        else:
            cls_new = AdapterConfig

        return cls_new

    @classmethod
    def load(cls, config: Union[dict, str], download_kwargs=None, **kwargs):
        """
        Loads a given adapter configuration specifier into a full AdapterConfigBase instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTER_CONFIG_MAP
                - the path to a file containing a full adapter configuration
                - an identifier string available in Adapter-Hub

        Returns:
            dict: The resolved adapter configuration dictionary.
        """
        if not config:
            return None
        # if force_download is set, skip the local map
        if download_kwargs and download_kwargs.get("force_download", False):
            local_map = None
        else:
            local_map = ADAPTER_CONFIG_MAP
        if download_kwargs:
            config_dict = resolve_adapter_config(
                config, local_map=local_map, **download_kwargs
            )
        else:
            config_dict = resolve_adapter_config(config, local_map=local_map)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterConfigBase):
            cls_new = config_dict.__class__
            config_dict = config_dict.to_dict()
        else:
            cls_new = cls._get_config_class(config_dict)
        # The check for "None" is necessary because of the example script flags.
        config_dict.update((k, v) for k, v in kwargs.items() if v is not None)
        return cls_new.from_dict(config_dict)


@dataclass(eq=False)
class AdapterConfig(AdapterConfigBase):
    """
    Base class that models the architecture of an adapter.

    Args:
        mh_adapter (:obj:`bool`): If True, add adapter modules after the multi-head attention block of each layer.
        output_adapter (:obj:`bool`): If True, add adapter modules after the output FFN of each layer.
        reduction_factor (:obj:`float` or :obj:`Mapping`):
            Either a scalar float (> 0) specifying the reduction factor for all layers or a mapping from layer ID
            (starting at 0) to values specifying the reduction_factor for individual layers. If not all layers are
            represented in the mapping a default value should be given e.g. {'1': 8, '6': 32, 'default': 16}.
            Specifying a reduction factor < 1 will result in an up-projection layer.
        non_linearity (:obj:`str`): The activation function to use in the adapter bottleneck.
        original_ln_before (:obj:`bool`, optional):
            If True, apply layer pre-trained normalization and residual connection before the adapter modules. Defaults
            to False. Only applicable if :obj:`is_parallel` is False.
        original_ln_after (:obj:`bool`, optional):
            If True, apply pre-trained layer normalization and residual connection after the adapter modules. Defaults
            to True.
        ln_before (:obj:`bool`, optional): If True, add a new layer normalization before the adapter bottleneck.
            Defaults to False.
        ln_after (:obj:`bool`, optional): If True, add a new layer normalization after the adapter bottleneck.
            Defaults to False.
        init_weights (:obj:`str`, optional): Initialization method for the weights of the adapter modules.
            Currently, this can be either "bert" (default) or "mam_adapter".
        is_parallel (:obj:`bool`, optional): If True, apply adapter transformations in parallel.
            By default (False), sequential application is used.
        scaling (:obj:`float` or :obj:`str`, optional):
            Scaling factor to use for scaled addition of adapter outputs as done by He et al. (2021). Can be either a
            constant factor (float) or the string "learned", in which case the scaling factor is learned. Defaults to
            1.0.
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False.
        residual_before_ln (:obj:`bool`, optional):
            If True, take the residual connection around the adapter bottleneck before the layer normalization. Only
            applicable if :obj:`original_ln_before` is True.
        adapter_residual_before_ln (:obj:`bool`, optional):
            If True, apply the residual connection around the adapter modules before the new layer normalization within
            the adapter. Only applicable if :obj:`ln_after` is True and :obj:`is_parallel` is False.
        inv_adapter (:obj:`str`, optional):
            If not None (default), add invertible adapter modules after the model embedding layer. Currently, this can
            be either "nice" or "glow".
        inv_adapter_reduction_factor (:obj:`float`, optional):
            The reduction to use within the invertible adapter modules. Only applicable if :obj:`inv_adapter` is not
            None.
        cross_adapter (:obj:`bool`, optional):
            If True, add adapter modules after the cross attention block of each decoder layer in an encoder-decoder
            model. Defaults to False.
        leave_out (:obj:`List[int]`, optional):
            The IDs of the layers (starting at 0) where NO adapter modules should be added.
        phm_layer (:obj:`bool`, optional): If True the down and up projection layers are a PHMLayer.
            Defaults to False
        phm_dim (:obj:`int`, optional): The dimension of the phm matrix.
            Only applicable if `phm_layer` is set to `True`. Defaults to 4.
        shared_phm_rule (:obj:`bool`, optional): Whether the phm matrix is shared across all layers.
            Defaults to True
        factorized_phm_rule (:obj:`bool`, optional):
            Whether the phm matrix is factorized into a left and right matrix. Defaults to False.
        learn_phm (:obj:`bool`, optional): Whether the phm matrix should be learned during training.
            Defaults to True
        factorized_phm_W (:
            obj:`bool`, optional): Whether the weights matrix is factorized into a left and right matrix. Defaults to
            True
        shared_W_phm (:obj:`bool`, optional): Whether the weights matrix is shared across all layers.
            Defaults to False.
        phm_c_init (:obj:`str`, optional): The initialization function for the weights of the phm matrix.
            The possible values are `["normal", "uniform"]`. Defaults to `normal`.
        phm_init_range (:obj:`float`, optional): std for initializing phm weights if `phm_c_init="normal"`.
            Defaults to 0.0001.
        hypercomplex_nonlinearity (:obj:`str`, optional):
            This specifies the distribution to draw the weights in the phm layer from. Defaults to `glorot-uniform`.
        phm_rank (:obj:`int`, optional):
            If the weight matrix is factorized this specifies the rank of the matrix. E.g. the left matrix of the down
            projection has the shape (phm_dim, _in_feats_per_axis, phm_rank) and the right matrix (phm_dim, phm_rank,
            _out_feats_per_axis). Defaults to 1
        phm_bias (:obj:`bool`, optional):
            If True the down and up projection PHMLayer has a bias term. If `phm_layer` is False this is ignored.
            Defaults to True
        apply_tsigmoid (:obj:`bool`, optional):
            If True, apply a t-sigmoid function to the Adapter computation, but before the residual connection.
            Defaults to False
        only_one_w (:obj:`bool`, optional):
            If True, use only one weight matrix for the projection, as in the default ConGater setup.
            Defaults to False
        kill_adapter_residual (:obj:`bool`, optional):
            If True, do not apply the residual connection around the adapter modules.
            Defaults to False
        use_tsigmoid_gating (:obj:`str`, optional):
            If 'input', use the t-sigmoid output for gating the input.
            If 'adp', use the t-sigmoid output for gating the adapter output (i.e., before applying t-sigmoid)
            If 'adp2', use the 2nd adapter output after applying t-sigmoid for gating the adapter output
            Defaults to False
        add_second_adapter (:obj:`bool`, optional):
            If True, add another Adapter with the same setup as the first one.
            Defaults to False
        second_adapter_input (:obj:`str`, optional):
            If 'input', use the input, x, as input to the second Adapter.
            If 'adp', use the output of the first Adapter, v, as input to the second Adapter.
            Defaults to None
        omega (:obj:`float`, optional):
            If not None, use the omega value for the t-sigmoid function.
            Otherwise, use the default value of 1.0.
        use_ttsigmoid (:obj:`bool`, optional):
            If True, use the TTsigmoid function instead of the Tsigmoid function.
            Defaults to False
        variable_omega (:obj:`bool`, optional):
            If True, use a variable omega value for the t-sigmoid function.
            Otherwise, use the default value of 1.0.
    """

    # Required options
    mh_adapter: bool
    output_adapter: bool

    reduction_factor: Union[float, Mapping]
    non_linearity: str

    # Options with defaults
    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    init_weights: str = "bert"
    is_parallel: bool = False
    scaling: Union[float, str] = 1.0
    use_gating: bool = False
    residual_before_ln: bool = True
    adapter_residual_before_ln: bool = False
    inv_adapter: Optional[str] = None
    inv_adapter_reduction_factor: Optional[float] = None
    cross_adapter: bool = False
    leave_out: List[int] = field(default_factory=list)
    phm_layer: bool = False
    phm_dim: int = 4
    factorized_phm_W: Optional[bool] = True
    shared_W_phm: Optional[bool] = False
    shared_phm_rule: Optional[bool] = True
    factorized_phm_rule: Optional[bool] = False
    phm_c_init: Optional[str] = "normal"
    phm_init_range: Optional[float] = 0.0001
    learn_phm: Optional[bool] = True
    hypercomplex_nonlinearity: Optional[str] = "glorot-uniform"
    phm_rank: Optional[int] = 1
    phm_bias: Optional[bool] = True
    apply_tsigmoid: Optional[bool] = False
    only_one_w: Optional[bool] = False
    kill_adapter_residual: Optional[bool] = False
    use_tsigmoid_gating: Optional[Union[bool, str]] = False
    add_second_adapter: Optional[bool] = False
    second_adapter_input: Optional[Union[None, str]] = None
    omega: Optional[Union[float, Mapping]] = 1.0
    use_ttsigmoid: Optional[bool] = True
    variable_omega: Optional[bool] = False
    gating_type: str = "sigmoid"

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    # def __setattr__(self, name, value):
    #     if name in self.__dict__:
    #         raise FrozenInstanceError()
    #     elif name == "invertible_adapter":
    #         # This is for backwards compatibility. In v1, invertible adapters were specified in a nested config dict.
    #         # Now, we have two config keys directly in the adapter config.
    #         if value:
    #             object.__setattr__(self, "inv_adapter", value["block_type"])
    #             object.__setattr__(
    #                 self, "inv_adapter_reduction_factor", value["reduction_factor"]
    #             )
    #     else:
    #         object.__setattr__(self, name, value)


@dataclass(eq=False)
class PfeifferConfig(AdapterConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    original_ln_before: bool = True
    original_ln_after: bool = True
    residual_before_ln: bool = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 16


@dataclass(eq=False)
class OriginalCongaterConfig(PfeifferConfig):
    """
    The original ConGater architecture proposed by Masoudian et al. (2023).
    It computes: g(W_u(tanh(W_d * x)), where g = t-sigmoid
    """

    non_linearity: str = "tanh"
    apply_tsigmoid: bool = True
    # only_one_w: bool = True (no longer latest version --> no longer relevant!)
    kill_adapter_residual: bool = True
    use_tsigmoid_gating: str = "input"
    init_weights: Union[str, float] = "bert"


@dataclass(eq=False)
class CongaterConfig(PfeifferConfig):
    """
    The custom ConGater architecture proposed by xyz.
    """

    non_linearity: str = "relu"
    apply_tsigmoid: bool = True
    # only_one_w: bool = True (no longer latest version --> no longer relevant!)
    kill_adapter_residual: bool = True
    use_tsigmoid_gating: str = "input"
    init_weights: Union[str, float] = "bert"


@dataclass(eq=False)
class CongaterV2Config(PfeifferConfig):
    """
    The new ConGater architecture from now on.
    """

    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False
    use_tsigmoid_gating: str = "adp"
    init_weights: Union[str, float] = "bert"
    use_ttsigmoid: bool = True


@dataclass(eq=False)
class CongaterV0Config(PfeifferConfig):
    """
    The new ConGater architecture from now on.
    """

    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False
    init_weights: Union[str, float] = "bert"
    use_ttsigmoid: bool = True
    ln_before: bool = True
    use_tsigmoid_gating: bool = False


@dataclass(eq=False)
class CongaterV5Config(PfeifferConfig):
    """
    The new ConGater architecture from now on.
    """

    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False
    init_weights: Union[str, float] = "bert"
    use_ttsigmoid: bool = True
    use_tsigmoid_gating: bool = False
    gating_type: str = "tanh"


@dataclass(eq=False)
class CongaterV6Config(PfeifferConfig):
    """
    The new ConGater architecture from now on.
    """

    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False
    init_weights: Union[str, float] = "bert"
    use_ttsigmoid: bool = True
    use_tsigmoid_gating: bool = False
    gating_type: str = "selu"


@dataclass(eq=False)
class CongaterV7Config(PfeifferConfig):
    """
    The new ConGater architecture from now on.
    """

    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False
    init_weights: Union[str, float] = "bert"
    use_ttsigmoid: bool = True
    use_tsigmoid_gating: bool = False
    gating_type: str = "seluV2"


@dataclass(eq=False)
class CongaterV8Config(PfeifferConfig):
    """
    The new ConGater architecture from now on.
    """

    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False
    init_weights: Union[str, float] = "bert"
    use_ttsigmoid: bool = True
    use_tsigmoid_gating: str = "adp"
    gating_type: str = "tanh"


@dataclass(eq=False)
class CongaterV3Config(PfeifferConfig):
    """
    The custom ConGater architecture proposed by xyz.
    """

    non_linearity: str = "relu"
    init_weights: Union[str, float] = "bert"
    # ConGater
    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False  # + x
    use_tsigmoid_gating: str = "adp2"  # o = x + v * g
    add_second_adapter: bool = True  # fn2
    second_adapter_input: Optional[Union[None, str]] = "input"  # fn2(x)
    reduction_factor: Union[float, Mapping] = 32  # same # of params as Pfeiffer
    use_ttsigmoid: bool = True


@dataclass(eq=False)
class CongaterV4Config(PfeifferConfig):
    """
    The custom ConGater architecture proposed by xyz.
    """

    non_linearity: str = "relu"
    init_weights: Union[str, float] = "bert"
    # ConGater
    apply_tsigmoid: bool = True
    kill_adapter_residual: bool = False  # + x
    use_tsigmoid_gating: str = "adp2"  # o = x + v * g
    add_second_adapter: bool = True  # fn2
    second_adapter_input: Optional[Union[None, str]] = "input"  # fn2(x)
    reduction_factor: Union[float, Mapping] = 32  # same # of params as Pfeiffer
    second_adapter_input: Optional[Union[None, str]] = "adp"  # fn2(x)
    use_ttsigmoid: bool = True


@dataclass(eq=False)
class CompacterPlusPlusConfig(PfeifferConfig):
    """
    The Compacter++ architecture proposed by Mahabadi et al. (2021). See https://arxiv.org/pdf/2106.04647.pdf.
    """

    phm_layer: bool = True
    reduction_factor: Union[float, Mapping] = 32
    non_linearity: str = "gelu"


@dataclass(eq=False)
class PfeifferInvConfig(PfeifferConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[float] = 2


@dataclass(eq=False)
class HoulsbyConfig(AdapterConfig):
    """
    The adapter architecture proposed by Houlsby et al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
    """

    original_ln_before: bool = False
    original_ln_after: bool = True
    residual_before_ln: bool = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = True
    output_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: Union[float, Mapping] = 16


@dataclass(eq=False)
class CompacterConfig(HoulsbyConfig):
    """
    The Compacter architecture proposed by Mahabadi et al. (2021). See https://arxiv.org/pdf/2106.04647.pdf.
    """

    phm_layer: bool = True
    reduction_factor: Union[float, Mapping] = 32
    non_linearity: str = "gelu"


@dataclass(eq=False)
class HoulsbyInvConfig(HoulsbyConfig):
    """
    The adapter architecture proposed by Houlsby et. al. (2019). See https://arxiv.org/pdf/1902.00751.pdf.
    """

    inv_adapter: Optional[str] = "nice"
    inv_adapter_reduction_factor: Optional[float] = 2


@dataclass(eq=False)
class ParallelConfig(AdapterConfig):
    """
    The parallel adapter architecture proposed by He et al. (2021). See https://arxiv.org/pdf/2110.04366.pdf.
    """

    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 2

    init_weights: str = "mam_adapter"
    is_parallel: bool = True
    scaling: Union[float, str] = 4.0


@dataclass(eq=False)
class PrefixTuningConfig(AdapterConfigBase):
    """
    The Prefix Tuning architecture proposed by Li & Liang (2021). See https://arxiv.org/pdf/2101.00190.pdf.

    Args:
        encoder_prefix (bool): If True, add prefixes to the encoder of an encoder-decoder model.
        cross_prefix (bool): If True, add prefixes to the cross attention of an encoder-decoder model.
        flat (bool): If True, train the prefix parameters directly. Otherwise, reparametrize using a bottleneck MLP.
        prefix_length (int): The length of the prefix.
        bottleneck_size (int): If flat=False, the size of the bottleneck MLP.
        non_linearity (str): If flat=False, the non-linearity used in the bottleneck MLP.
        dropout (float): The dropout rate used in the prefix tuning layer.
        leave_out (List[int]): The IDs of the layers (starting at 0) where NO prefix should be added.
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False.
        shared_gating (:
            obj:`bool`, optional): Whether to use a shared gate for the prefixes of all attention matrices. Only
            applicable if `use_gating=True`. Defaults to True.
    """

    architecture: Optional[str] = "prefix_tuning"

    encoder_prefix: bool = True
    cross_prefix: bool = True
    leave_out: List[int] = field(default_factory=list)

    flat: bool = False
    prefix_length: int = 30
    bottleneck_size: int = 512
    non_linearity: str = "tanh"
    dropout: float = 0.0
    use_gating: bool = False
    shared_gating: bool = True


@dataclass(eq=False)
class LoRAConfig(AdapterConfigBase):
    """
    The Low-Rank Adaptation (LoRA) architecture proposed by Hu et al. (2021). See https://arxiv.org/pdf/2106.09685.pdf.
    LoRA adapts a model by reparametrizing the weights of a layer matrix. You can merge the additional weights with the
    original layer weights using ``model.merge_adapter("lora_name")``.

    Args:
        selfattn_lora (bool, optional): If True, add LoRA to the self-attention weights of a model.
            Defaults to True.
        intermediate_lora (bool, optional): If True, add LoRA to the intermediate MLP weights of a model.
            Defaults to False.
        output_lora (bool, optional): If True, add LoRA to the output MLP weights of a model.
            Defaults to False.
        r (int, optional): The rank of the LoRA layer. Defaults to 8.
        alpha (int, optional): The hyperparameter used for scaling the LoRA reparametrization. Defaults to 8.
        dropout (float, optional): The dropout rate used in the LoRA layer. Defaults to 0.0.
        attn_matrices (List[str], optional): Determines which matrices of the self-attention module to adapt.
            A list that may contain the strings "q" (query), "k" (key), "v" (value). Defaults to ["q", "v"].
        composition_mode (str, optional):
            Defines how the injected weights are composed with the original model weights. Can be either "add"
            (addition of decomposed matrix, as in LoRA) or "scale" (element-wise multiplication of vector, as in
            (IA)^3). "scale" can only be used together with r=1. Defaults to "add".
        init_weights (:obj:`str`, optional): Initialization method for the weights of the LoRA modules.
            Currently, this can be either "lora" (default) or "bert".
        use_gating (:obj:`bool`, optional):
            Place a trainable gating module besides the added parameter module to control module activation. This is
            e.g. used for UniPELT. Defaults to False. Note that modules with use_gating=True cannot be merged using
            `merge_adapter()`.
    """

    architecture: Optional[str] = "lora"

    selfattn_lora: bool = True
    intermediate_lora: bool = False
    output_lora: bool = False

    r: int = 8
    alpha: int = 8
    dropout: float = 0.0
    attn_matrices: List[str] = field(default_factory=lambda: ["q", "v"])
    composition_mode: str = "add"
    init_weights: str = "lora"
    use_gating: bool = False


@dataclass(eq=False)
class IA3Config(LoRAConfig):
    """
    The 'Infused Adapter by Inhibiting and Amplifying Inner Activations' ((IA)^3) architecture proposed by Liu et al.
    (2022). See https://arxiv.org/pdf/2205.05638.pdf. (IA)^3 builds on top of LoRA, however, unlike the additive
    composition of LoRA, it scales weights of a layer using an injected vector.
    """

    selfattn_lora: bool = True
    intermediate_lora: bool = True
    output_lora: bool = False

    r: int = 1
    alpha: int = 1
    dropout: float = 0.0
    attn_matrices: List[str] = field(default_factory=lambda: ["k", "v"])
    composition_mode: str = "scale"
    init_weights: str = "ia3"
    use_gating: bool = False


class ConfigUnion(AdapterConfigBase):
    """
    Composes multiple adaptation method configurations into one. This class can be used to define complex adaptation
    method setups.
    """

    architecture: Optional[str] = "union"

    configs: List[AdapterConfigBase]

    def __init__(self, *configs: List[AdapterConfigBase]):
        self.validate(configs)
        self.configs = configs

    @staticmethod
    def validate(configs):
        """
        Performs simple validations of a list of configurations to check whether they can be combined to a common
        setup.

        Args:
            configs (List[AdapterConfigBase]): list of configs to check.

        Raises:
            TypeError: One of the configurations has a wrong type. ValueError: At least two given configurations
            conflict.
        """
        # perform single config checks
        for config in configs:
            if not isinstance(config, AdapterConfigBase):
                raise TypeError(f"{config} is not an instance of AdapterConfigBase")
            elif isinstance(config, ConfigUnion):
                raise TypeError(
                    f"{config} of type {type(config)} is not supported in a config union."
                )
        # perform pairwise check
        for c_a, c_b in [
            (c_a, c_b)
            for i, c_a in enumerate(configs)
            for j, c_b in enumerate(configs)
            if i > j
        ]:
            if c_a.architecture != c_b.architecture:
                continue
            # if at least one config specifies a leave_out, we cannot make a final decision at this point
            elif c_a.get("leave_out", []) or c_b.get("leave_out", []):
                continue
            elif c_a.architecture is None or c_a.architecture == "bottleneck":
                is_valid = (
                    c_a.mh_adapter != c_b.mh_adapter
                    and c_a.output_adapter != c_b.output_adapter
                )
                if not is_valid:
                    raise ValueError(f"{c_a} and {c_b} cannot be combined.")
                else:
                    continue
            # at this point, we know that the architectures are the same
            raise ValueError(
                f"{c_a} and {c_b} have the same adapter architecture and cannot be combined."
            )

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.configs[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            i, k = key.split(".")
            return self.configs[int(i)][k]

    def __iter__(self):
        for i, c in enumerate(self.configs):
            for k in iter(c):
                yield f"{i}.{k}"

    def __len__(self):
        return sum([len(c) for c in self.configs])

    def __eq__(self, other):
        return all([c_a == c_b for c_a, c_b in zip(self.configs, other.configs)])

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "configs": [c.to_dict() for c in self.configs],
        }

    def replace(self, **changes):
        return ConfigUnion(*[c.replace(**changes) for c in self.configs])

    @classmethod
    def from_dict(cls, config):
        if isinstance(config, AdapterConfigBase):
            return config

        configs = []
        for c in config["configs"]:
            config_class = cls._get_config_class(c)
            configs.append(config_class.from_dict(c))

        return cls(*configs)


class MAMConfig(ConfigUnion):
    """
    The Mix-And-Match adapter architecture proposed by He et al. (2021). See https://arxiv.org/pdf/2110.04366.pdf.
    """

    def __init__(
        self,
        prefix_tuning: Optional[PrefixTuningConfig] = None,
        adapter: Optional[AdapterConfig] = None,
    ):
        prefix_tuning = prefix_tuning or PrefixTuningConfig(bottleneck_size=800)
        adapter = adapter or ParallelConfig()

        assert isinstance(prefix_tuning, PrefixTuningConfig)
        assert isinstance(adapter, AdapterConfig)
        super().__init__(prefix_tuning, adapter)

    @property
    def prefix_tuning(self):
        return self[0]

    @property
    def adapter(self):
        return self[1]


class UniPELTConfig(ConfigUnion):
    """
    The UniPELT adapter architecture proposed by Mao et al. (2022). See https://arxiv.org/pdf/2110.07577.pdf.
    """

    def __init__(
        self,
        prefix_tuning: Optional[PrefixTuningConfig] = None,
        adapter: Optional[AdapterConfig] = None,
        lora: Optional[LoRAConfig] = None,
    ):
        components = [
            prefix_tuning or PrefixTuningConfig(prefix_length=10),
            adapter or PfeifferConfig(reduction_factor=16),
            lora or LoRAConfig(r=8),
        ]

        super().__init__(*[c.replace(use_gating=True) for c in components])


# IMPORTANT: When adding a new config here, also add it to adapter_docs/overview.md!
ADAPTER_CONFIG_MAP = {
    "pfeiffer": PfeifferConfig(),
    "houlsby": HoulsbyConfig(),
    "parallel": ParallelConfig(),
    "scaled_parallel": ParallelConfig(scaling="learned"),
    "pfeiffer+inv": PfeifferInvConfig(),
    "houlsby+inv": HoulsbyInvConfig(),
    "compacter++": CompacterPlusPlusConfig(),
    "compacter": CompacterConfig(),
    "prefix_tuning": PrefixTuningConfig(),
    "prefix_tuning_flat": PrefixTuningConfig(flat=True),
    "lora": LoRAConfig(),
    "ia3": IA3Config(),
    "mam": MAMConfig(),
    "unipelt": UniPELTConfig(),
    "congater-original": OriginalCongaterConfig(),
    "congater": CongaterConfig(),
    "congaterV0": CongaterV0Config(),
    "congaterV2": CongaterV2Config(),
    "congaterV3": CongaterV3Config(),
    "congaterV4": CongaterV4Config(),
    "congaterV5": CongaterV5Config(),
    "congaterV6": CongaterV6Config(),
    "congaterV7": CongaterV7Config(),
    "congaterV8": CongaterV8Config(),
}

DEFAULT_ADAPTER_CONFIG = "pfeiffer"


class ModelAdaptersConfig(Collection):
    """This class manages the setup and configuration of adapter modules in a pre-trained model."""

    def __init__(self, **kwargs):
        adapters_list = kwargs.pop("adapters", {})
        # this is for backwards compability: in v1.x, self.adapters values had shape (<type>, <config_name>)
        adapters_list = dict(
            map(
                lambda t: (
                    t[0],
                    t[1][1] or t[1][0] if isinstance(t[1], tuple) else t[1],
                ),
                adapters_list.items(),
            )
        )
        self.adapters: Mapping[str, str] = adapters_list
        self.config_map = kwargs.pop("config_map", {})

        self.fusions: Mapping[str, str] = kwargs.pop("fusions", {})
        self.fusion_config_map = kwargs.pop("fusion_config_map", {})

        # TODO-V2 Save this with config?
        self.active_setup: Optional[AdapterCompositionBlock] = None
        self.skip_layers = None

    def __contains__(self, item):
        return item in self.adapters.keys()

    def __iter__(self):
        return iter(self.adapters)

    def __len__(self):
        return len(self.adapters)

    def get(self, adapter_name: str) -> Optional[dict]:
        """
        Gets the config dictionary for a given adapter.

        Args:
            adapter_name (str): The name of the adapter.

        Returns:
            Mapping: The adapter configuration.
        """
        if adapter_name in self.adapters:
            config_name = self.adapters[adapter_name]
            if config_name in self.config_map:
                config = self.config_map.get(config_name, None)
            else:
                config = ADAPTER_CONFIG_MAP.get(config_name, None)
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
        else:
            config = None
        return config

    def match(
        self,
        adapter_name: str,
        config_type: type,
        layer_idx: Optional[int] = None,
        location_key: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Tries to match the given criteria to an existing adapter. Return the adapter config if a match is found,
        otherwise None.
        """
        config = self.get(adapter_name)
        if config is None:
            return None
        elif not isinstance(config, AdapterConfigBase):
            config = AdapterConfigBase.load(config)

        if isinstance(config, config_type):
            leave_out = config.get("leave_out", [])
            if layer_idx is None or layer_idx not in leave_out:
                if location_key is None or config.get(location_key, False):
                    return config
        # if we have a config union, match with all child configs
        elif isinstance(config, ConfigUnion):
            results = []
            for c in config.configs:
                if isinstance(c, config_type):
                    leave_out = c.get("leave_out", [])
                    if layer_idx is None or layer_idx not in leave_out:
                        if location_key is None or c.get(location_key, False):
                            results.append(c)
            if len(results) == 1:
                return results[0]
            elif len(results) > 1:
                raise ValueError(
                    "Multiple adapter definitions conflict for adapter '{}' in layer {}. "
                    "Please make sure there is only one adaptation block used per location and adapter.".format(
                        adapter_name, layer_idx
                    )
                )

        return None

    def add(self, adapter_name: str, config: Optional[Union[str, dict]] = None):
        """
        Adds a new adapter of the name to the model config.

        Args:
            adapter_name (str): The name of the adapter.
            config (Optional[Union[str, dict]], optional): The adapter config. Defaults to None.
        """
        if adapter_name in self.adapters:
            raise ValueError(
                f"An adapter with the name '{adapter_name}' has already been added."
            )
        if config is None:
            config = DEFAULT_ADAPTER_CONFIG
        if isinstance(config, str):
            if config not in ADAPTER_CONFIG_MAP and config not in self.config_map:
                raise ValueError(f"Invalid adapter config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.config_map[config_name] = AdapterConfigBase.load(config)
        else:
            raise ValueError("Invalid adapter config: {}".format(config))
        self.adapters[adapter_name] = config_name
        logger.info(f"Adding adapter '{adapter_name}'.")

    def get_fusion(self, fusion_name: Union[str, List[str]]) -> Optional[dict]:
        """
        Gets the config dictionary for a given AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.

        Returns:
            Optional[dict]: The AdapterFusion configuration.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            config_name = self.fusions[fusion_name]
            if config_name in self.fusion_config_map:
                config = self.fusion_config_map.get(config_name, None)
            else:
                config = ADAPTERFUSION_CONFIG_MAP.get(config_name, None)
        else:
            config = None
        return config

    def get_congosition_v1(self, fusion_name: Union[str, List[str]], grid_values=None) -> Optional[dict]:
        """
        Gets the config dictionary for a given AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.

        Returns:
            Optional[dict]: The AdapterFusion configuration.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            config_name = self.fusions[fusion_name]
            if config_name in self.fusion_config_map:
                config = self.fusion_config_map.get(config_name, None)
            else:
                config = CONGOSITIONV1_CONFIG_MAP.get(config_name, None)
        else:
            config = None
        if grid_values is not None:
        # map grid values to config
            for key, value in grid_values.items():
                setattr(config, key, value)
        return config

    def add_fusion(
        self,
        fusion_name: Union[str, List[str]],
        config: Optional[Union[str, dict]] = None,
    ):
        """
        Adds a new AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.
            config (Optional[Union[str, dict]], optional): AdapterFusion config. Defaults to None.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            raise ValueError(
                f"An AdapterFusion with the name '{fusion_name}' has already been added."
            )
        if config is None:
            config = DEFAULT_ADAPTERFUSION_CONFIG
        if isinstance(config, str):
            if (
                config not in ADAPTERFUSION_CONFIG_MAP
                and config not in self.fusion_config_map
            ):
                raise ValueError(f"Invalid AdapterFusion config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.fusion_config_map[config_name] = config
        else:
            raise ValueError("Invalid AdapterFusion config: {}".format(config))
        self.fusions[fusion_name] = config_name
        logger.info(f"Adding AdapterFusion '{fusion_name}'.")

    def add_congosition_v1(
        self,
        fusion_name: Union[str, List[str]],
        config: Optional[Union[str, dict]] = None,
        grid_values: dict = None,
    ):
        """
        Adds a new AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.
            config (Optional[Union[str, dict]], optional): AdapterFusion config. Defaults to None.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            raise ValueError(
                f"An AdapterFusion with the name '{fusion_name}' has already been added."
            )
        if config is None:
            config = DEFAULT_CONGOSITION_CONFIG
            # config = DEFAULT_ADAPTERFUSION_CONFIG
        if isinstance(config, str):
            if (
                config not in CONGOSITIONV1_CONFIG_MAP
                and config not in self.fusion_config_map
            ):
                raise ValueError(f"Invalid AdapterFusion config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.fusion_config_map[config_name] = config
        else:
            raise ValueError("Invalid AdapterFusion config: {}".format(config))
        self.fusions[fusion_name] = config_name
        logger.info(f"Adding Congosition '{fusion_name}'.")

    def common_config_value(self, adapter_names: list, attribute: str):
        """
        Checks whether all adapters in a list share the same config setting for a given attribute and returns the
        shared value.

        Args:
            adapter_names (list): The adapters to check.
            attribute (str): The config attribute to check.
        """
        common_value = None
        for i, name in enumerate(adapter_names):
            config = self.get(name)
            if not config:
                raise ValueError(
                    f"No adapter with name '{name}' found. Make sure that an adapter with this name is loaded."
                )
            config_value = config.get(attribute, None)
            if i > 0 and config_value != common_value:
                raise ValueError(
                    f"All given adapters must define the same value for config attribute {attribute}."
                )
            common_value = config_value
        return common_value

    def to_dict(self):
        output_dict = {}
        output_dict["adapters"] = copy.deepcopy(self.adapters)
        output_dict["config_map"] = {}
        for k, v in self.config_map.items():
            if isinstance(v, AdapterConfigBase):
                output_dict["config_map"][k] = v.to_dict()
            else:
                output_dict["config_map"][k] = copy.deepcopy(v)
        output_dict["fusions"] = copy.deepcopy(self.fusions)
        output_dict["fusion_config_map"] = {}
        for k, v in self.fusion_config_map.items():
            if isinstance(v, AdapterConfigBase):
                output_dict["fusion_config_map"][k] = v.to_dict()
            else:
                output_dict["fusion_config_map"][k] = copy.deepcopy(v)
        return output_dict


def build_full_config(adapter_config, model_config, save_id2label=False, **kwargs):
    config_dict = {
        "model_type": model_config.model_type,
        # some models such as encoder-decoder don't have a model-wide hidden size
        "hidden_size": getattr(model_config, "hidden_size", None),
    }
    config_dict.update(kwargs)
    if not hasattr(model_config, "prediction_heads") and save_id2label:
        config_dict["label2id"] = model_config.label2id
    if isinstance(adapter_config, AdapterConfigBase):
        config_dict["config"] = adapter_config.to_dict()
    else:
        config_dict["config"] = adapter_config
    config_dict["version"] = __version__
    return config_dict


@dataclass(eq=False)
class AdapterFusionConfig(AdapterConfigBase):
    """Base class that models the architecture of an adapter fusion layer."""

    key: bool
    query: bool
    value: bool
    query_before_ln: bool
    regularization: bool
    residual_before: bool
    temperature: bool
    value_before_softmax: bool    
    omega: bool = False
    w_omega: bool = False
    w_omega_input: str = "key"
    w_omega_sigmoid: bool = False
    omega_init: Union[None, float] = None
    clamp_omega: Union[None, float] = None
    ttsigmoid_omega_offset: Union[None, float] = None
    value_initialized: float = 1.0
    learn_omega: bool = False
    congaterV2: bool = False
    exclude_target_adapter: bool = False
    softmax: bool = True
    tttanh: bool = False
    tttanhV2: bool = False
    ttsigmoid: bool = False
    adapter_skip_tt: bool = False
    add_dk_scaling: bool = False
    value_initialized_normal: bool = False
    target_adapter_tanh: bool = False
    omega_init_12_only: bool = False
    omega_init_BIG: bool = False
    omega_init_MID: bool = False
    diff_lr: bool = False
    w_omega_softmax: bool = False
    tttanhV3: bool = False
    att_scores_as_omega: bool = False
    att_scores_as_omega_MEAN: bool = False

    @classmethod
    def load(cls, config: Union[dict, str], **kwargs):
        """
        Loads a given adapter fusion configuration specifier into a full AdapterFusionConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTERFUSION_CONFIG_MAP
                - the path to a file containing a full adapter fusion configuration

        Returns:
            dict: The resolved adapter fusion configuration dictionary.
        """
        # currently storing AdapterFusion weights on AdapterHub is not supported.
        config_dict = resolve_adapter_config(
            config, local_map=ADAPTERFUSION_CONFIG_MAP, try_loading_from_hub=False
        )
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterFusionConfig):
            config_dict = config_dict.to_dict()
        config_dict.update(kwargs)
        return AdapterFusionConfig.from_dict(config_dict)


@dataclass(eq=False)
class CongositionV1Config(AdapterConfigBase):
    """Base class that models the architecture of an adapter fusion layer."""

    fn: bool = False
    query_before_ln: bool = False
    qqp: Union[None, float] = None
    mnli: Union[None, float] = None
    qnli: Union[None, float] = None
    cb: Union[None, float] = None
    copa: Union[None, float] = None
    rte: Union[None, float] = None
    mrpc: Union[None, float] = None
    wic: Union[None, float] = None
    wsc: Union[None, float] = None
    sst2: Union[None, float] = None
    stsb: Union[None, float] = None
    boolq: Union[None, float] = None
    learn_omega: bool = False
    omega_shape: Union[None, tuple] = None
    omega_init: Union[None, float] = None
    learn_beta: bool = False
    beta_shape: Union[None, tuple] = None
    beta_init: Union[None, float] = None
    beta_first: bool = False
    sigmoid: bool = False
    sigmoid_temperature: float = 1.0
    clamp: bool = False
    uplift_target: bool = False
    ln: bool = False
    ln_before_residual: bool = True
    tanh: bool = False
    residual: bool = True
    dropout_ratio: float = 0.0
    rescaling_factor: Union[None, tuple] = None

    @classmethod
    def load(cls, config: Union[dict, str], **kwargs):
        """
        Loads a given adapter fusion configuration specifier into a full AdapterFusionConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTERFUSION_CONFIG_MAP
                - the path to a file containing a full adapter fusion configuration

        Returns:
            dict: The resolved adapter fusion configuration dictionary.
        """
        # currently storing AdapterFusion weights on AdapterHub is not supported.
        config_dict = resolve_adapter_config(
            config, local_map=CONGOSITIONV1_CONFIG_MAP, try_loading_from_hub=False
        )
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, CongositionV1Config):
            config_dict = config_dict.to_dict()
        config_dict.update(kwargs)
        return CongositionV1Config.from_dict(config_dict)


@dataclass(eq=False)
class StaticAdapterFusionConfig(AdapterFusionConfig):
    """
    Static version of adapter fusion without a value matrix. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = False
    query_before_ln: bool = False
    regularization: bool = False
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True


@dataclass(eq=False)
class DynamicAdapterFusionConfig(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True


@dataclass(eq=False)
class DynamicCongaterV5FusionConfig(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    tttanh: bool = True
    adapter_skip_tt: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV0FusionConfig(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    ttsigmoid: bool = True
    adapter_skip_tt: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV0FusionConfigOmega(DynamicCongaterV0FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmega(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigAttAsOmega(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    att_scores_as_omega: bool = True
    tttanhV2: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigAttAsOmegaMEAN(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    att_scores_as_omega: bool = True
    tttanhV2: bool = True
    att_scores_as_omega_MEAN: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigWOmega(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = False
    w_omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigWOmegaVN(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = False
    w_omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True 
    w_omega_input: str = "v_N"
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigWOmegaSigmoid(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = False
    w_omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True 
    w_omega_sigmoid: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmegaMinus1(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    omega_init: float = 0
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmegaMinus1V3(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    omega_init: float = 0
    tttanhV3: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmegaMinus1Softmax(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    omega_init: float = 0
    w_omega_softmax: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmegaMinus1_12only(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    omega_init: float = 0
    omega_init_12_only: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmegaMinus1_BIG(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    omega_init: float = 0
    omega_init_BIG: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmegaMinus1_MID(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    omega_init: float = 0
    omega_init_MID: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigOmegaDiffLR(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    tttanh: bool = False
    tttanhV2: bool = True
    omega_init: float = 0
    diff_lr: bool = True

@dataclass(eq=False)
class DynamicCongaterV0FusionConfigOmegaNormal5(DynamicCongaterV0FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    omega_init: float = 5
    
@dataclass(eq=False)
class DynamicCongaterV0FusionConfigOmegaNormal0Plus2(DynamicCongaterV0FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    omega: bool = True
    omega_init: float = 0
    ttsigmoid_omega_offset: float = 2
    
@dataclass(eq=False)
class DynamicCongaterV0FusionConfigOmegaClamp1(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    adapter_skip_tt: bool = True
    omega: bool = True
    omega_init: float = 1
    clamp_omega: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV0FusionConfigDoubleTT(DynamicCongaterV0FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    adapter_skip_tt: bool = False
    

@dataclass(eq=False)
class DynamicCongaterV5FusionConfigValueNormal(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    value_initialized_normal: bool = True


@dataclass(eq=False)
class DynamicCongaterV5FusionConfigScaleDk(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    add_dk_scaling: bool = True


@dataclass(eq=False)
class DynamicCongaterV5FusionConfigND(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    exclude_target_adapter: bool = True


@dataclass(eq=False)
class DynamicCongaterV5FusionConfigNDSigmoid(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    exclude_target_adapter: bool = True
    softmax: bool = False
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigNDSigmoidTargetTanh(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    exclude_target_adapter: bool = True
    softmax: bool = False
    target_adapter_tanh: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigDoubleTTNDSigmoidTargetTanh(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    exclude_target_adapter: bool = True
    softmax: bool = False
    adapter_skip_tt: bool = True
    
@dataclass(eq=False)
class DynamicCongaterV5FusionConfigNDSigmoidTargetTanhValueInitAvg(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """
    exclude_target_adapter: bool = True
    softmax: bool = False
    target_adapter_tanh: bool = True
    value_initialized: float = 1 / 11

@dataclass(eq=False)
class DynamicCongaterV5FusionConfigNDSigmoidValueInitAvg(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    exclude_target_adapter: bool = True
    softmax: bool = False
    value_initialized: float = 1 / 11


@dataclass(eq=False)
class DynamicCongaterV5FusionConfigValueAfter(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    value_before_softmax: bool = False


@dataclass(eq=False)
class DynamicCongaterV5FusionConfigDoubleTT(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    adapter_skip_tt: bool = False


@dataclass(eq=False)
class DynamicCongaterV5FusionConfigDoubleTTValueAfter(DynamicCongaterV5FusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    adapter_skip_tt: bool = False
    value_before_softmax: bool = False


@dataclass(eq=False)
class DynamicAdapterFusionConfigND(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: float = 1.0
    exclude_target_adapter: bool = True


@dataclass(eq=False)
class DynamicAdapterFusionConfigNDSigmoid(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: float = 1.0
    exclude_target_adapter: bool = True
    softmax: bool = False


@dataclass(eq=False)
class DynamicAdapterFusionConfigNDSigmoidValueInitAvg(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: float = 1 / 11
    exclude_target_adapter: bool = True
    softmax: bool = False


@dataclass(eq=False)
class DynamicAdapterFusionConfigCV2(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: str = True
    congaterV2: bool = True


@dataclass(eq=False)
class DynamicAdapterFusionConfigOmega(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: str = True
    learn_omega: bool = True


@dataclass(eq=False)
class StaticCongositionV1Config(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 0.5
    
@dataclass(eq=False)
class StaticCongositionV1ConfigAvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 1 / 12
    
@dataclass(eq=False)
class StaticCongositionV1ConfigSigmoid(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = -2.0
    sigmoid: bool = True
    
@dataclass(eq=False)
class StaticCongositionV1ConfigSigmoidAvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = -2.3978952728
    sigmoid: bool = True
    
# -----------------
# ElWise
# -----------------

@dataclass(eq=False)
class ElWiseCongositionV1ConfigSigmoidAvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = -2.3978952728
    sigmoid: bool = True
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClampAvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 1/12
    clamp: bool = True
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitRescale1(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    rescaling_factor: tuple = (1, 1)
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitRescale12(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    rescaling_factor: tuple = (12, 1)
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp3AvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 3/12
    clamp: bool = True
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitDropout01(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    dropout_ratio: float = 0.1
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitDropout025(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    dropout_ratio: float = 0.25
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitDropout05(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    dropout_ratio: float = 0.5
    
@dataclass(eq=False)
class ElWiseCongositionV1Config2AvgInitDropout025(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = False
    dropout_ratio: float = 0.25

@dataclass(eq=False)
class ElWiseCongositionV1ConfigClampAvgInitTanh(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 1/12
    clamp: bool = True
    tanh: bool = True
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitBetaFirstElwiseInit0(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    learn_beta: bool = True
    beta_shape: tuple = (12, 768)
    beta_init: float = 0.0
    beta_first: bool = True
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitBetaFirstSingleInit0(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    learn_beta: bool = True
    beta_shape: tuple = (12, 1)
    beta_init: float = 0.0
    beta_first: bool = True
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitBetaFirstElwiseInit0Dropout025(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    learn_beta: bool = True
    beta_shape: tuple = (12, 768)
    beta_init: float = 0.0
    beta_first: bool = True
    dropout_ratio: float = 0.25
    
@dataclass(eq=False)
class ElWiseCongositionV1ConfigClamp2AvgInitBetaAfterElwiseInit0(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 768)
    omega_init: float = 2/12
    clamp: bool = True
    learn_beta: bool = True
    beta_shape: tuple = (12, 768)
    beta_init: float = 0.0
    beta_first: bool = False
    
    
@dataclass(eq=False)
class StaticCongositionV1ConfigSigmoidMinusAvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 2.3978952728
    sigmoid: bool = True
    
@dataclass(eq=False)
class StaticCongositionV1ConfigClampAvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 1/12
    clamp: bool = True
    
@dataclass(eq=False)
class StaticCongositionV1ConfigClampAvgInitLNBefore(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 1/12
    clamp: bool = True
    ln: bool = True
    ln_before_residual: bool = True
    
@dataclass(eq=False)
class StaticCongositionV1ConfigClampAvgInitLNAfter(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 1/12
    clamp: bool = True
    ln: bool = True
    ln_before_residual: bool = False
    
@dataclass(eq=False)
class StaticCongositionV1ConfigClamp05Init(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 0.5
    clamp: bool = True
    
@dataclass(eq=False)    
class StaticCongositionV1ConfigClampMinusAvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 11/12
    clamp: bool = True
    
@dataclass(eq=False)    
class StaticCongositionV1ConfigClampAvgInitTanh(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 1/12
    clamp: bool = True
    tanh: bool = True

@dataclass(eq=False)    
class StaticCongositionV1ConfigSigmoidAvgInitTanh(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = -2.3978952728
    sigmoid: bool = True
    tanh: bool = True
    
@dataclass(eq=False)    
class StaticCongositionV1ConfigClampAvgInitNoRes(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 1/12
    clamp: bool = True
    residual: bool = False

@dataclass(eq=False)    
class StaticCongositionV1ConfigSigmoidAvgInitNoRes(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = -2.3978952728
    sigmoid: bool = True
    residual: bool = False
    
@dataclass(eq=False)
class StaticCongositionV1ConfigSigmoid05Init(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = 0.5
    sigmoid: bool = True 
    
@dataclass(eq=False)
class StaticCongositionV1ConfigSigmoid5AvgInit(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = -2.3978952728 / 5
    sigmoid: bool = True
    sigmoid_temperature: float = 5.0
    
@dataclass(eq=False)
class StaticCongositionV1ConfigSigmoidUplift(CongositionV1Config):
    learn_omega: bool = True
    omega_shape: tuple = (12, 1)
    omega_init: float = -2.0
    sigmoid: bool = True
    uplift_target: bool = True
        
@dataclass(eq=False)
class OmegaGrid(CongositionV1Config):
    fn: bool = False
    query_before_ln: bool = False
    qqp: Union[None, float] = None
    mnli: Union[None, float] = None
    qnli: Union[None, float] = None
    cb: Union[None, float] = None
    copa: Union[None, float] = None
    rte: Union[None, float] = None
    mrpc: Union[None, float] = None
    wic: Union[None, float] = None
    wsc: Union[None, float] = None
    sst2: Union[None, float] = None
    stsb: Union[None, float] = None
    boolq: Union[None, float] = None


ADAPTERFUSION_CONFIG_MAP = {
    "dynamic_omega": DynamicAdapterFusionConfigOmega(),
    "dynamic_cv2": DynamicAdapterFusionConfigCV2(),
    #
    "static": StaticAdapterFusionConfig(),
    "dynamic": DynamicAdapterFusionConfig(),
    "dynamic_nd": DynamicAdapterFusionConfigND(),
    #
    "dynamic_nd_sigmoid": DynamicAdapterFusionConfigNDSigmoid(),
    "dynamic_nd_sigmoid_value_init_avg": DynamicAdapterFusionConfigNDSigmoidValueInitAvg(),
    #
    "dynamic_congaterV5": DynamicCongaterV5FusionConfig(),
    "dynamic_congaterV5_value_after": DynamicCongaterV5FusionConfigValueAfter(),
    "dynamic_congaterV0": DynamicCongaterV0FusionConfig(),
    #
    "dynamic_congaterV5_double_tt": DynamicCongaterV5FusionConfigDoubleTT(),
    "dynamic_congaterV5_double_tt_value_after": DynamicCongaterV5FusionConfigDoubleTTValueAfter(),
    "dynamic_congaterV0_double_tt": DynamicCongaterV0FusionConfigDoubleTT(),
    #
    "dynamic_congaterV5_nd": DynamicCongaterV5FusionConfigND(),
    "dynamic_congaterV5_nd_sigmoid": DynamicCongaterV5FusionConfigNDSigmoid(),
    "dynamic_congaterV5_nd_sigmoid_value_init_avg": DynamicCongaterV5FusionConfigNDSigmoidValueInitAvg(),
    "dynamic_congaterV5_nd_sigmoid_target_tanh": DynamicCongaterV5FusionConfigNDSigmoidTargetTanh(),
    "dynamic_congaterV5_double_tt_nd_sigmoid_target_tanh": DynamicCongaterV5FusionConfigDoubleTTNDSigmoidTargetTanh(),
    "dynamic_congaterV5_nd_sigmoid_value_init_avg_target_tanh": DynamicCongaterV5FusionConfigNDSigmoidTargetTanhValueInitAvg(),
    #
    "dynamic_congaterV5_scale_dk": DynamicCongaterV5FusionConfigScaleDk(),
    "dynamic_congaterV5_value_normal": DynamicCongaterV5FusionConfigValueNormal(),
    #
    "dynamic_congaterV0_omega": DynamicCongaterV0FusionConfigOmega(),
    "dynamic_congaterV0_omega_normal5": DynamicCongaterV0FusionConfigOmegaNormal5(),
    "dynamic_congaterV0_omega_clamp1": DynamicCongaterV0FusionConfigOmegaClamp1(),
    "dynamic_congaterV0_omega_normal0_plus2": DynamicCongaterV0FusionConfigOmegaNormal0Plus2(),
    "dynamic_congaterV5_omega": DynamicCongaterV5FusionConfigOmega(),
    "dynamic_congaterV5_omega_normal0_minus1": DynamicCongaterV5FusionConfigOmegaMinus1(),
    "dynamic_congaterV5_omega_normal0_minus1_softmax": DynamicCongaterV5FusionConfigOmegaMinus1Softmax(),
    "dynamic_congaterV5_omega_normal0_minus1_12only": DynamicCongaterV5FusionConfigOmegaMinus1_12only(),
    "dynamic_congaterV5_omega_normal0_minus1_BIG": DynamicCongaterV5FusionConfigOmegaMinus1_BIG(),
    "dynamic_congaterV5_omega_normal0_minus1_MID": DynamicCongaterV5FusionConfigOmegaMinus1_MID(),
    "dynamic_congaterV5_omega_normal0_minus1_difflr": DynamicCongaterV5FusionConfigOmegaDiffLR(),
    "dynamic_congaterV5_omega_normal0_minus1_V3": DynamicCongaterV5FusionConfigOmegaMinus1V3(),
    "dynamic_congaterV5_omega_normal0_minus1_att-as-omega": DynamicCongaterV5FusionConfigAttAsOmega(),
    "dynamic_congaterV5_omega_normal0_minus1_att-as-omega-MEAN": DynamicCongaterV5FusionConfigAttAsOmegaMEAN(),

    #
    "dynamic_congaterV5_Womega": DynamicCongaterV5FusionConfigWOmega(),
    "dynamic_congaterV5_WomegaVN": DynamicCongaterV5FusionConfigWOmegaVN(),
    "dynamic_congaterV5_Womega_sigmoid": DynamicCongaterV5FusionConfigWOmegaSigmoid(),

}

CONGOSITIONV1_CONFIG_MAP = {
    "param_direct": StaticCongositionV1Config(),
    # "dynamic":
    "omega_grid": OmegaGrid(),
    "param_direct_sigmoid": StaticCongositionV1ConfigSigmoid(), 
    "param_direct_sigmoid_uplift": StaticCongositionV1ConfigSigmoidUplift(),
    "param_direct_sigmoid_avg-init": StaticCongositionV1ConfigSigmoidAvgInit(),
    "param_direct_clamp_avg-init": StaticCongositionV1ConfigClampAvgInit(),
    "param_direct_clamp_avg-init_LN-before": StaticCongositionV1ConfigClampAvgInitLNBefore(),
    "param_direct_clamp_avg-init_LN-after": StaticCongositionV1ConfigClampAvgInitLNAfter(),
    "param_direct_avg-init": StaticCongositionV1ConfigAvgInit(),
    "param_direct_sigmoid5_avg-init": StaticCongositionV1ConfigSigmoid5AvgInit(),
    "param_direct_clamp_05-init": StaticCongositionV1ConfigClamp05Init(),
    "param_direct_sigmoid_05-init": StaticCongositionV1ConfigSigmoid05Init(),
    "param_direct_sigmoid_minus-avg-init": StaticCongositionV1ConfigSigmoidMinusAvgInit(),
    "param_direct_clamp_minus-avg-init": StaticCongositionV1ConfigClampMinusAvgInit(),
    "param_direct_clamp_avg-init-tanh": StaticCongositionV1ConfigClampAvgInitTanh(),
    "param_direct_sigmoid_avg-init-tanh": StaticCongositionV1ConfigSigmoidAvgInitTanh(),
    "param_direct_clamp_avg-init-no_res": StaticCongositionV1ConfigClampAvgInitNoRes(),
    "param_direct_sigmoid_avg-init-no_res": StaticCongositionV1ConfigSigmoidAvgInitNoRes(),
    # ELWISE
    "param_elwise_sigmoid_avg-init": ElWiseCongositionV1ConfigSigmoidAvgInit(),
    "param_elwise_clamp_avg-init": ElWiseCongositionV1ConfigClampAvgInit(),
    "param_elwise_clamp_avg-init-tanh": ElWiseCongositionV1ConfigClampAvgInitTanh(),
    "param_elwise_clamp_2avg-init": ElWiseCongositionV1ConfigClamp2AvgInit(),
    "param_elwise_clamp_3avg-init": ElWiseCongositionV1ConfigClamp3AvgInit(),
    "param_elwise_clamp_2avg-init-dropout01": ElWiseCongositionV1ConfigClamp2AvgInitDropout01(),
    "param_elwise_clamp_2avg-init-dropout025": ElWiseCongositionV1ConfigClamp2AvgInitDropout025(),
    "param_elwise_clamp_2avg-init-dropout05": ElWiseCongositionV1ConfigClamp2AvgInitDropout05(),
    "param_elwise_clamp_2avg-init-rescale1": ElWiseCongositionV1ConfigClamp2AvgInitRescale1(),
    "param_elwise_clamp_2avg-init-rescale12": ElWiseCongositionV1ConfigClamp2AvgInitRescale12(),
    "param_elwise_2avg-init-dropout025": ElWiseCongositionV1Config2AvgInitDropout025(),
    # ELWISE + BETA
    "param_elwise_clamp_2avg-init-BETA_elwise-first-0": ElWiseCongositionV1ConfigClamp2AvgInitBetaFirstElwiseInit0(),
    "param_elwise_clamp_2avg-init-BETA_elwise-after-0": ElWiseCongositionV1ConfigClamp2AvgInitBetaAfterElwiseInit0(),
    "param_elwise_clamp_2avg-init-BETA_elwise-first-0-dropout025": ElWiseCongositionV1ConfigClamp2AvgInitBetaFirstElwiseInit0Dropout025(),
    "param_elwise_clamp_2avg-init-BETA_single-first-0": ElWiseCongositionV1ConfigClamp2AvgInitBetaFirstSingleInit0(),
    

}
# for each entry in the map, add another key with key-difflr and the same config class
map_copy = CONGOSITIONV1_CONFIG_MAP.copy()
for k, v in map_copy.items():
    CONGOSITIONV1_CONFIG_MAP[k + "-difflr"] = v

DEFAULT_ADAPTERFUSION_CONFIG = "dynamic"
DEFAULT_CONGOSITION_CONFIG = "static"
