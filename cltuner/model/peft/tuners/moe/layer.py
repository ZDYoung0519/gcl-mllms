import math
import warnings
from typing import Any, Optional, Union

import torch
from torch import nn

from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
from deepspeed.moe.layer import MoE, Experts

from .config import MOELoraConfig


class MyMoE(nn.Module):
    def __init__(self, hidden_size, expert, num_experts):
        super(MyMoE, self).__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = Experts(expert, num_experts)

    def forward(self, x):
        expert_outputs = self.experts(x)    # E, ..., d
        logits = self.router(expert_outputs)
        weights = nn.functional.softmax(logits, dim=-1)
        outputs = expert_outputs * weights
        return outputs, 0, expert_outputs.shape[0]


class MoELayer(LoraLayer):
    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        super().__init__(base_layer, ephemeral_gpu_offload, **kwargs)
        self.expert_num = getattr(kwargs, "expert_num", None)
        self.topk = getattr(kwargs, "topk", None)
        self.moe = nn.ModuleDict({})

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        super().update_layer(
            adapter_name,
            r,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            use_rslora,
            use_dora,
            lora_bias
        )

        assert self.topk <= self.expert_num

        if self.topk <= self.expert_num:
            # Sparse MoE (Deepspeed)
            self.moe[adapter_name] = MoE(
                hidden_size=self.in_features,
                expert=LoRAExpert(
                    lora_a=self.lora_A[adapter_name],
                    lora_b=self.lora_B[adapter_name],
                    scaling=self.scaling[adapter_name],
                    dropout=self.lora_dropout[adapter_name],
                ),
                num_experts=self.expert_num,
                k=self.topk
            )
        else:
            self.moe[adapter_name] = MyMoE(
                hidden_size=self.in_features,
                expert=LoRAExpert(
                    lora_a=self.lora_A[adapter_name],
                    lora_b=self.lora_B[adapter_name],
                    scaling=self.scaling[adapter_name],
                    dropout=self.lora_dropout[adapter_name],
                ),
                num_experts=self.expert_num
            )


class MoELoRALinear(nn.Module, MoELoRALayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MoELoRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue
                out, l_aux, exp_counts = self.moe[active_adapter](x)
                result += out
            result = result.to(torch_result_dtype)
        return result


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: MOELoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = MoELoRALinear(target, adapter_name, **kwargs)
    else:
        raise ValueError(
            f"Target module {target} is not supported. "
            f"Currently, only `torch.nn.Linear` are supported."
        )
    return new_module
