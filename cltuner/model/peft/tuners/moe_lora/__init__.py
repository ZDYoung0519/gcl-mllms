from peft.utils import register_peft_method

from .config import MOELoRAConfig
from .layer import MoELoRALayer
from .model import MoELoRAModel


__all__ = ['MoELoRALayer', 'MOELoRAConfig', 'MoELoRAModel']


register_peft_method(
    name="moelora", config_cls=MOELoRAConfig, model_cls=MoELoRAModel, prefix="lora_", is_mixed_compatible=False
)


