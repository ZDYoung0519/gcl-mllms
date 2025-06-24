from dataclasses import dataclass, field
from peft.tuners import LoraConfig


@dataclass
class MOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOELoraConfig`]
    """
    expert_num: int = field(default=4)
    topk: int = field(default=4)

