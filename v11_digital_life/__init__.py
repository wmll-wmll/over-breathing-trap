"""V11 数字生命体 —— 包入口。

单场设计：全场只有一个状态（NLSE 场 ψ），没有矩阵层。
"""

from .brain import Brain
from .config import Config, DEFAULT, FieldConfig, InterfaceConfig, LifeConfig
from .field import NLSEField
from .interface import EntityInterface
from .sandbox import SandboxWorld

__all__ = [
    "Brain",
    "Config",
    "DEFAULT",
    "EntityInterface",
    "FieldConfig",
    "InterfaceConfig",
    "LifeConfig",
    "NLSEField",
    "SandboxWorld",
]
