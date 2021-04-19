from .LowerSetLearn import (
    LowerSetLearn,
    Oracle, OracleFunction, OracleFunctionWithMeta,
    ExtraPrec, ExtraPrec_LowerSet,
)
from .LearnModule import LearnModule

from .LevelLearn import LevelLearn
from .RandomLearn import RandomLearn, RandomLower, RandomUpper
from .GainanovSAT import GainanovSAT

Modules = {cls.__name__: cls for cls in LearnModule.__subclasses__()}
