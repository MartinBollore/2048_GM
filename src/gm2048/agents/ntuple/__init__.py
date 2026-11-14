"""Agent n-tuple : un réseau de tables de poids qui apprend à noter les plateaux."""

from gm2048.agents.ntuple.network import (
    DEFAULT_TUPLES,
    MAX_TUPLE_LENGTH,
    NTupleNetwork,
    Tuple,
    symmetries,
    tuple_index,
)

__all__ = [
    "DEFAULT_TUPLES",
    "MAX_TUPLE_LENGTH",
    "NTupleNetwork",
    "Tuple",
    "symmetries",
    "tuple_index",
]
