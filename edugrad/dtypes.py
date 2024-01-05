"""Defines the allowed datatypes for intializing and casting Tensors.

For simplicity we only use bool, int32 and float32. Note that after applying operations, the results are usually
float32 (see `data.TensorData.elementwise()`).

"""
from typing import ClassVar, Dict, Optional, Final
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class DType:
    """Data type class for managing different data types."""

    priority: int  # Priority for upcasting
    itemsize: int  # Size of the data type in bytes
    name: str  # Name of the data type
    np: Optional[type]  # Corresponding numpy data type
    sz: int = 1  # Size factor

    def __repr__(self):
        return f"dtypes.{self.name}"


class dtypes:
    """Container for different data types and utility methods.
    We need this because some layer operation might use different trade-offs between precision and efficiency. In such
    cases, we have to translate b/w dtypes.
    """

    @staticmethod
    def is_int(x: DType) -> bool:
        """Check if a data type is an integer type."""
        return x in (
            dtypes.int32,
        )

    @staticmethod
    def is_float(x: DType) -> bool:
        """Check if a data type is a float type."""
        return x in (dtypes.float32)

    @staticmethod
    def from_np(x) -> DType:
        """Convert a numpy data type to a DType."""
        return DTYPES_DICT[np.dtype(x).name]

    @staticmethod
    def fields() -> Dict[str, DType]:
        return DTYPES_DICT

    @staticmethod  # NOTE: isinstance(True, int) is True in python
    def from_py(x) -> DType:
        return (
            dtypes.only_float if isinstance(x, float) else dtypes.bool if isinstance(x, bool) else dtypes.only_int
        )

    # Definition of various data types
    bool: Final[DType] = DType(0, 1, "bool", np.bool_)
    float32: Final[DType] = DType(2, 4, "float", np.float32)
    int32: Final[DType] = DType(1, 4, "int", np.int32)
 
    only_float: ClassVar[DType] = float32
    only_int: ClassVar[DType] = int32


# Dictionary mapping data type names to DType objects
DTYPES_DICT = {
    k: v
    for k, v in dtypes.__dict__.items()
    if not k.startswith("__")  and not k.startswith("only") and not callable(v) and not v.__class__ == staticmethod
}
