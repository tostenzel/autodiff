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
            dtypes.int8,
            dtypes.int16,
            dtypes.int32,
            dtypes.int64,
            dtypes.uint8,
            dtypes.uint16,
            dtypes.uint32,
            dtypes.uint64,
        )

    @staticmethod
    def is_float(x: DType) -> bool:
        """Check if a data type is a float type."""
        return x in (dtypes.float16, dtypes.float32, dtypes.float64)

    @staticmethod
    def is_unsigned(x: DType) -> bool:
        """Check if a data type is an unsigned type."""
        return x in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)

    @staticmethod
    def from_np(x) -> DType:
        """Convert a numpy data type to a DType."""
        return DTYPES_DICT[np.dtype(x).name]
    
    @staticmethod
    def fields() -> Dict[str, DType]:
        return DTYPES_DICT
    
    @staticmethod  # NOTE: isinstance(True, int) is True in python
    def from_py(x) -> DType:
        return dtypes.default_float if isinstance(x, float) else dtypes.bool if isinstance(x, bool) else dtypes.default_int

    # Definition of various data types
    bool: Final[DType] = DType(0, 1, "bool", np.bool_)
    float16: Final[DType] = DType(9, 2, "half", np.float16)
    half = float16
    float32: Final[DType] = DType(10, 4, "float", np.float32)
    float = float32
    float64: Final[DType] = DType(11, 8, "double", np.float64)
    double = float64
    int8: Final[DType] = DType(1, 1, "char", np.int8)
    int16: Final[DType] = DType(3, 2, "short", np.int16)
    int32: Final[DType] = DType(5, 4, "int", np.int32)
    int64: Final[DType] = DType(7, 8, "long", np.int64)
    uint8: Final[DType] = DType(2, 1, "unsigned char", np.uint8)
    uint16: Final[DType] = DType(4, 2, "unsigned short", np.uint16)
    uint32: Final[DType] = DType(6, 4, "unsigned int", np.uint32)
    uint64: Final[DType] = DType(8, 8, "unsigned long", np.uint64)

    default_float: ClassVar[DType] = float32
    default_int: ClassVar[DType] = int32


# Dictionary mapping data type names to DType objects
DTYPES_DICT = {
    k: v
    for k, v in dtypes.__dict__.items()
    if not k.startswith("__") and not callable(v) and not v.__class__ == staticmethod
}
