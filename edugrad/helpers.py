from typing import Union, Tuple, Iterator, Any
import os
import functools
from math import prod  # noqa: F401 # pylint:disable=unused-import
from dataclasses import dataclass

shape_int = int


def dedup(x):
    """Remove duplicates from a list while retaining the order."""
    return list(dict.fromkeys(x))


def argfix(*x):
    """Ensure the argument is a tuple, even if it's a single element or a list."""
    return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x


def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
    """Create a tuple pair from a single integer or a tuple."""
    return (x,) * cnt if isinstance(x, int) else x


def flatten(list_: Iterator):
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_ for item in sublist]


def fully_flatten(l):
    return [
        item for sublist in l for item in (fully_flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])
    ]


def argsort(x):
    """Return the indices that would sort an array.

    https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

    """
    return type(x)(sorted(range(len(x)), key=x.__getitem__))


def all_int(t: Tuple[Any, ...]) -> bool:
    """Check if all elements in a tuple are integers."""
    return all(isinstance(s, int) for s in t)


def round_up(num, amt: int):
    """Round up a number to the nearest multiple of 'amt'."""
    return (num + amt - 1) // amt * amt


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):
    """Get an environment variable and convert it to the type of 'default'."""
    return type(default)(os.getenv(key, default))


# Global flags for debugging and continuous integration
DEBUG = getenv("DEBUG")
