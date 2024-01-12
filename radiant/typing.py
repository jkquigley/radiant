from collections.abc import Callable
from numpy.typing import NDArray
from typing import TypeAlias


Function: TypeAlias = Callable[[NDArray], NDArray]
