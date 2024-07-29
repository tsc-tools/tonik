import importlib
from os import PathLike
from typing import Optional

from .storage import StorageGroup, Path 
from .utils import generate_test_data


def get_data(filename: Optional[PathLike] = None) -> str:
    """Return path to tonik package.

    Parameters
    ----------
    filename : Pathlike, default None
        Append `filename` to returned path.

    Returns
    -------
    pkgdir_path

    """
    f = importlib.resources.files(__package__)
    return str(f) if filename is None else str(f / filename)