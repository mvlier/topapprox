from .filter_image import *
from .filter_graph import *
from .tools import *

__all__ = ["TopologicalFilterImage","TopologicalFilterImage"]

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python < 3.8, you can use the importlib-metadata package
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("topapprox")
except PackageNotFoundError:
    __version__ = "unknown"
    
