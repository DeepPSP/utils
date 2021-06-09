# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import os, sys

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_BASE_DIR)
_IN_SYS_PATH = [p for p in [_BASE_DIR, _PARENT_DIR] if p in sys.path]
if len(_IN_SYS_PATH) == 0:
    sys.path.append(_PARENT_DIR)

# from .common import *
# from .utils_universal import *
# from .utils_image import *
# from .utils_signal import *
# from .utils_misc import *
# from .utils_audio import *


# __all__ = [s for s in dir() if not s.startswith("_")]
