import torch
try:
    from icecream import install
    install()
except Exception:
    pass

from . import env
from .data import *
from .deep import *
from .env import *
from .metrics import *
from .util import *
