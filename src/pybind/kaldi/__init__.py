import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.dirname(__file__))

from kaldi_pybind import *

from pytorch_util import *
from symbol_table import *
from table import *
