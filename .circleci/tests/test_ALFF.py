"""Test for ALFF."""

# Necessary imports

import os
import tempfile

import numpy as np
from scipy.fftpack import fft

from xcp_d.interfaces.prepostcleaning import Interpolate
from xcp_d.utils.write_save import read_ndata, write_ndata
