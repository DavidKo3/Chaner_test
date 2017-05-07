import atexit
import threading

import numpy
import six

import cupy
from cupy.core import internal
from cupy import cuda
from cupy.cuda import cudnn


_cudnn_version = cudnn.getVersion()
_thread_local = threading.local()

print (_cudnn_version)
