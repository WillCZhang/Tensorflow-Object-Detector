import os
import sys


if sys.version_info[0] < 3:
    raise Exception("Python 3 is required")


def loadEnv(var):
    if var in os.environ:
        return os.environ[var]
    raise Exception("Missing Required Environment Variable - " + var)


