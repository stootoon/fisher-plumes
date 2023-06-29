import os, re
from functools import reduce
import logging

def progn(*args):
    for a in args:
        a()

chain = lambda af: lambda x: reduce(lambda a, f: f(a), af[1:],af[0](x))         

fapply = lambda farr, a: [f(a) for f in farr] # applies each function in farr to a

d1 = lambda d: d[list(d.keys())[0]] # Return the first element of a dictionary

# Destructure a dictionary
def dd(*args): 
    return lambda d: [d[a] for a in args]

dict_update = lambda d, flds, vals: [d.update({fld:val for fld,val in zip(flds, vals)}),d][1]

# Take the arguments in twos as flds/src pairs
dict_update_from_field = lambda d, *args: [[d.update({fld:d[src] for fld in flds}) for flds,src in zip(args[::2], args[1::2])],d][1]

def expand_environment_variables(s, cleanup_multiple_slashes = True):
    for v in re.findall("(\$[A-Za-z_]+)", s):
        s = s.replace(v, os.environ[v[1:]]) # v[1:] to skip the $ at the beginning, so $HOME -> HOME
    if cleanup_multiple_slashes:
        s = re.sub("/+", "/", s)
    return s

def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        for h in logger.handlers:
            logger.removeHandler(h)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt='%(module)24s %(asctime)s %(levelname)8s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    logger.addHandler(ch)
    return logger

def get_args(req_args, kwargs):
    vals = []
    for req in req_args:
        if req not in kwargs:
            raise ValueError(f"{req} not found in kwargs.")
        vals.append(kwargs[req])
        del kwargs[req]
    return *vals, kwargs


