import os, re
from functools import reduce
import logging
from copy import deepcopy

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

def progn(*args):
    for a in args:
        a()

chain = lambda af: lambda x: reduce(lambda a, f: f(a), af[1:],af[0](x))         

fapply = lambda farr, a: [f(a) for f in farr] # applies each function in farr to a

d1 = lambda d: d[list(d.keys())[0]] # Return the first element of a dictionary

logger = create_logger("utils")
logger.setLevel(logging.INFO)
DEBUG = logger.debug

def eval_fields(d, context = None):
    # Generate code that will accept a dictionary
    # and evalutes all fields that are not dictionaries or dictionaries.
    # If the field evaluates without error, then the result is kept,
    # otherwise the original string is kept.
    
    for k,v in d.items():
        # Check that the value is not a dictionary.
        if isinstance(v, dict):
            # If it is a dictionary, then recurse.
            DEBUG(f"Value for {k} is a dictionary, so recursing.")
            v = eval_fields(v, context=context)
        else:
            # If the value is not a string, keep it as is.
            if not isinstance(v, str):
                DEBUG(f"Key={k:>12s}: Value {v} is not a string, so keeping it as is.")
            else:
                # Otherwise, try to evalute the string.
                try:
                    v1 = eval(v, context)
                    DEBUG(f"Key={k:>12s}: Evaluating the value {v} succeeded, so keeping the result {v1}")
                    v = v1
                except Exception as E:
                    # print the exception message
                    DEBUG(f"Key={k:>12s}: Evaluating the value {v} raised exception {E}, so keeping the original string.")
                    # If it fails, keep the original string.
                    pass
            d[k] = v
    return d

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

def get_args(req_args, kwargs):
    vals = []
    for req in req_args:
        if req not in kwargs:
            raise ValueError(f"{req} not found in kwargs.")
        vals.append(kwargs[req])
        del kwargs[req]
    return *vals, kwargs


def deepcopy_data_fields(obj, dont_copy = []):
    """
    Return a dictionary of all the data fields in the object.
    A field is considered a data field if it is not callable.
    """
    data = {}
    for fld,val in obj.__dict__.items():
        if not callable(val):
            if fld in dont_copy: continue
            data[fld] = deepcopy(val)
    return data
    
