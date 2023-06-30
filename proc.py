import os,sys,pickle
from importlib import reload
from collections import defaultdict
from copy import deepcopy
from builtins import sum as bsum
import yaml
import pdb
from argparse import ArgumentParser
from utils import eval_fields
from fisher_plumes import FisherPlumes
import itertools
import hashlib

import units; UNITS = units.UNITS
HZ = UNITS.Hz
SEC= UNITS.s
M  = UNITS.m
UM = UNITS.um

parser = ArgumentParser()
parser.add_argument("--mock", action = "store_true", help="Don't actually create the objects or run the computation.")
parser.add_argument("--verbose", action = "store_true", help="Print more information about what is happening. ")
parser.add_argument("--registry", help="Python dictionary containing the registry of all generated data.", type=str, default="proc/registry.p")
parser.add_argument("--overwrite", help="Overwrite existing runs, if they exist.", action="store_true")
parser.add_argument("run_spec", help="YAML file specifying the runs to perform.", type=str)
args = parser.parse_args()

verbose   = args.verbose

if not os.path.exists(args.registry):
    print(f"Registry file {args.registry} does not exist. Creating registry.")
    registry = []
else:
    print(f"Registry file {args.registry} exists. Loading registry.")
    registry = pickle.load(open(args.registry, "rb"))
    print(f"Registry loaded. Found {len(registry)} items.")
    
    
# Load the run specification file.
spec_file = args.run_spec
with open(spec_file, "r") as stream:
    spec = yaml.safe_load(stream)

# Set the evaluation context to be the current globals.
context = globals()
spec    = eval_fields(spec, context=context)
verbose and print(spec)

if not args.mock:
    print("Creating FisherPlumes object.")
    fp = FisherPlumes(**spec["init"])
    print(fp)

compute = spec["compute"]
# Generate a list of dictionaries, each with the same fielda as compute, but with the values
# set by taking all combinations of the corresponding values in compute.

# First, generate a list of all the keys in compute.
keys = list(compute.keys())
verbose and print(f"Found the following keys: {keys}.")
# Now, for each key generate a list of all the values.
# If a given value is not a list type, make it into a singleton list.
values = [compute[k] if hasattr(compute[k], "__len__") else [compute[k]] for k in keys]
# Now, generate a list of dictionaries, each with the same keys as compute, but with the values
# set by taking all combinations of the corresponding values in compute.
compute_list = [{k:v for k,v in zip(keys, vals)} for vals in itertools.product(*values)]

# Create a directory for the output file.
# The directory name will be the same as the spec file, but with the extension removed.
output_dir = os.path.splitext(spec_file)[0]
print(output_dir)
if not os.path.exists(output_dir):
    print(f"Creating output directory {output_dir}.")
    os.mkdir(output_dir)

def hash_init_compute(init, compute, length=16):
    """Stringify the init and compute dictionaries, concatenate them, and hash that using shake_128."""
    init_str = str(init)
    compute_str = str(compute)
    combined = init_str + compute_str
    return hashlib.shake_128(combined.encode("utf-8")).hexdigest(length//2)        
    
for item_id, compute_item in enumerate(compute_list):
    print(f"Processing {compute_item=}.")
    # Check the registry to see if this item has already been computed.
    # It will check the registry by checking the init and compute fields.

    item_hash = hash_init_compute(spec["init"], compute_item)
    matches = [i for i, r in enumerate(registry) if hash_init_compute(r["init"], r["compute"]) == item_hash]
    if len(matches) and not args.overwrite:
        print(f"{len(matches)} found in registry. Skipping.")
        continue

    # Copy the registry without the matches.
    registry = [r for i,r in enumerate(registry) if i not in matches]
    output_file = os.path.join(output_dir, f"{item_hash}.p")
    print(f"Writing output to {output_file}.")
    if not args.mock:
        fp.compute_all_for_window(**compute_item)
        results = fp.copy_data_fields()
        pickle.dump({"init":spec["init"], "compute":compute_item, "results":results}, open(output_file, "wb"))
        print(f"Results written to {output_file}.")

    registry.append({"init":spec["init"], "compute":compute_item, "hash":item_hash, "file":output_file})


# Write the registry to disk.
pickle.dump(registry, open(args.registry, "wb"))
print(f"Registry with {len(registry)} items written to {args.registry}.")
    
    
    
    

    



