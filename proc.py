#!/usr/bin/env python
"""
This script is used to run the FisherPlumes model for a variety of different parameter sets.
It takes as input a YAML file specifying the parameters to use for each run.
It will then run the model for each set of parameters, and save the results to a file.
It will also save a registry of all the runs that have been performed, so that it can be
determined which runs have already been performed, and which have not.
"""

import os,sys,pickle,datetime
import yaml
from importlib import reload
from builtins import sum as bsum
from argparse import ArgumentParser
from utils import eval_fields, deepcopy_data_fields, create_logger
import itertools
import hashlib


import units

logger = create_logger(__name__)
INFO = logger.info
DEBUG = logger.debug
WARN = logger.warning

# This function will read all the subdirectories of the given directory.
# For each subdirectory, it will read all the pickle files in that directory.
# It will then add the init and compute fields from each pickle file to a list,
# along with the file name.
def get_registry(registry_loc, build=False, write=False):
    if build:
        INFO("Building registry.")
        # Create a new registry.
        registry = []
        for dirpath, dirnames, filenames in os.walk(registry_loc):
            for filename in filenames:
                if filename.endswith(".p"):
                    file_path = os.path.join(dirpath, filename)
                    data = pickle.load(open(file_path, "rb"))
                    # if it's not a dictionary, skip it.
                    if not isinstance(data, dict):
                        continue
                    elif "init" not in data or "compute" not in data:
                        continue                
                    INFO(f"Read {file_path}.")
                    registry.append({"init":data["init"], "compute":data["compute"], "file":file_path})
        if write:
            pickle.dump(registry, open(os.path.join(registry_loc, "registry.p"), "wb"))
            INFO(f"Registry written to {os.path.join(registry_loc, 'registry.p')}.")
    else:
        if os.path.exists(os.path.join(registry_loc, "registry.p")):
            registry = pickle.load(open(os.path.join(registry_loc, "registry.p"), "rb"))
    return registry

# This function will take a registry and key-value pairs to look for in the init and compute fields.
# It will return a list of all the registry items that match the given key-value pairs.
def find_registry_matches(registry = None, init_filter = {}, compute_filter = {}):
    if registry is None:
        INFO("No registry given. Loading registry from proc/registry.p.")
        registry = pickle.load(open("proc/registry.p", "rb"))
        
    matches = []
    for item in registry:
        init_match = True
        compute_match = True
        for k,v in init_filter.items():
            if k not in item["init"] or item["init"][k] != v:
                init_match = False
                break
        for k,v in compute_filter.items():
            if k not in item["compute"] or item["compute"][k] != v:
                compute_match = False
                break
        if init_match and compute_match:
            matches.append(item)
    return matches


def load_data(init_filter, compute_filter, registry = None, return_matches = False):
    """ Load data from the file in the registry that matches the given init and compute filters. """
    if registry is None: registry = get_registry("./proc")

    matches = find_registry_matches(registry, init_filter = init_filter, compute_filter = compute_filter)

    if len(matches) == 0:
        WARN(f"No matches found for {init_filter=} and {compute_filter=}.")
        return None
    elif len(matches) > 1:
        WARN(f"{len(matches)} > 1 matches found for {init_filter=} an {compute_filter=}.")

    results = []
    for match in matches:
        data_file = match["file"]
        
        INFO(f"Loading {init_filter=} {compute_filter=} from {data_file}, last modified {get_last_mod_timestr(data_file)}.")
        sys.stdout.flush()
        results.append(pickle.load(open(match["file"], "rb"))["results"])

    INFO(f"Returning {len(results)} results.")
    return results if not return_matches else (results, matches)
        
def get_last_mod_timestr(file_path):
    """ Get the last modification time of the given file as a string. """
    return datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")

def hash_init_compute(init, compute, length=16):
    """Stringify the init and compute dictionaries, concatenate them, and hash that using shake_128."""
    init_str = str(init)
    compute_str = str(compute)
    combined = init_str + compute_str        
    hashed =  hashlib.shake_128(combined.encode("utf-8")).hexdigest(length//2)
    print(f"Hashed {combined} to {hashed}.")
    return hashed
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Run the FisherPlumes model for a variety of different parameter sets. Or, rebuild the registry of runs.")
    parser.add_argument("--run_spec", help="YAML file specifying the runs to perform.", type=str)    
    parser.add_argument("--mock", action = "store_true", help="Don't actually create the objects or run the computation.")
    parser.add_argument("--verbose", action = "store_true", help="Print more information about what is happening. ")
    parser.add_argument("--registry", help="Python dictionary containing the registry of all generated data.", type=str, default="proc/registry.p")
    parser.add_argument("--overwrite", help="Overwrite existing runs, if they exist.", action="store_true")
    parser.add_argument("--rebuild", help="Whether to rebuild the registry from scratch. The directory of the --registry flag will be used.", action="store_true")
    args = parser.parse_args()

    # If the registry is to be rebuilt, get the data from the registry directory.
    if args.rebuild:
        registry = get_registry(os.path.dirname(args.registry), build=True, write=True)
        exit(0)

    if "run_spec" not in args or args.run_spec is None:
        print("No run specification file given. Exiting.")
        exit(1)

    from fisher_plumes import FisherPlumes
    import units; UNITS = units.UNITS
    HZ = UNITS.Hz
    SEC= UNITS.s
    M  = UNITS.m
    UM = UNITS.um
                
    verbose   = args.verbose    
    if not os.path.exists(args.registry):
        print(f"Registry file {args.registry} does not exist. Creating registry.")
        registry = []
    else:
        print(f"Registry file {args.registry} exists.\nLast modified at {get_last_mod_timestr(args.registry)}.\nLoading registry.")
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
    values = [compute[k] if hasattr(compute[k], "__len__") and not k.endswith("__") else [compute[k]] for k in keys]
    # Now, generate a list of dictionaries, each with the same keys as compute, but with the values
    # set by taking all combinations of the corresponding values in compute.
    keys = [k if not k.endswith("__") else k[:-2] for k in keys]
    compute_list = [{k:v for k,v in zip(keys, vals)} for vals in itertools.product(*values)]
    
    # Create a directory for the output file.
    # The directory name will be the same as the spec file, but with the extension removed.
    output_dir = os.path.splitext(spec_file)[0]
    print(output_dir)
    if not os.path.exists(output_dir):
        print(f"Creating output directory {output_dir}.")
        os.mkdir(output_dir)
    
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
            results = deepcopy_data_fields(fp,dont_copy=["sim0", "sims"])
            if hasattr(fp, "sim0"):
                results["sim0"] = deepcopy_data_fields(fp.sim0)
            if hasattr(fp, "sims"):
                results["sims"] = {}
                for k,v in fp.sims.items():
                    results["sims"][k] = deepcopy_data_fields(v)                    
                
            pickle.dump({"init":spec["init"], "compute":compute_item, "results":results}, open(output_file, "wb"))
            print(f"Results written to {output_file}.")
    
        registry.append({"init":spec["init"], "compute":compute_item, "hash":item_hash, "file":output_file})
    
    
    # Write the registry to disk.
    pickle.dump(registry, open(args.registry, "wb"))
    print(f"Registry with {len(registry)} items written to {args.registry}.")
