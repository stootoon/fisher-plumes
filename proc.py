"""
This script is used to run the FisherPlumes model for a variety of different parameter sets.
It takes as input a YAML file specifying the parameters to use for each run.
It will then run the model for each set of parameters, and save the results to a file.
It will also save a registry of all the runs that have been performed, so that it can be
determined which runs have already been performed, and which have not.
"""

import os,sys,pickle
import numpy as np
import yaml
from importlib import reload
from builtins import sum as bsum
import argparse
from argparse import ArgumentParser
from utils import eval_fields, deepcopy_data_fields, create_logger
import itertools
import random
import hashlib
import shutil
import glob, re

import corr_models
corr_models.logger.setLevel("INFO")

import units; UNITS = units.UNITS
HZ = UNITS.Hz
SEC= UNITS.s
M  = UNITS.m
UM = UNITS.um

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


def load_data(init_filter, compute_filter, data_dir = "./proc", registry = None, return_matches = False, fit_corrs = None):
    """ Load data from the file in the registry that matches the given init and compute filters. """
    if registry is None: registry = get_registry(data_dir)

    matches = find_registry_matches(registry, init_filter = init_filter, compute_filter = compute_filter)

    if len(matches) == 0:
        WARN(f"No matches found for {init_filter=} and {compute_filter=}.")
        return None
    elif len(matches) > 1:
        WARN(f"{len(matches)} > 1 matches found for {init_filter=} an {compute_filter=}.")

    results = []
    for match in matches:
        data_file = match["file"]
        INFO(f"Loading {init_filter=} {compute_filter=} from {data_file}")
        sys.stdout.flush()
        resi = pickle.load(open(data_file, "rb"))["results"]
        if fit_corrs is not None:
            for fc in fit_corrs:
                # Look for a folder in the same directory as the data file called "fit_corrs/{fit_corrs}/basename.p"
                # If it exists, load it and add it to the results.
                fit_file = os.path.join(os.path.dirname(data_file), f"fit_corrs/{fc}/{os.path.basename(data_file)}")
                if os.path.exists(fit_file):
                    INFO(f"Loading {fit_file}.")
                    sys.stdout.flush()
                    if "fit_corrs" not in resi:
                        resi["fit_corrs"] = {}
                    resi["fit_corrs"][fc] = pickle.load(open(fit_file, "rb"))
                else:
                    INFO(f"{fit_file} does not exist.")
                    sys.stdout.flush()
                
        results.append(resi)

    INFO(f"Returning {len(results)} results.")
    return results if not return_matches else (results, matches)

def load_spec(spec_file, verbose = False):
    """ Load the YAML file containing the specification for the runs to perform. """
    with open(spec_file, "r") as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    # Set the evaluation context to be the current globals.
    context = globals()
    spec    = eval_fields(spec, context=context)
    verbose and print(spec)

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
    
    return spec, compute_list

def fit_corrs(corrs, search_spec, output_file):
    pass

def check_file_exists(file_path):
    """ Check if a file exists. """
    if os.path.exists(file_path):
        return file_path
    raise argparse.ArgumentTypeError(f"{file_path} does not exist.")

def check_positive(n):
    """ Check if a number is positive. """
    n = int(n)
    if n > 0:
        return n
    raise argparse.ArgumentTypeError(f"{n} is not positive.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Run the FisherPlumes model for a variety of different parameter sets. Or, rebuild the registry of runs.")
    parser.add_argument("--run_spec", help="YAML file specifying the runs to perform.", type=str)    
    parser.add_argument("--mock", action = "store_true", help="Don't actually create the objects or run the computation.")
    parser.add_argument("--verbose", action = "store_true", help="Print more information about what is happening. ")
    parser.add_argument("--registry",  help="Python dictionary containing the registry of all generated data.", type=str, default="proc/registry.p")
    parser.add_argument("--overwrite", help="Overwrite existing runs, if they exist.", action="store_true")
    parser.add_argument("--rebuild",   help="Whether to rebuild the registry from scratch. The directory of the --registry flag will be used.", action="store_true")
    parser.add_argument("--set", help="Field (e.g. 'compute.window_length') and value (e.g. '[1*SEC,2*SEC]') to set in spec file.", nargs=2, action="append", default=None)

    parser.add_argument("--fp_data",      help="Pickle file or folder containing processed FisherPlumes data containing the correlations.", type=check_file_exists)
    parser.add_argument("--search_spec", help="YAML file specifying the gridsearch to perform.", type=check_file_exists)
    parser.add_argument("--gen_jobs",    help="Number of jobs to split the FREQS x DISTS data of each file into .", type=check_positive, default=1)
    parser.add_argument("--fit_corrs",    help="Spec of a fit to correlations to perform.", type=check_file_exists)
    parser.add_argument("--collect_fits", help="Directory containing correlation fits to combine.", type=check_file_exists)
    args = parser.parse_args()

    if any([args.fit_corrs, args.fp_data, args.collect_fits]):
        # Fitting correlation data
        if args.fp_data:
            assert args.search_spec, "Must specify a search spec file with --search_spec."
            # Get the search spec filename without the extension
            search_spec_file = os.path.splitext(args.search_spec)[0]
                        
            if os.path.isdir(args.fp_data):
                # If the fp_data is a directory, then we need to find all the files in it.
                fp_files = [os.path.join(args.fp_data, f) for f in os.listdir(args.fp_data) if f.endswith(".p")]
            else:
                # Otherwise, it's a single file.
                fp_files = [args.fp_data]

            # Now, for each file, we need to split it into jobs.
            for fp_file in fp_files:
                INFO(f"Splitting {fp_file} into {args.gen_jobs} jobs.")
                # Load the file and figure out how many probes, frequencies, and distances there are.
                fp_data = pickle.load(open(fp_file, "rb"))
                assert "results" in fp_data, f"Correlation data {fp_file} does not contain 'results' key."
                assert "rho" in fp_data["results"], f"Correlation data {fp_file} does not contain 'rho' key."
                rho = fp_data["results"]["rho"]
                n_probes  = len(rho)
                dists     = [d for d in list(rho[0].keys()) if d>=0]
                n_dists   = len(dists)
                n_freqs   = rho[0][dists[0]].shape[1]
                INFO(f"Found {n_probes} probes, {n_dists} distances, and {n_freqs} frequencies.")                
                # Compute all possible combinations of probes, distances, and frequencies.
                combos = list(itertools.product(range(n_probes), dists, range(n_freqs)))
                INFO(f"There {len(combos)} combinations of probes, distances, and frequencies.")
                # Randomly shuffle the combinations.
                random.shuffle(combos)
                # Split the combinations into jobs.
                jobs = np.array_split(combos, args.gen_jobs)
                INFO(f"Split into {len(jobs)} jobs.")
                # Now, create a directory for the jobs.
                # The directory will be that of the file, with /fit_corrs/search_spec_file/ appended.
                job_dir = os.path.join(os.path.dirname(fp_file), "fit_corrs", search_spec_file)
                # Remove the directory if it already exists.
                if os.path.exists(job_dir):
                    INFO(f"Removing any job files in {job_dir}.")
                    # Job files have the same name as fp_file, but with .p removed and the job number and .yaml appended.
                    cmd = f"rm -f {os.path.join(job_dir, f'{os.path.splitext(os.path.basename(fp_file))[0]}.*.yaml')}"
                    INFO(cmd)
                    os.system(cmd)
                    
                os.makedirs(job_dir, exist_ok=True)
                INFO(f"Created directory {job_dir}.")
                # Now, for each job, create a spec file.
                for i, job in enumerate(jobs):
                    # Create a spec file for this job.
                    # The name of the spec file will be fp_file with .p removed and the job number appended.
                    spec_file = os.path.join(job_dir, f"{os.path.splitext(os.path.basename(fp_file))[0]}.{i}.yaml")
                    # The spec file will contain the fp_file, the search_spec, and the job.
                    spec = {"fp_file": fp_file, "search_spec": args.search_spec, "probe_dist_ifreq": job.tolist()}
                    # Write the spec file to a human-readable yaml file.
                    yaml.dump(spec, open(spec_file, "w"), default_flow_style=True)
                    INFO(f"Wrote spec file {spec_file}.")
        elif args.fit_corrs:
            # If we're fitting correlations, then we need to load the spec file.
            spec = yaml.load(open(args.fit_corrs, "r"), Loader=yaml.FullLoader)
            fp_file          = spec["fp_file"]
            search_spec      = spec["search_spec"]
            probe_dist_ifreq = spec["probe_dist_ifreq"]

            # Load the correlations
            fp_data   =  pickle.load(open(fp_file, "rb"))["results"] 
            rho       = fp_data["rho"]
            freqs     = fp_data["freqs"]            

            dists = list(rho[0].keys())

            results = {}
            for i, (probe_id, dist, ifreq) in enumerate(probe_dist_ifreq):
                # Get the correlation data for this probe, distance, and frequency.
                rhoi = rho[probe_id][dist][0, ifreq] # 0 is for the full data, not the bootstrapped data.
                # Fit the correlation data.
                INFO(f"Fitting correlation data for probe {probe_id}, distance {dist:g}, frequency {freqs[ifreq]:g} using {search_spec}.")
                resultsi = corr_models.fit_corrs(rhoi, search_spec)
                # Report the best model params and score from the results["search"] grid search CV object.
                INFO(f"Best model params: {resultsi['search'].best_params_}")
                INFO(f"Best model score:  {resultsi['search'].best_score_}")
                INFO(f"Done fitting correlation data for probe {probe_id}, distance {dist:g}, frequency {freqs[ifreq]:g} to {search_spec}.")
                # Append the results to the results list.
                results[(probe_id, dist, ifreq)] = resultsi

            # The output file will be the same as the input file, but with yaml replaced with p.
            output_file = os.path.splitext(args.fit_corrs)[0] + ".p"
            # Write the results to a pickle file.            
            pickle.dump({"results": results,
                         "search_spec": search_spec,
                         "probe_dist_ifreq": probe_dist_ifreq,
                         "fp_file": fp_file},
                        open(output_file, "wb"))
            INFO(f"Wrote results to {output_file}.")
        elif args.collect_fits:
            # The files in the directory have names XYZ.1.p, XYZ.2.p, etc.
            # We want to collect all of the results into a single file XYZ.p.
            # First, find all the pickle files in the directory that are named XYZ.*.p.
            to_combine = {}
            for file_name in glob.glob(os.path.join(args.collect_fits, "*.p")):
                # Check if the file name matches the pattern XYZ.[number].p.
                match = re.match(r"(.*)\.(\d+)\.p", os.path.basename(file_name))
                if match:
                    # Split the file name into the base name and the job number.
                    base_name, job_num = match.groups()
                    # Add the file name to the list of files to combine.
                    if base_name not in to_combine:
                        to_combine[base_name] = [file_name]
                    else:
                        to_combine[base_name].append(file_name)

            # Now, for each base name, combine the data in the files
            # by merging the results dictionaries and the probe_dist_ifreq lists.
            for base_name, file_names in to_combine.items():
                # Load the first file.
                results = pickle.load(open(file_names[0], "rb"))
                # For each subsequent file, load the results and append the probe_dist_ifreq list.
                for file_name in file_names[1:]:
                    resultsi = pickle.load(open(file_name, "rb"))
                    # Check that there are results for each probe_dist_ifreq in the first file.
                    expected_keys = sorted([tuple(t) for t in resultsi["probe_dist_ifreq"]])
                    actual_keys   = sorted(resultsi["results"].keys())
                    assert expected_keys == actual_keys, f"expected_keys {expected_keys} != actual_keys {actual_keys} in {file_name}"                    
                    # Check that the fp_file and search_spec are the same in this file as in the first file.
                    assert results["fp_file"]     == resultsi["fp_file"],     f"fp_file {results['fp_file']} != {resultsi['fp_file']} for {file_name}"
                    assert results["search_spec"] == resultsi["search_spec"], f"search_spec {results['search_spec']} != {resultsi['search_spec']} for {file_name}"
                    results["probe_dist_ifreq"].extend(resultsi["probe_dist_ifreq"])
                    results["results"].update(resultsi["results"])
                # The output file will be the same as the input file, but with yaml replaced with p.
                output_file = os.path.join(args.collect_fits, os.path.splitext(base_name)[0] + ".p")
                # Write the results to a pickle file.            
                pickle.dump(results, open(output_file, "wb"))
                INFO(f"Collect fits results to {output_file}.")
            
            
            
            
            
                

                

        exit(0)
    
    if args.set is not None:
        if args.run_spec is None:
            print("No run specification file given. Exiting.")
            exit(1)
        # If the spec file doesn't exist, exit.
        if not os.path.exists(args.run_spec):
            print(f"Spec file {args.run_spec} does not exist. Exiting.")
            exit(1)

        top_field, sub_field  = args.set[0][0].split(".")
        value = args.set[0][1]
        # Load the yaml file, and set the field
        spec = yaml.load(open(args.run_spec, "r"), Loader=yaml.FullLoader)
        spec[top_field][sub_field] = value
        # Write the yaml file back out.
        with open(args.run_spec, "w") as f:
            yaml.dump(spec, f)
        print(f"Set {top_field}.{sub_field} to {value} in {args.run_spec}.")
        exit(0)
    
    # If the registry is to be rebuilt, get the data from the registry directory.
    if args.rebuild:
        registry = get_registry(os.path.dirname(args.registry), build=True, write=True)
        exit(0)

    if "run_spec" not in args or args.run_spec is None:
        print("No run specification file given. Exiting.")
        exit(1)

    verbose   = args.verbose    
    if not os.path.exists(args.registry):
        print(f"Registry file {args.registry} does not exist. Creating registry.")
        registry = []
    else:
        print(f"Registry file {args.registry} exists. Loading registry.")
        registry = pickle.load(open(args.registry, "rb"))
        print(f"Registry loaded. Found {len(registry)} items.")
           
    # Load the run specification file.
    spec, compute_list  = load_spec(args.run_spec, verbose = verbose)

    if not args.mock:
        from fisher_plumes import FisherPlumes            
        print("Creating FisherPlumes object.")
        fp = FisherPlumes(**spec["init"])
        print(fp)
    
    # Create a directory for the output file.
    # The directory name will be the same as the spec file, but with the extension removed.
    output_dir = os.path.splitext(args.run_spec)[0]
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
