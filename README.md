# fisher-plumes
Code for the information in Fisher information in plumes manuscript (still under construction).
## Requirements
The code was written in Python 3.8 and jupyter-notebook 6.0.3. The versions contained in the latest [Anaconda](anaconda.com) distribution should be sufficient.

## Installation
After downloading this repository, installation merely requires downloading the CFD simulation data, if necessary, and updating paths to point to this data.

1. Download the CFD simulation data, if needed.
- Data for the simulations in the Main Text (171 GB): [Re100_0_5mm_50Hz_16source_manuscript.h5](https://www.dropbox.com/s/p3bbuq4r84nrrr2/Re100_0_5mm_50Hz_16source_manuscript.h5?dl=0)
- Data for the simulations in the Supplementary Material (19 G): [crick-data.tar.gz](https://www.dropbox.com/s/4t2h3dg11oq14vg/crick-data.tar.gz?dl=0)
2. Updating paths to the data.
- For the data in the Main Text: Update the `root` field in [boulder.json](boulder.json) with the path to the folder containing `Re100...orig.h5`.
- For the data in the Supplementary Material: Updat the `root` field in [crick.json](crick.json) with the path to the unzipped folder that contains sub folders `ff_int_...`

You should now be ready to create the figures for the paper by running  the [make_figs.ipynb](make_figs.ipynb) notebook.

## Code Description
The analysis in the paper is applied to data from two sets of simulations run on two different simulation platforms.
The `FisherPlumes` class provides a uniform interface to both sets of data.
Here we will describe the steps involved in loading the simulations and computing the fisher information metrics.
These steps are carried out in [make_figs.ipynb](make_figs.ipynb) before the figures for the manuscript are made.
### Loading the data
- The data from a set of simulations are loaded by constructing an instance of the `FisherPlumes` class.
- The desired simulation data is indicated by the name passed to the constructor, `boulder16` for the data in the main text and `n12dishT` for the data in the Supplementary Information.
- Other notable parameters passed to `FisherPlumes` are:
  - `pitch`: The pitch units of the dataset;
  - `freq_max`: The maximum frequency to compute metrics for;
  - `which_coords`: A list of `(x,y)` tuples indicating probe locations;
  - `py_mode`: Set to `relative` if the `y` coordinates of `which_coords` are expressed as fractions of the heights of the simulated wind tunnel.
  - `pairs_mode`: Odour sources are located at a fixed x-location but at different y-locations in each simulated wind tunnel. We will be computing metrics comparing for pairs of sources located at different locations and pooling the results for sources located at the same intersource separation. The `pairs_mode` parameter determines which sources will be paired and how they will be pooled.
	- `all`: For a given pair located at `y1` and `y2`, add the data for `(y1, y2)` to the pool for separation of `y1 - y2`, and the data for `(y2, y1)` to the `y2-y1` pool.
	- `unsigned`: Add the data for `(y2,y1)` and `(y1,y2)` to the `|y1 - y2|` pool.
	- `sym`: As in `all` but only using `y1` and `y2` that are symmetric around the midline.
- Given these parameters, `FisherPlumes` will then all the `load_sims` method of the desired dataset to load the simulation results.

### Steps in computing Fisher information
Fisher information is computed in the following steps. These are called in sequence by `compute_all_for_window`.
1. The desired window size for the computation is set using `set_window`.
2. The trigonometric coefficients are computed using `compute_trig_coefs`.
   - The sine and cosine coefficients and the time window they were computed over, are stored in the `F.ss` ,`F.cc`, `F.tt` fields, respectively. 
     - Trigonometric coefficients are determined by computing a short-time Fourier Transform and converting the complex-valued result to the corresponding sine and cosine coefficients.
	 - The detrender applied by the STFT to each window consists of Tukey windowing followed by z-scoring.
     - Z-scoring normalizes the signal by its amplitude and ensures that the variability we measure in coefficients is not dominated by fluctuations in signal amplitude.
	 - The results are indexed as e.g. `F.ss[PROBE][SRC]`, where the integer `PROBE` indexes the probe for which...
   
## Processing multiple datasets with `proc.py`
When making the figures for the paper one possibility is to process the datasets afresh each time. This can become very time-consuming when we want to process 
multiple datasets and using a variety of windows and other processing parameters.
To avoid having to reprocess the data each time, we precompute the results of the processing and store them to file. 

To do so, we use `proc.py`. This script requires a run specification provided
in a YAML file. The YAML file contains both initialization and computation parameters.
The `proc.py` script uses the initialization parameters to create a FisherPlumes object, and then uses the computation parameters to compute Fisher information.
An example YAML file for processing the dataset in the Main Text is [bw.1.yaml](./proc/bw.1.yaml). 

To `init` field of the YAML file specifies key-value pairs that are passed directly to the FisherPlumes object constructor. The only restriction is that while the FisherPlumes object allows multiple probe coordinates to be specified, the initialization parameters in a YAML file must only use a single probe location.

The `compute` field of the YAML file can specify multiple computation parameters to run. This is mainly to allow windows of different shape and length to be used. The different parameters are provided as python strings that are then evaluated to determine the required values for each key. 

Fisher information is computed for each combination of values for each key. The results are stored, along with the initialization parameters of the FisherPlumes object and the computation parameters, in a pickle file in a directory with the same name as the YAML file but without the extension (e.g. `bw.1` for the example above).

Before the computation starts, a registry containing the list of all existing data is consulted to see whether data already exists for the specified initialization and computation parameters. If so, the computation is either skipped, or optionally overwritten by using the `--overwrite` flag.

The registry can be rebuilt from scratch by calling `proc.py` with the `--rebuild` flag. The registry can then be consulted by the plotting scripts to extract the data that they need.