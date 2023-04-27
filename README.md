# fisher-plumes
Code for the information in Fisher information in plumes manuscript.
## Requirements
The code was written in Python 3.8 and jupyter-notebook 6.0.3. The versions contained in the latest [Anaconda](anaconda.com) distribution should be sufficient.

## Installation
After downloading this repository, installation merely requires downloading the CFD simulation data, if necessary, and updating paths to point to this data.

1. Download the CFD simulation data, if needed.
- Data for the simulations in the Main Text (171 GB): [Re100_0_5mm_50Hz_16source_manuscript.h5](https://www.dropbox.com/s/p3bbuq4r84nrrr2/Re100_0_5mm_50Hz_16source_manuscript.h5?dl=0)
- Data for the simulations in the Supplementary Material (19 G): [crick-data.tar.gz](https://www.dropbox.com/s/4t2h3dg11oq14vg/crick-data.tar.gz?dl=0)
2. Updating paths to the data.
- For the data in the Main Text: Update the `root` field in `boulder.json` with the path to the folder containing `Re100...orig.h5`.
- For the data in the Supplementary Material: Updat the `root` field in `crick.json` with the path to the unzipped folder that contains sub folders `ff_int_...`

You should now be ready to create the figures for the paper by running  the `make_figs.ipynb` notebook.

## Code Description
The analysis in the paper is applied to data from two sets of simulations. 
### Steps in computing Fisher information
Fisher information is computed in the following steps. These are called in sequence by `compute_all_for_window`.
1. The desired window size for the computation is set using `set_window`.
2. The trigonometric coefficients are computed using `compute_trig_coefs`.
   - The sine and cosine coefficients and the time window they were computed over, are stored in the `F.ss` ,`F.cc`, `F.tt` fields, respectively. 
     - Trigonometric coefficients are determined by computing a short-time Fourier Transform and converting the complex-valued result to the corresponding sine and cosine coefficients.
	 - The detrender applied by the STFT to each window consists of Tukey windowing followed by z-scoring.
     - Z-scoring normalizes the signal by its amplitude and ensures that the variability we measure in coefficients is not dominated by fluctuations in signal amplitude.
	 - The results are indexed as e.g. `F.ss[PROBE][SRC]`, where the integer `PROBE` indexes the probe for whic
   
