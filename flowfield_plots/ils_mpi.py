# Source code to run on CU Boulder HPC resources: Blanca cluster
# Computes integral length scales for multisource plume dataset for u and v in both x-stream and streamwise directions
# v2 uses distributed data loading to load chunks of u_data into different processes
# Elle Stark May 2024

import h5py
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import time

# Set up logging for convenient messages
logger = logging.getLogger('ilsmpipy')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warn
DEBUG = logger.debug

def load_data_chunk(filename, dataset_name, start_idx, end_idx, adjust, streamwise):
    with h5py.File(filename, 'r') as f:
        if streamwise:
            data_chunk = f[dataset_name][:, start_idx+adjust:end_idx+adjust, :].astype(np.float64)  # original: [time, y, x] array with dims [3001, 846, 1001]
            data_chunk = data_chunk.transpose(1, 2, 0)
        else:
            data_chunk = f[dataset_name][:, :, start_idx+adjust:end_idx+adjust].astype(np.float64)
            data_chunk = data_chunk.transpose(2, 1, 0)

        DEBUG(f'Memory size of each chunk: {data_chunk.itemsize * data_chunk.size}')
    return data_chunk

def main():

    # MPI setup and related data
    comm_world = MPI.COMM_WORLD
    rank = comm_world.Get_rank()
    # number of processes will be determined from Slurm job script (.sh file)
    # script takes ~5 min to run with ntasks ~210 for streamwise, ~250 for cross-stream (using pre-emptable Blanca nodes) 
    num_procs = comm_world.Get_size()
    INFO(f'RUNNING ON {num_procs} PROCESSES.')

    # Information for loading data from PetaLibrary
    # filename = '/pl/active/odor2action/Data/odorPlumeData/multisource-plume-simulations/hdf5/Re100_0_5mm_50Hz_16source_FTLE_manuscript.h5'
    # dataset_name = 'Flow Data/u'
    filename = '/pl/active/odor2action/Stark_data/ms_plume_supplementarydata/fluctfields.h5'
    dataset_name = 'u_flx' # Choose u_flx or v_flx for fluctuating component of streamwise (u) or cross-stream(v) velocity

    # Select integral length scale correlation direction: streamwise = True, or streamwise = False for cross-stream
    streamwise = False

    # Define time, x, and y sizes - hardcoded here for convenience
    t, x, y = 3001, 1001, 846

    # Compute chunk sizes for each process
    # for streamwise ILS, use y, for cross-stream, use x
    if streamwise:
        primary_ax = y
    else:
        primary_ax = x
    chunk_size = primary_ax // num_procs
    DEBUG(f'Chunk size: {chunk_size}')
    remainder = primary_ax % num_procs

    # Find start and end index for each process
    # adjust = 423 - y // 2  # Adjust indices if needed for subsetting in the y-direction
    adjust = 0
    start_idx = rank * chunk_size + min(rank, remainder) 
    end_idx = start_idx + chunk_size + (1 if rank < remainder else 0) 
    DEBUG(f'start idx: {start_idx + adjust}; end idx: {end_idx + adjust}')

    # Load chunk of data to each process
    local_u_chunk = load_data_chunk(filename, dataset_name, start_idx, end_idx, adjust, streamwise=streamwise)
    DEBUG(f'Process {rank} loaded data chunk with shape {local_u_chunk.shape}')

    # Simple QC test for mpi: sum along time axis in each process
    # local_result = np.sum(local_u_chunk, axis=0)

    # LOCAL RESULTS: CALCULATE INTEGRAL LENGTH SCALE
    # Follows approach used in Tootoonian et. al. 2023, see https://github.com/stootoon/fisher-plumes/blob/main/boulder.py

    dims = local_u_chunk.shape
    # For streamwise ILS, idx_1=0 and idx_2=1; reverse for cross-stream
    idx_1 = 0
    idx_2 = 1 

    # Initialize array of autocorrelation functions f(r)
    ils_array = np.zeros((dims[idx_1], dims[idx_2]))
    DEBUG(f"i range: {dims[idx_1]}. j range: {dims[idx_2]}. r range:{int(dims[idx_2])}.")

    # Loop through each location
    for i in range(dims[idx_1]):
        for j in range(dims[idx_2]):
            # lists of autocorrelation values at each r
            q_minus = np.empty(int(dims[idx_2]))
            q_plus = np.empty(int(dims[idx_2]))

            for r in range(int(dims[idx_2])):
                if (j-r) >= 0:
                    q_minus[r] = np.mean(local_u_chunk[i, j, :]*local_u_chunk[i, j-r, :])
                else:
                    q_minus[r] = np.nan
                
                if (j+r) < dims[idx_2]:
                    q_plus[r] = np.mean(local_u_chunk[i, j, :]*local_u_chunk[i, j+r, :])
                else:
                    q_plus[r] = np.nan

            # Combine plus and minus into single Q array and average to get autocorrelation function f(r)
            q_list = np.array([q_minus, q_plus])
            fr = np.nanmean(q_list, axis=0)
            fr /= fr[0]

            # QC Plot: autocorrelation line plot
            # if ((rank==14) & (j==900) & (i==0)):
            #     plt.close()
            #     plt.plot(fr)
            #     DEBUG(f'fr at 423, {j}: {fr}')
            #     plt.savefig(f'/rc_scratch/elst4602/tests/frline_{j}_423.png')

            # Integrate f(r) until first zero crossing 
            dx = 0.0005
            # fr_to_zero = fr[0:np.argmax(fr<0)]
            # if len(fr_to_zero)==0:
            #     ils = np.nansum(fr * dx)
            # else:
            #     ils = np.nansum(fr_to_zero * dx)
            
            ils = np.nansum(fr * dx)
            ils_array[i, j] = ils

            # # Add q_plus and q_minus to get components of autocorrelation function
            # qsum = q_plus + q_minus
            # DEBUG(f'qsum dimensions: {qsum.shape}')
            # qsum_array[:, i, j] = qsum


    # local_result = qsum_array.flatten()
    local_result = ils_array.flatten()
    # local_u_mean = np.mean(local_u_chunk, axis=2)
    # local_u_mean = local_u_chunk[:, :, 0]
    # local_u_mean = local_u_mean.flatten()
    DEBUG(f"Process {rank} completed with result size {local_result.size}")


    # GATHER ALL RESULTS INTO PROCESS 0

    if streamwise:
        recvcounts = np.array([chunk_size * x] * num_procs, dtype=int)
        for i in range(remainder):
            recvcounts[i] += x
    else:
        recvcounts = np.array([chunk_size * y] * num_procs, dtype=int)
        for i in range(remainder):
            recvcounts[i] += y

    recvdisplacements = np.array([sum(recvcounts[:i]) for i in range(num_procs)], dtype=int)

    # Prepare buffer in root process for gathering
    if rank == 0:
        if streamwise:
            gathered_u = np.empty((y, x), dtype=np.float64)
            # data_test = np.empty((y, x), dtype=np.float64)
        else:
            gathered_u = np.empty((x, y), dtype=np.float64)
    else:
        gathered_u = None
        # data_test = None

    comm_world.Gatherv(local_result, [gathered_u, recvcounts, recvdisplacements, MPI.DOUBLE], root=0)
    # comm_world.Gatherv(local_u_mean, [data_test, recvcounts, recvdisplacements, MPI.DOUBLE], root=0)

    # Reshape gathered data
    if rank == 0: 
        if streamwise:
            gathered_u = gathered_u.reshape((y, x))
        else:
            gathered_u = gathered_u.reshape((x, y)).T
        DEBUG(f'gathered data shape: {gathered_u.shape}')
        DEBUG(f'y = {y}')
        # data_test = data_test.reshape((y, x))

        gathered_u = np.flip(gathered_u, axis=0)
        INFO("ils array data shape: " + str(gathered_u.shape[0]) + " x " + str(gathered_u.shape[1]))
        DEBUG("spatial average across domain: " + str(np.mean(gathered_u)))

        if streamwise:
            direction = 'streamwise'
        else:
            direction = 'cross_stream'
        np.save(f'/rc_scratch/elst4602/FisherPlumePlots/ILS_{dataset_name}_{direction}.npy', gathered_u)

        # QC Plots
        fig, ax = plt.subplots(figsize=(5.9, 4))
        plt.pcolormesh(gathered_u, cmap='magma_r', vmin=0, vmax=0.20)
        plt.colorbar()
        plt.savefig(f'/rc_scratch/elst4602/FisherPlumePlots/ILS_{dataset_name}_{direction}.png', dpi=600)

        # plt.close()
        # plt.pcolormesh(data_test)
        # plt.colorbar()
        # plt.savefig('/rc_scratch/elst4602/FisherPlumePlots/uprime0.png', dpi=600)

if __name__=="__main__":
    main()


