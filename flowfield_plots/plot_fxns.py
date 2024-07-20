# Functions for computing relavent flow statistics for turbulent flow field
# Elle Stark, University of Colorado Boulder, June 2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cmasher as cmr

def reynolds_decomp(flowfield, time_ax=0):
    """decomposes time series of 2D flowfield data into its mean and fluctuating components

    Args:
        flowfield (nd-array): stack of flow data of interest.
        time_ax (int, optional): Specify time axis along which to calculate the mean. Defaults to 0.

    Returns:
        decomp_fields (list): [mean field, fluctuating field], where mean field is 2D array of flow averaged over time and 
            fluctuating field is 3D array of fluctuating component of flow at each timestep.
    """
    mean_field = np.mean(flowfield, axis=time_ax, keepdims=True)
    flx_field = flowfield - mean_field
    mean_field = mean_field[0, :, :]

    decomp_fields = [mean_field, flx_field]
    return decomp_fields

def plot_field_xy(x_grid, y_grid, field, title, cmap, range=None, filepath='plot.png', colorbar=True, save=True, 
                  dpi=300, vecs=False, field2=None, shading='auto', trimmed=False):
    """plot 2D field with specified colormap and, if save=True, save at high resolution.

    Args:
        x_grid (2d-array): x locations 
        y_grid (2d-array): y locations
        field (2d-array): values for field of interest
        cmap (matlap color map): color map to use for plotting.
        title (string): plot title
        filepath (string, optional): path to desired save location. Defaults to 'plot.png'.
        colorbar (bool, optional): whether to include color bar on figure. Defaults to True.
        save (bool, optional): whether to save plot to file. Defaults to true.
        dpi (int, optional): resolution at which to save plot. Defaults to 300.
        vecs (bool, optional): if true, overlays arrow plot of field specified in "field2" arg. Defaults to false.
        field2 (2d-array, optional): data for second field for overlaying arrow plot. Defaults to None. 
        shading (string, optional): specifies shading for pcolormesh plot. Defaults to 'auto'.
        trimmed (bool, optional): if true, trims plotted domain to y[-0.15, 0.15].
    """
    fig, ax = plt.subplots(figsize=(5.9, 4))

    if vecs:
        plt.streamplot(x_grid, y_grid, field, field2, density=1, linewidth=0.5, color='white')

    # define color map
    ctype = cmr.get_cmap_type(cmap)
    if ctype=='diverging':
        if range is None:
            vmin = np.min([np.nanmin(field), -np.nanmax(field)])
            vmax = np.max([np.nanmax(field), -np.nanmin(field)])
        else:
            vmin = range[0]
            vmax = range[1]
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        if range is None:
            vmin = np.nanmin(field)
            vmax = np.nanmax(field)
        else:
            vmin = range[0]
            vmax = range[1]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    mesh = ax.pcolormesh(x_grid, y_grid, field, cmap=cmap, norm=norm, shading=shading)

    if trimmed:
        ymin = -0.15
        ymax = 0.15
    else:
        ymin = min(y_grid[:, 0])
        ymax = max(y_grid[:, 0])
    
    ax.axis('equal')
    ax.set_adjustable('box')
    ax.set_ylim(ymin=ymin, ymax=ymax)
    # plt.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_title(title)
    if colorbar:
        ticks = [vmin, 3*vmin/4, vmin/2, vmin/4, 0, vmax/4, vmax/2, 3*vmax/4, vmax]
        fig.colorbar(mesh, ax=ax, ticks=ticks)
    if save:
        fig.savefig(filepath, dpi=dpi)
    else:
        fig.show()

