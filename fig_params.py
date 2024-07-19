import os, sys
from importlib import reload
from collections import defaultdict, namedtuple

import utils
import boulder
import fisher_plumes_fig_tools as fpft
logger = utils.create_logger(__name__)
INFO = logger.info
DEBUG = logger.debug

from utils import dict_update_from_field

DEFAULT   = "default"
isdefault = lambda x: type(x) is str and x == DEFAULT

class PlumesDemo:
    def __init__(self, UNITS, su_ds = []):
        self.which_srcs   = dict_update_from_field({"bw":[7,-8], #[-3750, 3750],                                       
                                                    "bw_X": [1, -2], # [-48750, 48750],
                                                    "bw_45":[1,-2], #[-48749, 48749],
                                                    "16Ts":[7,-8], #[496000,504000],
                                                    "16Ts_X":[7,-8], #[16000,104000],
                                                    "16Ts_45":[7,-8], #[16000, 104000],
                                                    },       
                                      su_ds, "bw")

        all_but_bw = ["bw_X", "bw_45", "16Ts", "16Ts_X", "16Ts_45"]
        self.t_wnd        = dict_update_from_field({"bw":[-4,4]*UNITS.sec}, su_ds + all_but_bw, "bw")
        self.which_idists = dict_update_from_field({"bw":[0,2,3]}, su_ds + all_but_bw, "bw")
        self.tticks       = dict_update_from_field({"bw":DEFAULT}, su_ds + all_but_bw, "bw")
        self.xticks       = dict_update_from_field({"bw":DEFAULT}, su_ds + all_but_bw, "bw")
        self.yticks       = dict_update_from_field({"bw":DEFAULT}, su_ds + all_but_bw, "bw")
        self.snapshot_time = defaultdict(lambda: 40000*UNITS.ms, {"16Ts":40010*UNITS.ms, "16Ts_X":40010*UNITS.ms, "16Ts_45":40010*UNITS.ms})
        self.snapshots_dir = defaultdict(lambda: None,
                                         {"bw": os.path.join(boulder.data_root, "original", "saved-snapshots"),
                                          "bw_X": os.path.join(boulder.data_root, "streamwise", "saved-snapshots"),
                                          "bw_45": os.path.join(boulder.data_root, "45deg", "saved-snapshots"),
                                          "16Ts": None,
                                          "16Ts_X": None,
                                          "16Ts_45": None,
                                          })

class CorrDecomp:
    def __init__(self, UNITS):
        self.xlims = defaultdict(lambda: DEFAULT)
        self.xticks = defaultdict(lambda: DEFAULT)
        self.which_freqs = defaultdict(lambda: [1,2,5,10] * UNITS.Hz)
        
class FigParams:
    def __init__(self, UNITS, compute_filter, su_ds = []):
        window_shape  = compute_filter["window_shape"]
        window_length = compute_filter["window_length"]
        fit_k         = compute_filter["fit_k"]
        
        self.fig_dir_full        = fpft.get_fig_dir(window_shape = window_shape, window_length = window_length, fit_k = fit_k, create = True); DEBUG(f"{self.fig_dir_full=}")
        self.fig_dir_wnd_shp_len = fpft.get_fig_dir(window_shape = window_shape, window_length = window_length, fit_k = None,  create = True); DEBUG(f"{self.fig_dir_wnd_shp_len=}")
        self.fig_dir_wnd_shp     = fpft.get_fig_dir(window_shape = window_shape, window_length = None,          fit_k = None,  create = True); DEBUG(f"{self.fig_dir_wnd_shp=}")
        self.fig_dir_top         = fpft.get_fig_dir(window_shape = None,         window_length = None,          fit_k = None,  create = True); DEBUG(f"{self.fig_dir_top=}")
        self.fig_dir_fitk        = fpft.get_fig_dir(window_shape = None,         window_length = None,          fit_k = fit_k, create = True); DEBUG(f"{self.fig_dir_fitk=}")


        self.plumes_demo = PlumesDemo(UNITS, su_ds = su_ds)
        self.corr_decomp = CorrDecomp(UNITS)
