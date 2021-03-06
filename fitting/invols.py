from __future__ import annotations


import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from ._base_analyzer import FCBaseProcessor, pathLike
from ..parameters import *

class InvolsProcessing(FCBaseProcessor):
    def __init__(self,
                 save_path : pathLike,
                 measurament_dict: dict,
                 afm_param_dict: dict[str,float],
                 data_path : pathLike=None,
                 logfile: str = 'fitlog.log'
                 ):
        super().__init__(save_path, measurament_dict, afm_param_dict,data_path, logfile)

    def fit(self, def_app, z_app, missing_num, length_same=True):
        if self.save_name2path( "invols.npy"):
            
            if length_same:
                def_app_max = np.max(def_app, axis=1) * 0.99
                def_app_min = np.max(def_app, axis=1) * 0.1
            else:
                def_app_max = [ np.max(def_app_) * 0.99 for def_app_ in def_app ]
                def_app_min =[ np.max(def_app_) * 0.1 for def_app_ in def_app ]
            os.makedirs("da",exist_ok=True)
            
            invols = [1e9 / (self.linefit(z[(d < d_th_max) & (d > d_th_min)], d[(d < d_th_max) & (d > d_th_min)])[1][0])
                    for i, (d, d_th_max, d_th_min, z) in enumerate(zip(def_app, def_app_max, def_app_min, z_app))
                    if not (i in missing_num)]
            invols_mean = np.mean(invols)

            plt.hist(invols, label="mean : {}\nmax : {}\nmin : {}".format(
                invols_mean, np.max(invols), np.min(invols)))
            plt.legend()
            plt.savefig(self.save_name2path("hist"))
            plt.close()
            np.save(self.save_name2path( "invols_all.npy"), invols)
            np.savetxt(self.save_name2path("invols.txt"), [invols_mean])

        else:
            try:
                invols_mean = np.loadtxt(self.save_name2path("invols.txt"))[0]
            except ValueError:
                invols_mean = np.load(self.save_name2path("invols.npy"))
            finally:
                invols_mean = 200
        return invols_mean

    def load_data(self, fc_path,  load_row_fc_kargs={}):
        load_row_fc_kargs{"map_shape_square_strict"} = False 
        self.measurament_dict["zig"]=  False 
        super().load_data(fc_path, load_row_fc_kargs)
        def_app, def_ret = self.split_app_ret(self.deflection)
        z_app, z_ret = self.split_app_ret(self.zsensor)
        data = ((def_app, def_ret), (z_app, z_ret))
        return data, self.missing_num, self.length_same #????????????????????????????????????

if __name__ == "__main__":
    f = "../data_20210222/data_210542_invols"
    f = "../data_20210309/data_014506_invols/"
    xstep, ystep, zig = get_config(f)
    map_shape = (int(xstep), int(ystep))
    ip = InvolsProcessing(map_shape=map_shape, save_path=f)
    ip.fit(f)
