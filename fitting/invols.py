import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from ._base_analyzer import FCBaseProcessor
from ..parameters import *

class InvolsProcessing(FCBaseProcessor):
    def __init__(self,iofilePathes,
                 meas_dict,
                 afmParam=AFMParameters(),
                 ):
        super().__init__(meas_dict=meas_dict,
                         iofilePathes=iofilePathes,
                         afmParam=afmParam)

    def fit(self, def_app, z_app, complement_num):
        if self.ioPathes.save_name2path( "invols.npy"):

            def_app_max = np.max(def_app, axis=1) * 0.99
            def_app_min = np.max(def_app, axis=1) * 0.1

            invols = [1e9 / (self.linefit(z[(d < d_th_max) & (d > d_th_min)], d[(d < d_th_max) & (d > d_th_min)])[1][0])
                      for i, (d, d_th_max, d_th_min, z) in enumerate(zip(def_app, def_app_max, def_app_min, z_app))
                      if not (i in complement_num)]
            invols_mean = np.mean(invols)

            plt.hist(invols, label="mean : {}\nmax : {}\nmin : {}".format(
                invols_mean, np.max(invols), np.min(invols)))
            plt.legend()
            plt.savefig(self.ioPathes.save_name2path("hist"))
            plt.close()
            np.save(self.ioPathes.save_name2path( "invols_all.npy"), invols)
            np.savetxt(self.ioPathes.save_name2path("invols.txt"), [invols_mean])

        else:
            try:
                invols_mean = np.loadtxt(self.ioPathes.save_name2path("invols.txt"))[0]
            except ValueError:
                invols_mean = np.load(self.ioPathes.save_name2path("invols.npy"))
            finally:
                invols_mean = 200
        return invols_mean


if __name__ == "__main__":
    f = "../data_20210222/data_210542_invols"
    f = "../data_20210309/data_014506_invols/"
    xstep, ystep, zig = get_config(f)
    map_shape = (int(xstep), int(ystep))
    ip = InvolsProcessing(map_shape=map_shape, save_path=f)
    ip.fit(f)
