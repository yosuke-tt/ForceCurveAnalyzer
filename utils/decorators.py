import os

import numpy as np
import matplotlib.pyplot as plt


def data_statistics_deco(ds_dict: dict):
    """データ保存のためのデコレータ

    Parameters
    ----------
    ds_dict : dict
        データ保存のためのデコレータのdict
    """
    def ds_func(func:callable):
        """デコレータ

        Parameters
        ----------
        func : callable
            リターンを保存する関数。

        Returns
        -------
        [type]
            [description]
        """
        try:
            data_name = ds_dict["data_name"]
        except KeyError:
            data_name = func.__name__
        try:
            vmin = ds_dict["vmin"]
        except KeyError:
            vmin = 0
        try:
            skip_idx = ds_dict["skip_idx"]
        except KeyError:
            skip_idx = [-1]

        try:
            stat_type = ds_dict["stat_type"]
        except KeyError:
            stat_type = ["map", "map_only", "hist"]

        def data_statistics(self, *args):
            nonlocal data_name
            nonlocal vmin
            nonlocal stat_type
            nonlocal skip_idx
            data = func(self, *args)

            if not isinstance(data, tuple):
                data = [data]

            if (not isinstance(data_name, tuple)) and (not isinstance(data_name, list)):
                data_name = [data_name]
            if len(data_name) < len(data):
                data_name = [data_name + "_" + str(i) for i in range(len(data))]

            for i, (d, dn) in enumerate(zip(data, data_name)):
                if i in skip_idx:
                    continue
                try:
                    d = np.array(d)
                    np.save(self.save_name2path( dn), d)

                    num_plot = len(stat_type)
                    min_idx = np.argsort(d.reshape(-1, 1))[1]

                    data_std = np.std(d[d > 0])
                    vmin = np.nanmax([0, np.nanmean(d[d > 0]) - 2 * data_std])

                    vmax = np.nanmean(d) + 2 * data_std
                    d = np.nan_to_num(d, nan=vmin)
                    if "map_only" in stat_type:
                        plt.imshow(d.reshape(self.measurament_dict["map_shape"]),
                                   vmin=vmin, vmax=vmax, cmap="Greys")
                        plt.colorbar()
                        plt.savefig(self.save_name2path( dn))
                        plt.close()
                        num_plot -= 1

                        # plt.imshow(d.reshape(self.measurament_dict["map_shape"]),vmin=vmin, vmax=vmax, cmap="seismic")
                        # plt.colorbar()
                        # plt.savefig(self.save_name2path(dn + "_colored"))
                        # plt.close()
                    fig, ax = plt.subplots(
                        1, num_plot, figsize=(8 * num_plot, 5))
                    i = 0
                    data_max = np.nanmax(d)
                    data_min = np.nanmin(d)
                    data_mean = np.nanmean(d)
                    try:
                        data_median = np.nanmedian(d)
                    except TypeError:
                        data_median = np.nanmean(d)
                    if "map" in stat_type:
                        mapple = ax[i].imshow(
                            d.reshape(self.measurament_dict["map_shape"]), vmin=vmin, vmax=vmax, cmap="Greys")
                        plt.colorbar(mapple, ax=ax[i])
                        i += 1
                    if "hist" in stat_type:
                        d_hist = d[vmin < d].reshape(-1, 1)
                        d_hist = d_hist[d_hist < vmax].reshape(-1, 1)
                        ax[i].hist(d_hist)
                        i += 1
                    if "plot" in stat_type:
                        ax[i].plot(d.reshape(-1, 1))
                        i += 1
                    ax[i // 2].set_title("{} ,max : {:6.2f}, min : {:6.2f}, mean : {:6.2f}, median : {:6.2f}".format(
                        dn, data_max, data_min, data_mean, data_median))
                    plt.savefig(self.save_name2path( dn + "stat"))
                    plt.close()
                except TypeError:
                    print("data type error")
            return data
        return data_statistics

    return ds_func
