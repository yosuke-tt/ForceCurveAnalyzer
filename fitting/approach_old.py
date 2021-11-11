from __future__ import annotations
import os
import sys


import matplotlib.pyplot as plt
import numpy as np

import japanize_matplotlib

from ._base_analyzer import FCBaseProcessor, pathLike
from .gradientAdjsutment import GradientAdjsutment
from .cp import ContactPoint

from ..utils import data_statistics_deco
from ..parameters import *
#　文字設定
plt.rcParams["font.family"] = "sans-serif"


class FCApproachAnalyzer(FCBaseProcessor):
    def __init__(self,
                 save_path : pathLike,
                 measurament_dict: dict,
                 afm_param_dict: dict[str,float],
                 data_path : pathLike=None,
                 logfile: str = 'fitlog.log',
                 ):
        super().__init__(save_path, measurament_dict, afm_param_dict,data_path, logfile)


    def fit_hertz_E(self, a_fit=None):
        """
        線形フィットしたデータからヤング率を求める関数
        Parameters:
        a_fit : float
            線形近似によるパラメータ
        """
        para = (4 * self.afm_param_dict["bead_radias"]**0.5) \
                / (3 * (1 - self.afm_param_dict["poission_ratio"]**2))
        self.E_hertz = (1 / para) * (a_fit**(3 / 2))
        return self.E_hertz

    @data_statistics_deco(ds_dict={"data_name": "YoungE"})
    def get_E(self):
        E = self.fit_hertz_E(self.line_fitted_data[:, 1])
        return np.array(E)

    @data_statistics_deco(ds_dict={"data_name": "cross_topo_contact"})
    def get_cross_topo(self, zsensor, cross_cp):
        topo = np.array([z[int(c)] if isinstance(c, int) else z[0]
                        for z, c in zip(zsensor, cross_cp)]).reshape(self.measurament_dict["map_shape"])
                
        return self.topo_contact

    @data_statistics_deco(ds_dict={"data_name": "topo_contact"})
    def get_topo_img(self, zsensor, contact):
        """
        トポグラフィー像の取得

        Parameters
        ----------
        zsensor : arr_like
            zセンサー値
        contact : arr_like
            コンタクトポイント
        topo_trig : bool
            トリガー電圧でのトポグラフィー像
        topo_contact : bool
            コンタクトポイントでのトポグラフィー像
        """
        self.topo_contact = np.array([z[c] for z, c in zip(zsensor, contact)]).reshape(self.measurament_dict["map_shape"])
        return self.topo_contact

    def fit(
            self,
            delta_app,
            force_app,
            zsensor,
            map_fc=False,
            fitted_all_img=False,
            processed=False,
            topo_trig=False,
            topo_contact=True,
            contact_all_img=False,
            logger=False):
        self.logger.debug("Started searching contact point")
        #TODO: ContactPoint分ける。GradientAdjasmentの分割
        print()
        cp = ContactPoint(save_path=self.save_path, 
                          measurament_dict=self.measurament_dict,
                          data_path=self.data_path, 
                          afm_param_dict=self.afm_param_dict)

        self.line_fitted_data, self.cross_cp = cp.fit(delta_app=delta_app, force_app=force_app, plot=True)
        self.E = self.get_E()
        self.contact = np.array(self.line_fitted_data[:, 0], dtype=np.int32)
        np.save(self.save_name2path("contact"), self.contact)

        self.topo_contact = self.get_topo_img(zsensor, self.contact)
        self.get_cross_topo(zsensor, self.cross_cp[:, 0])
        return self.E, self.line_fitted_data, self.cross_cp, self.topo_contact