import os
import sys


import matplotlib.pyplot as plt
import numpy as np

import japanize_matplotlib

from ._base_analyzer import FCBaseProcessor
from .gradientAdjsutment import GradientAdjsutment
from .cp import ContactPoint

from ..utils import data_statistics_deco
from ..parameters import *
#　文字設定
plt.rcParams["font.family"] = "sans-serif"


class FCApproachAnalyzer(FCBaseProcessor):
    def __init__(self,
                 meas_dict: dict,
                 iofilePathes: IOFilePathes,
                 afmParam: AFMParameters = AFMParameters()
                 ):
        super().__init__(meas_dict=meas_dict, iofilePathes=iofilePathes, afmParam=afmParam)

    def check_z(self, i, z):
        """
        zsensorが二段階に曲がる時に変更する

        Parameters
        ----------
        i : int
            データの番号
        z : np.ndarray
            zsensor

        Returns
        -------
        z_r : np.ndarray
            チェックしたデータ
        """
        if z[0] > z[self.meas_dict["app_points"]]:
            min_leftidx = np.argmin(z[:self.meas_dict["app_points"]])
            z_left = (z[self.meas_dict["app_points"]] - z[min_leftidx]) * (np.arange(0, min_leftidx) - \
                      min_leftidx) / (self.meas_dict["app_points"] - min_leftidx) + z[min_leftidx]
            z_r = np.hstack([z_left, z[min_leftidx:]])
            try:
                with open(self.ioPathes.save_name2path( "ForceCurve/ForceCurve_{:>03}_zchanged.txt".format(i)), "w") as f:
                    print(z_r, file=f)
                plt.plot(z_r)
                plt.savefig(self.ioPathes.save_name2path("ForceCurve/ForceCurve_{:>03}_zchanged".format(i)))
                plt.close()
            except FileNotFoundError:
                with open(self.ioPathes.save_name2path("ForceCurve_{:>03}_zchanged.txt".format(i)), "w") as f:
                    print(z_r, file=f)
                plt.plot(z_r)
                plt.savefig(self.ioPathes.save_name2path("ForceCurve_{:>03}_zchanged".format(i)))
                plt.close()
        else:
            z_r = z
        return z_r

    def set_zbase(self):
        """
        zセンサーの基準値を端の最大値に合わせる
        """
        # カンチレバーのZを基準からにする。
        z_base = np.array([np.max([z[0], z[-1]])
                          for z in self.zsensor]).reshape(-1, 1)

        # z_base = np.array([ z[0] for z in self.zsensor]).reshape(-1,1)
        self.zsensor -= z_base
        self.zsensor = np.array([self.check_z(i, zz)
                                for i, zz in enumerate(self.zsensor - z_base)])
        return self.zsensor



    def fit_hertz_E(self, a_fit=None):
        """
        線形フィットしたデータからヤング率を求める関数
        Parameters:
        a_fit : float
            線形近似によるパラメータ
        """
        para = (4 * self.afmParam.bead_radias**0.5) \
                / (3 * (1 - self.afmParam.poission_ratio**2))
        self.E_hertz = (1 / para) * (a_fit**(3 / 2))
        return self.E_hertz

    @data_statistics_deco(ds_dict={"data_name": "YoungE"})
    def get_E(self):
        E = self.fit_hertz_E(self.line_fitted_data[:, 1])
        return np.array(E)

    def get_cross_topo(self, zsensor, cross_cp):
        topo = np.array([z[int(c)] if isinstance(c, int) else z[0]
                        for z, c in zip(zsensor, cross_cp)]).reshape(self.meas_dict["map_shape"])
        
        np.save(self.ioPathes.save_name2path("cross_topo"), topo)
        
        plt.imshow(topo, cmap="Greys")
        plt.savefig(self.ioPathes.save_name2path("cross_topo"))
        plt.close()
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
        self.topo_contact = np.array(
            [z[c] for z, c in zip(zsensor, contact)]).reshape(self.meas_dict["app_points"])
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
        cp = ContactPoint(meas_dict=self.meas_dict, 
                          iofilePathes=self.iofilePathes, 
                          afmParam=self.afmParam)

        self.line_fitted_data, self.cross_cp = cp.get_cp(delta_app=delta_app, force_app=force_app, plot=True)
        self.E = self.get_E()
        self.contact = np.array(self.line_fitted_data[:, 0], dtype=np.int32)
        np.save(self.ioPathes.save_name2path("contact"), self.contact)

        self.topo_contact = self.get_topo_img(zsensor, self.contact)
        self.get_cross_topo(zsensor, self.cross_cp[:, 0])
        return self.E, self.line_fitted_data, self.cross_cp, self.topo_contact
