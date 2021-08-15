import os
import sys


import matplotlib.pyplot as plt
import numpy as np

import japanize_matplotlib

from ._base_analyzer import FCBaseProcessor
from .gradientAdjsutment import GradientAdjsutment
from .cp import ContactPoint

from ..utils.decorators import data_statistics_deco

#　文字設定
plt.rcParams["font.family"] = "sans-serif"


class FCApproachAnalyzer(FCBaseProcessor):
    def __init__(self, config_dict, param_filename="params.txt", save_path="./data", data_path=False):
        super().__init__(save_path=save_path, data_path=data_path, config_dict=config_dict)

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
        if z[0] > z[self.length_of_app]:
            min_leftidx = np.argmin(z[:self.length_of_app])
            z_left = (z[self.length_of_app] - z[min_leftidx]) * (np.arange(0, min_leftidx) - \
                      min_leftidx) / (self.length_of_app - min_leftidx) + z[min_leftidx]
            z_r = np.hstack([z_left, z[min_leftidx:]])
            try:
                with open(os.path.join(self.save_path, "ForceCurve/ForceCurve_{:>03}_zchanged.txt".format(i)), "w") as f:
                    print(z_r, file=f)
                plt.plot(z_r)
                plt.savefig(os.path.join(self.save_path,
                            "ForceCurve/ForceCurve_{:>03}_zchanged".format(i)))
                plt.close()
            except FileNotFoundError:
                with open(os.path.join(self.save_path, "ForceCurve_{:>03}_zchanged.txt".format(i)), "w") as f:
                    print(z_r, file=f)
                plt.plot(z_r)
                plt.savefig(os.path.join(self.save_path,
                            "ForceCurve_{:>03}_zchanged".format(i)))
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

    def polynomialfit(self, coefs, x, y):
        """
        二次関数のフィッティングをし残差を計算する関数。

        Parameters
        ----------
        coefs : np.ndarray
            二次関数の係数
        x, y : np.ndarray-
            データ

        Returns

        residential : np.ndarray
            残差
        """
        a = coefs[0]
        b = coefs[1]
        c = coefs[2]
        residual = y - (a * x**2 + b * x + c)

        return residual

    def fit_hertz_E(self, a_fit=None):
        """
        線形フィットしたデータからヤング率を求める関数
        Parameters:
        a_fit : float
            線形近似によるパラメータ
        """
        para = (4 * self.R**0.5) / (3 * (1 - self.v**2))
        self.E_hertz = (1 / para) * (a_fit**(3 / 2))
        return self.E_hertz

    @data_statistics_deco(ds_dict={"data_name": "YoungE"})
    def get_E(self):
        E = self.fit_hertz_E(self.line_fitted_data[:, 1])
        return np.array(E)

    def get_cross_topo(self, zsensor, cross_cp):
        topo = np.array([z[int(c)] if isinstance(c, int) else z[0]
                        for z, c in zip(zsensor, cross_cp)]).reshape(self.map_shape)
        np.save(os.path.join(self.save_path, "cross_topo"), topo)
        plt.imshow(topo, cmap="Greys")
        plt.savefig(os.path.join(self.save_path, "cross_topo"))
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
            [z[c] for z, c in zip(zsensor, contact)]).reshape(self.map_shape)
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
        cp = ContactPoint(save_path=self.save_path, data_path=self.data_path,
                          config_dict=self.config_dict)
        self.line_fitted_data, self.cross_cp = cp.get_cp(
            delta_app=delta_app, force_app=force_app, plot=True)
        self.E = self.get_E()
        self.contact = np.array(self.line_fitted_data[:, 0], dtype=np.int32)
        np.save(os.path.join(self.save_path, "contact"), self.contact)

        self.topo_contact = self.get_topo_img(zsensor, self.contact)
        self.get_cross_topo(zsensor, self.cross_cp[:, 0])

        self.cos_map = self.isfile_in_data_or_save("cos_map")
        if isinstance(self.cos_map, bool):
            ga = GradientAdjsutment(map_shape=self.map_shape)
            self.cos_map = ga.gradient_topo(self.topo_contact[0])
            self.cos_map = ga.edge_filter(self.cos_map)
            np.save(self.savefile2savepath("cos_map"), self.cos_map)
        return self.E, self.line_fitted_data, self.cross_cp, self.topo_contact, self.cos_map
