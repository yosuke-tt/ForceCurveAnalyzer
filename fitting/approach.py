from __future__ import annotations
import os


import numpy as np
import matplotlib.pyplot as plt

from ..utils.TimeKeeper import TimeKeeper
from ._base_analyzer import *
from ..utils.fc_helper import *

from .hertz_contact_model import *


class FCApproachAnalyzer(FCBaseProcessor):
    def __init__(self,
                 measurament_dict: dict,
                 afm_param_dict: dict[str,float],
                 save_path: str = "./",
                 data_path: str = "",
                 logfile: str = 'fitlog.log',
                 invols : float = 200
                ) -> None:
        super().__init__(save_path, measurament_dict, afm_param_dict,data_path, logfile, invols)

    def get_cp_pre(self,
                   force_app: np.ndarray,
                   fm_div:float = 1e-11,
                   num_th:int = 1000,
                   cp_th: float = 0.1,
                   minimum_length = 4,

                   ):
        """
        コンタクトポイントの候補を決める関数。

        Parameters
        ----------
        cp_th : tuple, optional
            コンタクトポイントの候補の範囲の最大値の割合。, by default (0.01, 0.2)

        Returns
        -------
        cp_pre : list
            コンタクトポイントの候補のindexの範囲
        """
        fmax = np.array([np.max(fa) for fa in force_app])
        fmin = np.array([np.min(fa) for fa in force_app])
        
        # 最大最小の幅の、割合から、コンタクトポイント決定のための最低値を決定。
        cp_pre_th_e = fmin.reshape(-1, 1) + ((fmax - fmin) * cp_th).reshape(-1, 1)
        
        # 上で決定した値以下の中での最大値(最も下から考えたときに近い)のidx
        cp_pre_th_e = np.array([np.where(f < c)[0][-1]
                                if len(np.where(f < c)[0]) > 1
                                else 0
                                for f, c in zip(force_app, cp_pre_th_e)])
        # (負にならない)コンタクトポイントの探索幅の下側の決定。
        # 最小値からちょっと上。fm_div
        # コンタクトポイントの探索幅の上側の決定。
        # 上で決めたやつ
        cp_pre = [
                    [
                        np.max([np.where(f < fm + fm_div)[0][-1], 0]), 
                        np.max([c, minimum_length])
                    ]
                  if np.where(f < fm + fm_div)[0][-1] < c - num_th
                  else [np.max([c - num_th, 0]), np.max([c, minimum_length])]
                  for f, fm, c in zip(force_app, fmin, cp_pre_th_e)
                ]

        return cp_pre



    def noisy_linefit(self, x: list, y: list, cp: list, d=1) -> tuple(float, list(float, ...)):
        """
        データ点数が少ない基板などの線形フィッティングの関数

        Parameters
        ----------
        x : arr_like
            x軸のデータ
        y : arr_like
            y軸のデータ
        cp : int
            コンタクトポイント
        d : int, optional
            次元, by default 1

        Returns
        -------
        residuals:float
            残差
        [a,b]:list[float, float]
            線形回帰の係数
        """
        dr = int(len(x[cp:]) / 10)
        x_med_start, x_med_end = np.median(x[cp:cp + dr]), np.median(x[-dr:])
        y_med_start, y_med_end = np.median(y[cp:cp + dr]), np.median(y[-dr:])

        a = (y_med_end - y_med_start) / (x_med_end - x_med_start)
        b = y_med_end - a * x_med_end
        coeffs = [a, b]

        residuals = np.mean(abs(y[cp:] - (x[cp:] * coeffs[0] + coeffs[1])))
        return residuals, [*coeffs]

    def intersection_cp(self, coeff1: list(float, float), coeff2: list(float, float)) -> list(float, float):
        """
        二直線の交点によるコンタクトポイントを決定する関数。

        Parameters
        ----------
        coeff1,coeff2: list(float,float)
            係数

        Returns
        -------
        [x0, y0] : list(float,float)
            交点
        """
        x0 = -1 * (coeff1[1] - coeff2[1]) / (coeff1[0] - coeff2[0])
        y0 = x0 * coeff1[0] + coeff1[1]
        return [x0, y0]

    def cross_contactpoint(self, 
                           delta_app: np.ndarray, 
                           force_app: np.ndarray,
                           line_fitted_data: np.ndarray,
                           cp_mean_num:int =10) -> tuple(np.ndarray, np.ndarray):
        """二直線の交点によるコンタクトポイントを決定する関数。

        Parameters
        ----------
        delta_app, force_app: np.ndarray, np.ndarray
            押し込み両、力
        
        line_fitted_dat : np.ndarray
            立ち上がりを線形フィッティングした結果。
        
        cp_mean_num: int, optional
            ベースラインがデータとして少なすぎるものがある,その最低値 by default 10

        Returns
        -------
        [x0, y0] : list(float,float)
            交点
        """

        if line_fitted_data[0] > cp_mean_num:

            cp_base = np.min([line_fitted_data[0], cp_mean_num])
            
            #平均をとる範囲
            #NOTE:なんか変 　
            #(0~np.max([2,  cp_base-cp_mean_num]))~(0~np.max([cp_base, line_fitted_data[0]]))
            cp_base_range = np.arange(
                                        np.max([2,  cp_base-cp_mean_num]),#2点はないと直線fittingできないため。 
                                        np.max([cp_base, line_fitted_data[0]]) #最大でも現時点のコンタクトポイントまで。
                                    )
            base_fit_all = np.array([self.linefit(delta_app[:c], force_app[:c], cp=0)[1]
                                     for c in cp_base_range])
            
            #baselineの傾きを平均して決定。
            bfm = np.mean(base_fit_all[:, 0])
            base_fit = base_fit_all[np.argmin(np.abs(bfm - base_fit_all[:, 0]))]
            ic = self.intersection_cp(line_fitted_data[1:], base_fit)
            #x方向はindexで決定。
            #y方向は、その線形フィッティングしたものの最も近い(幅がthいない)
            cross_cp = np.argmin(np.abs(force_app - ic[1]))
        else:
            base_fit = [0, 0]
            cross_cp = line_fitted_data[0]
        return cross_cp, base_fit

    def search_cp(self, x: list, 
                        y: list,
                        cps: list,
                        is_plot: bool = False,
                        ratio_th : float = 0.05):
        """
        コンタクトポイントの候補からコンタクトポイントを決定する関数。
        Parameters
        ----------
        x : arr_like
        y : arr_like
            線形フィッティングするxyデータ
        cps : list
            コンタクトポイントの候補
        ratio_th: float , optional
            コンタクトポイントの候補がデータ点数に対して大きい場合noisyfitに変更する。
            その割合の閾値
        Returns
        -------
        [cp, coeffs[0], coeffs[1]]
            [コンタクトポイント, 傾き, 切片]
        """

        ratio = (len(x) - cps[1]) / len(x)

        if ratio > ratio_th:
            #後方の中をすべて線形fittingして最小値をコンタクトポイントとする。
            d = np.array([self.linefit(x, y, cp)
                         for cp in range(cps[0], cps[1])], dtype=object)
            d_idx = np.argmin(d[:, 0])
            cp = int(cps[0] + d_idx)
            coeffs = d[d_idx, 1]
        else:
            #なんで必要かちょっと忘れた。
            cp = np.where(y < np.median(
                    y[:100]) + (np.max(y) - np.median(y[:100])
                ) * 0.1)[0][-1]
            res, coeffs = self.noisy_linefit(x, y, cp)
        
        line_fitted_data = [int(cp), coeffs[0], coeffs[1]]
        
        #分離した方がいいかもしれんな
        cc, base_fit = self.cross_contactpoint(x, y, line_fitted_data)
        
        if cc > cp or coeffs[0] < base_fit[0]:
            cc = cp
        if is_plot:
            plot_contact(self.save_path,self.i, x, y, line_fitted_data,[cc, *base_fit], cps)
        self.tk.timeshow()
        return [line_fitted_data, [cc, *base_fit]]

    def contact_search(self, 
                        delta_app, 
                        force_app,
                        cp_pre):
        """cp_pre（コンタクトポイントの候補から、）

        Parameters
        ----------
        delta_app : arr-like
            押し込み両あっぽ
        force_app : arr-like
            押し込みフォースあっぽ
        cp_pre : list
            ｃｐ候補
        """
        scp = np.array([self.search_cp(d, f, cps)
                        for d, f, cps in zip(delta_app, (np.array(force_app)**2)**(1 / 3), cp_pre)])

        self.line_fitted_data = scp[:, 0]
        self.cross_cp = scp[:, 1]

        np.save(self.save_name2path( "linfitdata.npy"),
                self.line_fitted_data)
        np.save(self.save_name2path( "contact.npy"),
                self.line_fitted_data[:, 0])
        np.save(self.save_name2path( "cross_cp.npy"),
                self.cross_cp)
    

    @data_statistics_deco(ds_dict={"data_name": "YoungE"})
    def get_E(self):
        E = fit_hertz_E(
            self.line_fitted_data[:, 1],
            bead_radious=self.afm_param_dict["bead_radias"], #スペルミス
            poission_ratio = self.afm_param_dict["poission_ratio"]
        )
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

    def fit(self, 
            delta_app: np.ndarray, 
            force_app: np.ndarray, 
            zsensor,
            cp_kargs={},
            plot: bool = False):
        """
        コンタクトポイントを取得する関数
        Returns
        -------
        line_fitted_data:arr_like
            [コンタクトポイント, 傾き, 切片]
        """
        cp_pre = self.get_cp_pre(force_app, cp_kargs)
        self.tk = TimeKeeper(len(force_app))
        self.line_fitted_data = self.isfile_in_data_or_save("linfitdata.npy")
        self.cross_cp = self.isfile_in_data_or_save("cross_cp.npy")
        self.contact = self.isfile_in_data_or_save("contact.npy")

        if isinstance(self.line_fitted_data, bool) or isinstance(self.cross_cp, bool):
            self.contact_search(delta_app, force_app, cp_pre)

        self.E = self.get_E()
        self.contact = np.array(self.line_fitted_data[:, 0], dtype=np.int32)
        np.save(self.save_name2path("contact"), self.contact)

        self.topo_contact = self.get_topo_img(zsensor, self.contact)
        self.get_cross_topo(zsensor, self.cross_cp[:, 0])

        return self.line_fitted_data, self.cross_cp
    
    @staticmethod
    def set_cp(app_data: np.ndarray,
               cp      : list):
        """
        コンタクトポイントを基準にしたデータに変換する。

        Parameters
        ----------
        app_data : array_like
            アプローチ部分のデータ
        cp : array_like
            コンタクトポイントのデータ
        Returns
        -------
        app_data:np.ndarray
            コンタクトポイントを基準にしたアプローチ部分のデータ
        """
        data_cp = np.array([d[cp:]-d[cp] for d, cp in zip(app_data, cp)])
        return data_cp
                
    def load_data(self, fc_path, load_row_fc_kargs={}):
        
        fc_row_data = self.load_row_fc(fc_path=fc_path, 
                                       map_shape_square_strict=True,
                                       complement=True,
                                       **load_row_fc_kargs)
        self.deflection, self.zsensor = self.split_def_z(fc_row_data)
        self.deflection_row = self.deflection
        del fc_row_data
        self.deflection = self.set_deflectionbase()
        self.delta = self.get_indentaion()
        self.force = self.def2force()
        delta_app, delta_ret = self.split_app_ret(self.delta)
        force_app, force_ret = self.split_app_ret(self.force)
        data = ((delta_app, delta_ret), (force_app, force_ret), self.zsensor)

        return data, self.missing_num, self.length_same #なんかよくない気がする。






    