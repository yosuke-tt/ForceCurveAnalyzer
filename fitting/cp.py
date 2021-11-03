from __future__ import annotations
import os


import numpy as np
import matplotlib.pyplot as plt

from ..utils.TimeKeeper import TimeKeeper
from ._base_analyzer import *
from ..utils.fc_helper import *


class ContactPoint(FCBaseProcessor):
    def __init__(self,
                 measurament_dict: dict,
                 afm_param_dict: dict[str,float],
                 cp_th: float = 0.1,
                 num_th: int = 1000,
                 fm_div: float = 1e-11,
                 save_path: str = "./",
                 data_path: str = "",
                 dist_basecp: int = 1000,
                 logfile: str = 'fitlog.log'
                ) -> None:
        super().__init__(save_path, measurament_dict, afm_param_dict,data_path, logfile)
        self.cp_th = cp_th
        self.num_th = num_th
        self.fm_div = fm_div

    def get_cp_pre(self,
                   force_app: np.ndarray):
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
        cp_pre_th_e = fmin.reshape(-1, 1) + \
            ((fmax - fmin) * self.cp_th).reshape(-1, 1)
        cp_pre_th_e = np.array([np.where(f < c)[0][-1]
                                if len(np.where(f < c)[0]) > 1
                                else 0
                                for f, c in zip(force_app, cp_pre_th_e)])
        cp_pre = [[np.max([np.where(f < fm + self.fm_div)[0][-1], 0]), np.max([c, 4])]
                  if np.where(f < fm + self.fm_div)[0][-1] < c - self.num_th
                  else [np.max([c - self.num_th, 0]), np.max([c, 4])]
                  for f, fm, c in zip(force_app, fmin, cp_pre_th_e)]
        return cp_pre

    def linefit(self,
                x: np.ndarray,
                y: np.ndarray,
                cp: int = 0,
                d: int = 1) -> tuple(float, list(float, ...)) | list(float, ...):
        """
        線形フィッティングの関数

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

        coeffs = np.polyfit(x[cp:], y[cp:], d)

        if d == 1:
            residuals = np.mean(abs(y[cp:] - (x[cp:] * coeffs[0] + coeffs[1])))
            return residuals, [*coeffs]
        else:
            return [*coeffs]

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

    def search_cp(self, x: list, y: list, cps: list, is_plot: bool = False):
        """
        コンタクトポイントの候補から真のコンタクトポイントを決定する関数。
        Parameters
        ----------
        x : arr_like
        y : arr_like
            線形フィッティングするxyデータ
        cps : list
            コンタクトポイントの候補

        Returns
        -------
        [cp, coeffs[0], coeffs[1]]
            [コンタクトポイント, 傾き, 切片]
        """

        ratio = (len(x) - cps[1]) / len(x)
        if ratio > 0.05:
            d = np.array([self.linefit(x, y, cp)
                         for cp in range(cps[0], cps[1])], dtype=object)
            d_idx = np.argmin(d[:, 0])
            cp = int(cps[0] + d_idx)
            coeffs = d[d_idx, 1]
        else:
            cp = np.where(y < np.median(
                y[:100]) + (np.max(y) - np.median(y[:100])) * 0.1)[0][-1]
            res, coeffs = self.noisy_linefit(x, y, cp)
        line_fitted_data = [int(cp), coeffs[0], coeffs[1]]

        cc, base_fit = self.cross_contactpoint(x, y, line_fitted_data)

        if cc > cp or coeffs[0] < base_fit[0]:
            cc = cp
        if is_plot:
            plot_contact(self.save_path,self.i, x, y, line_fitted_data,[cc, *base_fit], cps)
        self.tk.timeshow()
        self.i += 1
        return [line_fitted_data, [cc, *base_fit]]

    def contact_search(self, delta_app, force_app, cp_pre):

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

    def fit(self, delta_app: np.ndarray, force_app: np.ndarray, plot: bool = False):
        """
        コンタクトポイントを取得する関数
        Returns
        -------
        line_fitted_data:arr_like
            [コンタクトポイント, 傾き, 切片]
        """
        cp_pre = self.get_cp_pre(force_app)
        self.i = 0
        self.tk = TimeKeeper(len(force_app))
        self.line_fitted_data = self.isfile_in_data_or_save("linfitdata.npy")
        self.cross_cp = self.isfile_in_data_or_save("cross_cp.npy")
        self.contact = self.isfile_in_data_or_save("contact.npy")
        if True or isinstance(self.line_fitted_data, bool) or isinstance(self.cross_cp, bool):
            self.contact_search(delta_app, force_app, cp_pre)
        return self.line_fitted_data, self.cross_cp
    
    @staticmethod
    def set_cp(cls,
               app_data: np.ndarray,
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




if __name__ == "__main__":
    force_app = np.load("contact/force_app.npy")
    delta_app = np.load("contact/delta_app.npy")

    cp = ContactPotint()
    # cp.get_cp(delta_app, force_app, plot=True)
