
from __future__ import annotations

import os
import time
from glob import glob
from datetime import datetime
from datetime import timedelta

import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt
from matplotlib import rcParams
# フォントの設定
rcParams['font.family'] = 'sans-serif'

from ._base_analyzer import FCBaseProcessor, pathLike

from ..parameters import *
from ..utils import TimeKeeper
from ..utils.fc_helper import * 

class FitFC2Ting(FCBaseProcessor):
    def __init__(self,
                 save_path       : pathLike,
                 measurament_dict: dict[str,float],
                 afm_param_dict  : dict[str,float],
                 data_path       : pathLike = None,
                 logfile         : str      = 'fitlog.log',
                 norm            : int      =  50
                 ):
        super().__init__(save_path, measurament_dict,data_path, afm_param_dict, logfile)
        # 測定FC時

        self.tstep = self.afm_param_dict["t_dash"]
        
        self.norm = norm
        self.Hertz = True

        self.model_param = self.elastic_modelconstant()#まとめたい
        self.err = []

        self.change_x_data, self.change_y_data = [], []
        self.change_idx = []

        self.residuals = []
        self.fitting_result = []
        self.alpha_upper = 0.6
        self.larger_idx=[]

    def elastic_modelconstant(self):
        """
        ヘルツモデルのパラメータ作成

        Parameters
        ----------
        v : float, optional
            ポアソン比, by default 0.5
        R : float, optional
            半径, by default 5*10**(-6)

        Returns
        -------
        model_param : float
            ヘルツモデルの球の定数。
        """
        if self.Hertz:
            model_param = (4 * self.afm_param_dict["bead_radias"]**0.5) \
                                / (3 * (1 - self.afm_param_dict["poission_ratio"]**2))
        else:
            model_param = (2 * self.afm_param_dict["tan_theta"]) \
                            / (np.pi * (1 - self.afm_param_dict["poission_ratio"]**2))
        return model_param

    def base2zero(self, 
                  delta_app : list,
                  delta_ret : list,
                  force_app:list,
                  force_ret:list,
                  contact:list,
                  ret_contact:list,
                  ret_baseline:str="app",
                  is_plot:bool=True) -> tuple(np.ndarray, np.ndarray):
        """ベースラインをゼロに合わせるプログラム
        
        Parameters
        ----------
        delta_app, delta_ret, force_app, force_ret, contact : list
            contactは、アプローチのもの。
        ret_contact : list
            リトラクションでのアプローチのコンタクトポイントの位置。
        ret_baseline: list
            リトラクションのベースライン補正の方法。
            "contact"　: アプローチのコンタクトポイントを全体から引く。
            "app"　　　: アプローチのベースラインを全体から引く。
            "ret"(非推奨): アプローチのベースラインをアプローチ、リトラクションのベースラインからリトラクションを引く。
        is_plot:bool, optional
            結果を画像表示する。,default False

        Returns
        -------
        [type]
            [description]
        """
        #baseline何回も求めてるのカラムだな気もする。
        app_base_coeffs = [
                            self.linefit(d[:c], f[:c], cp=0)
                           if c > 3 
                           else [0, 0, 0] 
                           for d, f, c in zip(delta_app, force_app, contact)]
        
        if ret_baseline == "contact":
            #アプローチのcontactポイントをゼロとする。
            force_app_base = np.array([fca - fca[int(c)]
                                      for fca, c in zip(force_app, contact)], dtype=object)
            force_ret_base = np.array([fcr - fca[int(c)] 
                                       for fca, fcr, c in zip(force_app, force_ret, contact)], dtype=object)
        else:
            if ret_baseline == "ret":
                ret_base_coeffs = np.array([self.linefit(d[rc:][::-1], f[rc:][::-1], cp=0, d=1)
                                            if rc < 19996 else [0, 0] 
                                            for d, f, rc in zip(delta_ret, force_ret, ret_contact)])
            elif ret_baseline == "app":
                ret_base_coeffs = app_base_coeffs
            force_app_base = np.array([fca - (apc[0] * da + apc[1]) for fca, apc,
                                    da in zip(force_app, app_base_coeffs, delta_app)], dtype=object)
            force_ret_base = np.array([fcr - (rc[0] * dr + rc[1]) for fcr, rc,
                                    dr in zip(force_ret, ret_base_coeffs, delta_ret)], dtype=object)

        if is_plot:
            plot_set_base(self.save_path,self.delta_app,self.delta_ret,self.force_app,self.force_ret,self.contact,self.ret_contact,force_app_base,force_ret_base)
        return force_app_base, force_ret_base
    
    def get_ret_contact(self,
                        delta_app,
                        delta_ret,
                        app_contact,
                        is_ret_contact_img=False
                        ):
        ret_contact = [ np.where(dr <= da[c])[0][0] if da[c] > np.min(dr) else len(dr) - 1
                for dr, da, c in zip(delta_ret, delta_app, app_contact)]
        if is_ret_contact_img:
            ret_contact_img(self.save_path,self.delta_ret,self.delta_app,self.force_app,self.force_ret,self.contact,self.ret_contact)
        return ret_contact
    
    def preprocessing(
            self,
            delta_app,
            delta_ret,
            force_app,
            force_ret,
            contact,
            is_processed_img=True,
            ret_baseline="contact",
            kargs_ret_contact={},
            kargs_base_fit={}):
        """delta, forceのベースライン補正により前処理。

        Parameters
        ----------
        delta_app, delta_ret, force_app, force_ret, contact : list
            フォースカーブ
        is_processed_img : bool, optional
            処理結果の画像, by default True
        ret_baseline : str, optional
            retの補正方法の種類, by default "contact"
        kargs_ret_contact : dict, optional
            [description], by default {}
        kargs_base_fit : dict, optional
            [description], by default {}

        Returns
        -------
        [type]
            [description]
        """
        ret_contact = self.get_ret_contact(delta_app,delta_ret,contact, **kargs_ret_contact)
        
        force_app_base, force_ret_base = self.base2zero(delta_app,delta_ret,force_app,force_ret, contact,ret_baseline,ret_baseline,**kargs_base_fit)
        # "ret"以外いらないかも
        force_app_max = [np.max(fab) for fab in force_app_base]
        force_ret_corrected = np.array(
            [np.fmin(fam, frb) for fam, frb in zip(force_app_max, force_ret_base)], dtype=object)
        
        #da[0]はda[c]な気がするー
        delta_data = np.array([np.concatenate([da[c:], dr[:rc + 1]]) - da[0]
                              for da, dr, c, rc in zip(delta_app, delta_ret, contact, ret_contact)], dtype=object)
        force_data = np.array([np.concatenate([fa[c:], fr[:rc + 1]]) for fa, fr, c,
                              rc in zip(force_app_base, force_ret_corrected, contact, ret_contact)], dtype=object)

        delta_data = np.array([self.normalize(d, norm_length=self.norm) for d in delta_data])
        force_data = np.array([self.normalize(f, norm_length=self.norm) for f in force_data])
        if is_processed_img:
            plot_preprocessed_img(self.save_path,delta_data,force_data,self.delta, self.force)

        np.save(self.save_name2path("delta_preprocessed"), delta_data)
        np.save(self.save_name2path("force_preprocessed"), force_data)

        return delta_data, force_data

    def power_law_rheology_model(self, xi, t, param):

        if len(param) == 2:
            return self.model_param * (param[0] * (1 + ((t - xi) / self.afm_param_dict["tdash"]))**((-1) * param[1]))
        else:
            # return param[2]+(param[0]-param[2])*(1+(t-xi)/self.tdash)**(-1*param[1])
            if (t - xi) != 0.0:
                # Waring対策
                return self.model_param * (param[2] + (param[0] - param[2]) * ((t - xi) / self.afm_param_dict["t_dash"])**(-1 * param[1]))
            else:
                return self.model_param * param[2]

    def eq_PLR_integrand(self, xi, t, param, dstep):
        return self.power_law_rheology_model(xi, t, param) * (dstep**(3 / 2) * (3 / 2) * (xi**(0.5)))

    def eq_PLR_integrand_for_t1(self, xi, t, param, dstep):
        return self.power_law_rheology_model(xi, t, param) * dstep

    def tingmodel(self, x, e0, alpha):
        """
        使用には、変更必要
        ting model

        Parameters
        ----------
        x : np.ndarray
            押し込み量
        e0 : float
            E0
        alpha : float
            alpha
        Returns
        -------
        np.concatenate([force_app_fitted, force_ret_fitted])[:int(len(d_app)+len(d_ret))] : np.ndarray
            ting modelの結果
        """
        raise ValueError("現在使えても使うのだめ。")
        # ind_index = self.contact_point[self.index]
        # d_app = self.delta_app[self.index][ind_index:]
        # d_ret = self.delta_ret[self.index][ind_index:]

        d_app = self.delta_app[self.index]
        d_ret = self.delta_ret[self.index]
        t_app = self.t_app[self.index]

        tm = self.tm[self.index]

        if len(t_app) < 2:
            return [0]
        else:
            app_dstep = (d_app[-1] - d_app[0]) / (t_app[-1] - t_app[0])

        force_app_fitted = [integrate.quad(lambda xi: self.eq_PLR_integrand(
            xi, t=t, e0=e0, alpha=alpha, dstep=app_dstep), 0, t)[0] for t in t_app]
        t_ret = np.arange(tm, (len(d_app) + len(d_ret) - 1)* self.tstep, self.tstep)

        ret_dstep = (d_ret[-1] - d_ret[0]) / (t_ret[-1] - t_ret[0])

        t1 = np.zeros(len(t_ret))
        t1s_pro = np.arange(0, tm, 0.001)

        for i_t1, tr in enumerate(t_ret):
            int_pro = np.zeros(len(t1s_pro))
            ret_integral = integrate.quad(lambda xi: self.eq_PLR_integrand_for_t1(
                xi, t=tr, e0=e0, alpha=alpha, dstep=ret_dstep), tm, tr)[0]
            for i, t1p in enumerate(t1s_pro):
                int_pro[i] += np.abs(integrate.quad(lambda xi: self.eq_PLR_integrand_for_t1(xi, t=tr,
                                     e0=e0, alpha=alpha, dstep=app_dstep), t1p, tm)[0] + ret_integral)
            # t1_searched = optimize.least_squares(lambda t1 : np.abs(integrate.quad(lambda xi : self.eq_PLR_integrand_for_t1(xi, e0=e0, t=tr, alpha = alpha, dstep= app_dstep),t1p, tm)[0]
            #                                                         +ret_integral)
            #                                     parameter0=[tm]
            #                                 )
            t1[i_t1] += t1s_pro[np.argmin(int_pro)]
            t1s_pro = t1s_pro[:np.argmin(int_pro) + 1]
        force_ret_fitted = [
            integrate.quad(
                lambda xi: self.eq_PLR_integrand(
                    xi,
                    t=t_ret[i],
                    e0=e0,
                    alpha=alpha,
                    dstep=app_dstep),
                0,
                t)[0] if t != 0 else 0 for i,
            t in enumerate(t1)]
        fit_data = np.concatenate([force_app_fitted, force_ret_fitted])
        if len(fit_data) != int(len(d_app) + len(d_ret)) and not (self.index in self.err):
            self.err.append(self.index)
            self.logger.warning("Length of fit data and (d_app+d_ret) does not much {} {} {}".format(
                self.index, len(fit_data), int(len(d_app) + len(d_ret))))

        return np.concatenate([force_app_fitted, force_ret_fitted])[:int(len(d_app) + len(d_ret))]

    def tingmodel_offset(self, x, e0, alpha, einf):
        """
        offsetアリでのting model

        Parameters
        ----------
        x : np.ndarray
            押し込み量
        e0 : float
            E0
        alpha : float
            alpha
        einf : float
            einf
        Returns
        -------
        np.concatenate([force_app_fitted, force_ret_fitted])[:int(len(d_app)+len(d_ret))] : np.ndarray
            ting modelの結果
        """
        #!! curvefitでboundsが上手くきかないので強引にここで。
        e0 = np.max([0, e0])
        alpha = np.min([self.alpha_upper, np.max([0, alpha])])
        einf = np.max([0, einf])
        
        tm = x[0] * self.tstep
        
        t_app = np.arange(0, tm, self.tstep)
        #FIXME:x[0]は強引にapprochの長さを入れるために入れてる。
        #変数の入れ方としてd = (dapp, dret)
        
        d_app = x[1:][:int(x[0])]
        d_ret = x[1:][int(x[0]):]
        
        #微分計算
        app_dstep = (d_app[-1] - d_app[0]) / (t_app[-1] - t_app[0])
        
        #labmda式じゃなくていいかも
        force_app_fitted = [
                                integrate.quad(
                                        lambda xi: self.eq_PLR_integrand(xi, t=t, param=[e0, alpha, einf], dstep=app_dstep), 
                                        0, 
                                        t
                                    )[0] 
                                for t in t_app
                            ]

        t_ret = np.arange(tm, (len(d_app) + len(d_ret))* self.tstep, self.tstep)
        
        ret_dstep = (d_ret[-1] - d_ret[0]) / (t_ret[-1] - t_ret[0])
        t1 = np.zeros(len(t_ret))
        
        #FIXME:ここの最適化計算は、ちゃんとなおす。最も時間のボトルネック。
        # t1s_pro = np.arange(0, tm, 0.00001)
        
        for i_t1, tr in enumerate(t_ret):
            # int_pro = np.zeros(len(t1s_pro))
            ret_integral = integrate.quad(lambda xi: self.eq_PLR_integrand_for_t1(
                xi, t=tr, param=[e0, alpha, einf], dstep=ret_dstep), tm, tr)[0]
            # for i, t1p in enumerate(t1s_pro):
            #     int_pro[i] += np.abs(integrate.quad(lambda xi: self.eq_PLR_integrand_for_t1(xi, t=tr,
            #                          param=[e0, alpha, einf], dstep=app_dstep), t1p, tm)[0] + ret_integral)
            #                                 )
            # t1[i_t1] += t1s_pro[np.argmin(int_pro)]
            #いいのをつかお
            t1_searched = optimize.fmin_ncg(lambda t1 : np.abs(integrate.quad(
                                                                lambda xi : self.eq_PLR_integrand_for_t1(xi, e0=e0, t=tr, alpha = alpha, dstep= app_dstep),
                                                                t1, 
                                                                tm)[0]
                                                                +ret_integral
                                                            ),
                                            x0=(t1_searched),
                                            bounds=(0,t1_searched)
                                            )
            t1[i_t1] += t1_searched 
        
        force_ret_fitted = [
            integrate.quad(
                lambda xi: self.eq_PLR_integrand(xi,
                                                t=t_ret[i],
                                                param=[e0,alpha,einf],
                                                dstep=app_dstep),
                0,
                t)[0] if t != 0 else 0 for i,
            t in enumerate(t1)]

        f_fit = np.concatenate([force_app_fitted, force_ret_fitted])[:len(x[1:])]
        try:
            if np.mean(np.abs(f_fit - self.y_tmp)) < self.res_tmp:
                self.popt_tmp = [e0, alpha, einf]
                self.res_tmp = np.mean(np.abs(f_fit - self.y_tmp))
        except ValueError:
            pass
        return f_fit

    def change_fit_range(self, delta, force, idx, start_fit=20, end_fit=40):
        """forceCurveの立ち上がり下りの部分のデータのforce方向を補正する。

        Parameters
        ----------
        delta : [type]
            [description]
        force : [type]
            [description]
        idx : [type]
            [description]
        start_fit : int, optional
            [description], by default 20
        end_fit : int, optional
            [description], by default 40

        Returns
        -------
        [type]
            [description]
        """
        force_app = force[:np.argmax(force)]
        delta_app = delta[:np.argmax(force)]

        coeffs = np.vstack([self.linefit(delta_app, force_app, cp=i, d="easy_fit")
                         for i in range(
                                    len(force_app) - end_fit,
                                    len(force_app) - start_fit
                                )])
        
        coeffs_m = np.median(coeffs, axis=0)
        base_mid = (force[-1]+force[0]) / 2
        delta_new_start = (base_mid - coeffs_m[1]) / coeffs_m[0]
        delta_new = delta[delta > delta_new_start]
        
        force_new = force[delta > delta_new_start]
        
        
        os.makedirs(self.save_name2path("processed_img_for_ting"), exist_ok=True)
        plt.plot(delta, force, label="original")
        plt.plot(delta, coeffs_m[0]*x+coeffs_m[1],label="lines")
        plt.plot(delta_new, force_new, label="new")
        plt.vlines([delta_new], 
                   ymin=np.min(force), 
                   ymax=np.max(force), 
                   label="x_new_start",
                   color="red")
        plt.hlines([base_mid,base_mid], 
                   xmin=np.min(delta), 
                   xmax=np.max(delta), 
                   label="base_mid",
                   color="red")
        plt.hlines([0], 
                   xmin=np.min(delta), 
                   xmax=np.max(delta),
                   label="0")
        plt.legend()
        plt.savefig(self.save_name2path("processed_img_for_ting/{:>03}".format(idx)))
        plt.close()
        return delta_new, force_new

    def change_data(self, file_path, all_data, change_data, idx, info):

        with open(self.save_name2path("change_data_info.txt"), "a") as f:
            print(time.strftime("%Y%m%d %H:%M:%S"), file=f)
            print(file_path, file=f)
            print(idx, file=f)
            print(info, file=f)

        pre_data_path = os.path.basename(file_path).split(".")[0] + "_pre.npy"

        data_old = self.isfile_in_data_or_save(pre_data_path)

        if isinstance(data_old, bool):
            data_old = np.array([[idx, change_data]])
        else:
            data_old = np.vstack([data_old, [idx, change_data]])
        np.save(self.save_name2path(pre_data_path), data_old)
        all_data[idx] = change_data
        np.save(self.save_name2path(file_path), all_data)

    def objective_ting(
            self,
            delta,
            force,
            idx=0,
            offset=False,
            data_ratio=[0, 0.6],
            optimize_times=3,
            res_th_upper=1e-19,
            e1_th_lawer=10):
        """
        tingmodelフィッティングの際に最適化する関数。

        Parameters
        ----------
        iv : array_like
            初期値
        delta : np.ndarray
            押し込み量
        force : np.ndarray
            力
        offset : bool, optional
            offsetを使用するかどうか, by default False

        Returns
        -------
        popt
            最適解[e0, alpha, e1,]
        """
        self.y_tmp_row = force
        self.res_tmp = np.inf
        self.index = idx
        
        data_range = len(force) * np.array(data_ratio)
        data_start, data_end = int(data_range[0]), int(data_range[1])

        force_fit = force[data_start:data_end]
        delta_fit = delta[data_start:data_end]
        #trigerから求めた方がいい。
        #FIXME: approachの長さを渡すため。
        #ただ、x,yでtupleで渡す方がいい。
        #tingmodel内で分割はよくない
        delta_fit = np.append(np.argmax(force), delta_fit)

        #FIXME: 現在適当。アプローチの結果から求めた方がいい。
        iv = [600, 0.3, 10] if offset else [400, 0.3]

        fit_model = self.tingmodel_offset if offset else self.tingmodel

        self.popt_tmp = iv
        self.y_tmp = force_fit

        try:
            #FIXME: ここ全体を初期値での最適化にした方がいい。下のelse都かだめだなー
            for i in range(optimize_times):  # あんまりよくないfor
                # TODO: base_lineを変更するようにして、
                # アプローチを急激に高くしないようにする。
                popt, _ = optimize.curve_fit(fit_model, xdata=delta_fit, ydata=force_fit, p0=iv)
                #! ここは、最適か計算の中（tingmodel_offset）ではじめに変更しているので、計算上
                #!　同様の計算をしているのでこの変換がいる。
                e0 = np.abs([0, popt[0]])
                alpha = np.min([self.alpha_upper, np.max([0, popt[1]])])
                if offset:
                    einf = np.max([0, popt[2]])
                    popt = [e0, alpha, einf]
                else:
                    popt = [e0, alpha]

                f_fit = fit_model(np.append(np.argmax(force), delta), *popt)
                #誤差が大きい場合と、二個目の条件の理由忘れた。
                if np.mean((f_fit - force)**2) < res_th_upper and popt[0] > e1_th_lawer:
                    break
                elif i == 0:
                    delta_, force_ = self.change_fit_range(delta, force, idx)
                    data_range = len(force) * np.array(data_ratio)
                    data_start, data_end = int(data_range[0]), int(data_range[1])
                    force_fit = force_[data_start:data_end]
                    delta_fit = delta_[data_start:data_end]
                    delta_fit = np.append(np.argmax(force_fit), delta_fit)
                    #怪しい。
                    self.change_x_data = np.append(self.change_x_data, delta_fit)
                    self.change_y_data = np.append(self.change_y_data, force_fit)
                    self.change_idx = np.append(self.change_idx, self.index)
                else:
                    iv = np.abs(np.random.randn(3)) * np.array(iv)
                    if popt[0] < 10:
                        popt = self.popt_tmp

            f_fit = fit_model(np.append(np.argmax(force), delta), *popt)
            self.residuals.append(np.abs(f_fit - force))
            
            fitting_ting(self.save_path, 
                         delta, force, delta_fit[1:], force_fit, f_fit,
                            popt, np.mean((f_fit - force)**2), self.index)
            if popt[0]>1000:
                self.larger_idx.append(idx)
            self.larger_idx
            self.fitting_result.append(f_fit)
            self.tk.timeshow()
        except IndexError:
            popt=[0,0,0]
        return popt
    # @data_statistics_deco(ds_dict={"data_name":"YoungE"})

    def fit_tingmodel(self, delta, force, offset=False, fit_index="all"):
        """
        tingmodelにoffsetなしの式でfittingする関数。

        Parameters
        ----------
        initial_values : array_like
            e0, alpha の初期値
        delta : np.ndarray
            押し込み量
        force : np.ndarray
            力
        offset : bool, optional
            offset(e_inf)を使用するかどうか, by default False
        fit_index : "all", or arr-like(int), optional
            フィッティングを行うインデックス。
            "all"の場合、すべて、arr-likeの場合、その要素のインデックス
        Returns
        -------
        coefs : np.ndarray
            結果
        """
    
        if fit_index == "all":
            self.index = 0
            self.num_of_data = len(delta)
            self.tk = TimeKeeper(self.num_of_data)
            result = np.array([self.objective_ting(d, f, idx=i, offset=offset)
                              for i, (d, f) in enumerate(zip(delta, force))])
        elif isinstance(fit_index, (list, np.ndarray)):
            self.num_of_data = len(fit_index)
            self.tk = TimeKeeper(self.num_of_data)
            result = np.array([self.objective_ting(delta[idx], force[idx],
                              idx=idx, offset=offset) for idx in fit_index])
        np.savetxt(self.save_name2path("large_idx"),self.larger_idx)
        # self.change_data("delta_preprocessed", self.delta_data,self.change_x_data, self.change_idx, "change delta data for fitting ting model")
        # self.change_data("force_preprocessed", self.force_data,self.change_y_data, self.change_idx, "change force data for fitting ting model")

        self.logger.debug("Finished fitting")
        return result

    def fit(self, 
            delta_app, 
            delta_ret, 
            force_app, 
            force_ret, 
            contact,
            E,
            fit_index="all",
            offset=True,
            ret_ratio=0.3,
            is_ret_contact_img=False,
            force_fitting=False):
        """

        Parameters
        ----------
        delta_app,delta_ret,force_app,force_ret : arr-like
            押し込み量と力のアプローチとリトラクション。
        contact : arr-like
            コンタクトポイント
        fit_index : str("all") or arr-like, optional
            今回、解析に使うインデックス。"all"の場合すべて、arr-likeの場合その中のindexのみを解析する, by default "all"
        offset : bool, optional
            Einfを解析にいれるか, by default True
        ret_ratio : float, optional
            解析に使用するリトラクションの割合, by default 0.3
        is_ret_contact_img : bool, optional
            いらない気がする。, by default False
        """
        start=time.time()
        delta_data = self.isfile_in_data_or_save("delta_preprocessed.npy")
        force_data = self.isfile_in_data_or_save("force_preprocessed.npy")

        if force_fitting or isinstance(delta_data, bool) or isinstance(force_data, bool):
            delta_data, force_data = self.preprocessing(delta_app,delta_ret,force_app,force_ret,contact)

        if isinstance(self.isfile_in_data_or_save("fit_result.npy"), bool) :
            self.logger.debug(f"Start fitting")
            result = self.fit_tingmodel(
                delta_data, force_data, offset=offset, fit_index=fit_index)
            e1 = np.array([self.power_law_rheology_model(
                xi=0, t=1, param=p) for p in result])
            np.save(self.save_name2path("fit_result"), result)
            np.save(self.save_name2path("e1"), e1)
            np.save(self.save_name2path("residuals"), self.residuals)
            np.save(self.save_name2path("fitting_result"), self.fitting_result)
        elif isinstance(fit_index, (list, np.ndarray)):
            result = np.load(self.save_name2path("fit_result.npy"), allaw_pickle=True)
            fitting_result = np.load(self.save_name2path("fitting_result.npy"), allaw_pickle=True)
            residuals = np.load(self.save_name2path("residuals"), allow_pickle=True)
            result_partial = self.fit_tingmodel(
                delta_data, force_data, offset=offset, fit_index=fit_index)
            result[fit_index] = result_partial
            fitting_result[fit_index] = self.fitting_result
            residuals[fit_index] = self.residual
            e1 = np.array([self.power_law_rheology_model(
                xi=0, t=1, param=p) for p in result])
            np.save(self.save_name2path("fit_result_changed"), result)
            np.save(self.save_name2path(
                "fitting_result_changed"), fitting_result)
        else:
            result = self.isfile_in_data_or_save("fit_result.npy")
            e1 = self.isfile_in_data_or_save("e1.npy")
            self.residuals = self.isfile_in_data_or_save("residuals.npy")

        topo_contact = self.isfile_in_data_or_save("topo_contact.npy")
        plot_ting_summury(self.save_path,self.measurament_dict["map_shape"], topo_contact, E, result, e1, self.residuals) 
        end = time.time() - start
        self.logger.debug(f"Finished TIME :{end}")
