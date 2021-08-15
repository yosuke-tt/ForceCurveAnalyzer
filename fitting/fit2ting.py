
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

from ._base_analyzer import FCBaseProcessor

from ..parameters import *
from ..utils import TimeKeeper

class FitFC2Ting(FCBaseProcessor):
    def __init__(self,iofilePathes, meas_dict, afmParam=AFMParameters(), norm=50):
        # 測定FC時
        super().__init__(meas_dict=meas_dict,
                         iofilePathes=iofilePathes,
                         afmParam=afmParam)
        
        self.norm = norm
        self.Hertz = True

        self.model_param = self.elastic_modelconstant()
        self.err = []

        self.change_x_data, self.change_y_data = [], []
        self.change_idx = []

        self.residuals = []
        self.fitting_result = []
        self.alpha_upper = 0.6

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
            model_param = (4 * self.afmParam.bead_radias**0.5) \
                                / (3 * (1 - self.afmParam.poission_ratio**2))
        else:
            model_param = (2 * self.afmParam.tan_theta) \
                            / (np.pi * (1 - self.afmParam.poission_ratio**2))
        return model_param

    def base2zero(self, delta_app, delta_ret, force_app, force_ret, contact, dim=1, ret_baseline="app", is_plot=True):
        app_base_coeffs = [self.linefit(d[:c], f[:c], cp=0, d=dim)[1]
                           if c > 3 else [0, 0, 0] for d, f, c in zip(delta_app, force_app, contact)]
        if ret_baseline == "contact":
            force_app_base = np.array([fca - fca[int(c)]
                                      for fca, c in zip(force_app, contact)], dtype=object)
            force_ret_base = np.array([fcr - fca[int(c)] for fca, fcr,
                                      c in zip(force_app, force_ret, contact)], dtype=object)
            return force_app_base, force_ret_base
        if ret_baseline == "ret":
            ret_base_coeffs = np.array([self.linefit(d[rc:][::-1], f[rc:][::-1], cp=0, d=dim)[1]
                                        if rc < 19996 else [0, 0, 0] for d, f, rc in zip(delta_ret, force_ret, ret_contact)])
        else:
            ret_base_coeffs = app_base_coeffs

        if dim == 1:
            force_app_base = np.array([fca - (apc[0] * da + apc[1]) for fca, apc,
                                      da in zip(force_app, app_base_coeffs, delta_app)], dtype=object)
            force_ret_base = np.array([fcr - (rc[0] * dr + rc[1]) for fcr, rc,
                                      dr in zip(force_ret, ret_base_coeffs, delta_ret)], dtype=object)
        else:
            force_app_base = np.array([fca - (apc[0] * da**2 + apc[1] * da + apc[2])
                                      for fca, apc, da in zip(force_app, app_base_coeffs, delta_app)], dtype=object)
            force_ret_base = np.array([fcr - (rc[0] * dr**2 + rc[1] * dr + rc[2])
                                      for fcr, rc, dr in zip(force_ret, ret_base_coeffs, delta_ret)], dtype=object)

        if is_plot:
            self.plot_set_base(force_app_base, force_ret_base)
        return force_app_base, force_ret_base

    def determine_data_range(
            self,
            delta_app,
            delta_ret,
            force_app,
            force_ret,
            contact,
            is_processed_img=True,
            is_ret_contact_img=False):
        contact = contact.astype(int)
        ret_contact = [np.where(dr <= da[c])[0][0] if da[c] > np.min(dr) else len(dr) - 1
                       for dr, da, c in zip(delta_ret, delta_app, contact)]
        if is_ret_contact_img:
            self.ret_contact_img()

        force_app_base, force_ret_base = self.base2zero(
            delta_app, delta_ret, force_app, force_ret, contact, dim=1, is_plot=True, ret_baseline="contact")
        force_app_max = [np.max(fab) for fab in force_app_base]
        force_ret_corrected = np.array(
            [np.fmin(fam, frb) for fam, frb in zip(force_app_max, force_ret_base)], dtype=object)

        delta_data = np.array([np.concatenate([da[c:], dr[:rc + 1]]) - da[0]
                              for da, dr, c, rc in zip(delta_app, delta_ret, contact, ret_contact)], dtype=object)
        force_data = np.array([np.concatenate([fa[c:], fr[:rc + 1]]) for fa, fr, c,
                              rc in zip(force_app_base, force_ret_corrected, contact, ret_contact)], dtype=object)

        delta_data = [self.normalize(d, norm_length=self.norm)
                      for d in delta_data]
        force_data = [self.normalize(f, norm_length=self.norm)
                      for f in force_data]

        if is_processed_img:
            self.plot_preprocessed_img()

        np.save(self.ioPathes.save_name2path("delta_preprocessed"), self.delta_data)
        np.save(self.ioPathes.save_name2path("force_preprocessed"), self.force_data)

        return delta_data, force_data

    def power_law_rheology_model(self, xi, t, param):

        if len(param) == 2:
            return self.model_param * (param[0] * (1 + ((t - xi) / self.tdash))**((-1) * param[1]))
        else:
            # return param[2]+(param[0]-param[2])*(1+(t-xi)/self.tdash)**(-1*param[1])
            if (t - xi) != 0.0:
                # Waring対策
                return self.model_param * (param[2] + (param[0] - param[2]) * ((t - xi) / self.tdash)**(-1 * param[1]))
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
        t_ret = np.arange(tm, (len(d_app) + len(d_ret) - 1)
                          * self.tstep, self.tstep)

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
        e0 = np.max([0, e0])
        alpha = np.min([self.alpha_upper, np.max([0, alpha])])
        einf = np.max([0, einf])
        # print(e0, alpha, einf)
        tm = x[0] * self.tstep
        t_app = np.arange(0, tm, self.tstep)
        d_app = x[1:][:int(x[0])]
        d_ret = x[1:][int(x[0]):]

        if len(t_app) < 2:
            return [0]
        else:
            app_dstep = (d_app[-1] - d_app[0]) / (t_app[-1] - t_app[0])

        force_app_fitted = [integrate.quad(lambda xi: self.eq_PLR_integrand(
            xi, t=t, param=[e0, alpha, einf], dstep=app_dstep), 0, t)[0] for t in t_app]

        t_ret = np.arange(tm, (len(d_app) + len(d_ret))
                          * self.tstep, self.tstep)
        ret_dstep = (d_ret[-1] - d_ret[0]) / (t_ret[-1] - t_ret[0])
        t1 = np.zeros(len(t_ret))
        t1s_pro = np.arange(0, tm, 0.00001)

        for i_t1, tr in enumerate(t_ret):
            int_pro = np.zeros(len(t1s_pro))
            ret_integral = integrate.quad(lambda xi: self.eq_PLR_integrand_for_t1(
                xi, t=tr, param=[e0, alpha, einf], dstep=ret_dstep), tm, tr)[0]

            for i, t1p in enumerate(t1s_pro):
                int_pro[i] += np.abs(integrate.quad(lambda xi: self.eq_PLR_integrand_for_t1(xi, t=tr,
                                     param=[e0, alpha, einf], dstep=app_dstep), t1p, tm)[0] + ret_integral)
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
                    param=[
                        e0,
                        alpha,
                        einf],
                    dstep=app_dstep),
                0,
                t)[0] if t != 0 else 0 for i,
            t in enumerate(t1)]

        fit_data = np.concatenate([force_app_fitted, force_ret_fitted])
        f_fit = np.concatenate([force_app_fitted, force_ret_fitted])[
            :len(x[1:])]
        try:
            if np.mean(np.abs(f_fit - self.y_tmp)) < self.res_tmp:
                self.popt_tmp = [e0, alpha, einf]
                self.res_tmp = np.mean(np.abs(f_fit - self.y_tmp))
        except ValueError:
            pass
        return f_fit

    def change_fit_range(self, x, y, idx, start_fit=20, end_fit=40):
        y_app = y[:np.argmax(y)]
        x_app = x[:np.argmax(y)]
        y_app_length = len(y_app)

        lfit = np.vstack([self.linefit(x_app, y_app, cp=i, d="easy_fit")
                         for i in range(len(y_app) - end_fit, len(y_app) - start_fit)])
        coeffs = lfit[:, 1:]
        coeffs_m = np.median(coeffs, axis=0)

        base_mid = (y[-1] - y[0]) / 2
        x_new_start = (base_mid - coeffs_m[1]) / coeffs_m[0]
        x_new = x[x > x_new_start]
        y_new_ = y[x > x_new_start]
        y_new = y_new_ - np.abs(base_mid)

        os.makedirs(self.ioPathes.save_name2path("processed_img_for_ting"), exist_ok=True)
        plt.plot(x, y, label="original")
        plt.plot(x_new, y_new, label="new")
        plt.plot(x_new, y_new, label="new")
        plt.hlines([base_mid], xmin=np.min(x), xmax=np.max(
            x), label="base_mid", color="red")
        plt.hlines([0], xmin=np.min(x), xmax=np.max(x), label="0")
        plt.legend()
        plt.savefig(self.ioPathes.save_name2path("processed_img_forting/{:>03}".format(idx)))
        plt.close()
        return x_new, y_new

    def change_data(self, file_path, all_data, change_data, idx, info):

        with open(self.ioPathes.save_name2path("change_data_info.txt"), "a") as f:
            print(time.strftime("%Y%m%d %H:%M:%S"), file=f)
            print(file_path, file=f)
            print(idx, file=f)
            print(info, file=f)

        pre_data_path = os.path.basename(file_path).split(".")[0] + "_pre.npy"

        data_old = self.ioPathes.isfile_in_data_or_save(pre_data_path)

        if isinstance(data_old, bool):
            data_old = np.array([[idx, change_data]])
        else:
            data_old = np.vstack([data_old, [idx, change_data]])
        np.save(self.ioPathes.save_name2path(pre_data_path), data_old)
        all_data[idx] = change_data
        np.save(self.ioPathes.save_name2path(file_path), all_data)

    def objective_ting(
            self,
            x,
            y,
            idx=0,
            offset=False,
            data_ratio=[0, 0.6],
            optimize_times=3,
            res_th_upper=1e-19,
            e1_th_lower=10):
        """
        tingmodelフィッティングの際に最適化する関数。

        Parameters
        ----------
        iv : array_like
            初期値
        x : np.ndarray
            押し込み量
        y : np.ndarray
            力
        offset : bool, optional
            offsetを使用するかどうか, by default False

        Returns
        -------
        popt
            最適解[e0, alpha, e1,]
        """
        self.y_tmp_row = y
        self.res_tmp = np.inf
        self.index = idx
        start_fit = time.time()
        data_range = len(y) * np.array(data_ratio)
        data_start, data_end = int(data_range[0]), int(data_range[1])

        y_ = y[data_start:data_end]
        x_data = x[data_start:data_end]
        x_ = np.append(np.argmax(y), x_data)
        iv = [600, 0.3, 10] if offset else [400, 0.3]
        fit_model = self.tingmodel_offset if offset else self.tingmodel
        self.popt_tmp = iv
        self.y_tmp = y_
        for i in range(optimize_times):  # あんまりよくないfor
            popt, pcov = optimize.curve_fit(
                fit_model, xdata=x_, ydata=y_, p0=iv)
            e0 = np.max([0, popt[0]])
            alpha = np.min([self.alpha_upper, np.max([0, popt[1]])])

            if offset:
                einf = np.max([0, popt[2]])
                popt = [e0, alpha, einf]
            else:
                popt = [e0, alpha]

            f_fit = fit_model(np.append(np.argmax(y), x), *popt)

            if np.mean((f_fit - y)**2) < res_th_upper and popt[0] > e1_th_lower:
                break
            elif i == 0:
                x, y = self.change_fit_range(x, y, idx)
                data_range = len(y) * np.array(data_ratio)
                data_start, data_end = int(data_range[0]), int(data_range[1])
                y_ = y[data_start:data_end]
                x_data = x[data_start:data_end]
                x_ = np.append(np.argmax(y), x_data)
                self.change_x_data = np.append(self.change_x_data, x)
                self.change_y_data = np.append(self.change_y_data, y)
                self.change_idx = np.append(self.change_idx, self.index)
            else:
                iv = np.abs(np.random.randn(3)) * np.array(iv)
                if popt[0] < 10:
                    popt = self.popt_tmp

        f_fit = fit_model(np.append(np.argmax(y), x), *popt)
        self.residuals.append(np.abs(f_fit - y))
        self.fitting_ting(x, y, x_[1:], y_, f_fit,
                          popt, np.mean((f_fit - y)**2), self.index)
        self.fitting_result.append(f_fit)
        self.tk.timeshow()
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

        # self.change_data("delta_preprocessed", self.delta_data,self.change_x_data, self.change_idx, "change delta data for fitting ting model")
        # self.change_data("force_preprocessed", self.force_data,self.change_y_data, self.change_idx, "change force data for fitting ting model")

        self.logger.debug("Finished fitting")
        return result

    def fit(self, delta_app, delta_ret, force_app, force_ret, contact,
            fit_index="all", offset=True, ret_ratio=0.3, is_ret_contact_img=False):
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
        start = time.time()

        delta_data = self.ioPathes.isfile_in_data_or_save("delta_preprocessed.npy")
        force_data = self.ioPathes.isfile_in_data_or_save("force_preprocessed.npy")

        if True or (not isinstance(delta_data, np.ndarray) or not isinstance(force_data, np.ndarray)):
            delta_data, force_data = self.determine_data_range(
                delta_app, delta_ret, force_app, force_ret, contact,)

        if not self.ioPathes.isfile_in_data_or_save("fit_result.npy")):
            self.logger.debug(f"Start fitting")
            result = self.fit_tingmodel(
                delta_data, force_data, offset=offset, fit_index=fit_index)
            e1 = np.array([self.power_law_rheology_model(
                xi=0, t=1, param=p) for p in result])
            np.save(os.path.join(self.save_path, "fit_result"), result)
            np.save(self.ioPathes.save_name2path("e1"), e1)
            np.save(self.ioPathes.save_name2path("residuals"), self.residuals)
            np.save(self.ioPathes.save_name2path(
                "fitting_result"), self.fitting_result)
        elif isinstance(fit_index, (list, np.ndarray)):
            result = np.load(self.ioPathes.save_name2path("fit_result.npy"), allow_pickle=True)
            fitting_result = np.load(self.ioPathes.save_name2path("fitting_result.npy"), allow_pickle=True)
            residuals = np.load(self.ioPathes.save_name2path("residuals"), allow_pickle=True)
            result_partial = self.fit_tingmodel(
                delta_data, force_data, offset=offset, fit_index=fit_index)
            result[fit_index] = result_partial
            fitting_result[fit_index] = self.fitting_result
            residuals[fit_index] = self.residual
            e1 = np.array([self.power_law_rheology_model(
                xi=0, t=1, param=p) for p in result])
            np.save(self.ioPathes.save_name2path("fit_result_changed"), result)
            np.save(self.ioPathes.save_name2path(
                "fitting_result_changed"), fitting_result)
        else:
            result = self.ioPathes.isfile_in_data_or_save("fit_result.npy")
            e1 = self.ioPathes.isfile_in_data_or_save("e1.npy")
            self.residuals = self.ioPathes.isfile_in_data_or_save("residuals.npy")

        topo_contact = self.ioPathes.isfile_in_data_or_save("topo_contact.npy")
        self.plot_ting_summury(topo_contact, E, result, e1, self.residuals)
        end = time.time() - start
        self.logger.debug(f"Finished TIME :{end}")
