import os

from datetime import datetime, timedelta
import time

import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#　文字設定
plt.rcParams["font.family"] = "sans-serif"

from .approach import FCApproachAnalyzer
from ._base_analyzer import FCBaseProcessor

from ..utils.fc_helper import *
from ..utils.decorators import data_statistics_deco


import japanize_matplotlib

import warnings
# import tqdm






class StressRelaxationPreprocessor(FCBaseProcessor):
    def __init__(self, config_dict: dict, save_path: str = "./data", data_path: str = None, invols: float = 200):
        super().__init__(save_path, data_path, config_dict)
        warnings.simplefilter('ignore')

        self.zig = zig
        self.map_shape = map_shape
        self.app_length = length_data[0]
        self.sr_length = 50000
        self.ret_length = length_data[0]
        self.all_length = self.ret_length + self.sr_length + self.app_length

    def sep_srdata(self, data):
        """
        データをアプローチ、応力緩和、リトラクションのデータに分割する関数。
        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        app_data, sr_data, ret_data
            押し込みのデータ、応力緩和のデータ、リトラクションのデータ
        """
        app_data = data[:, :self.app_length]
        sr_data = data[:, self.app_length:self.app_length + self.sr_length]
        ret_data = data[:, self.app_length + self.sr_length:]
        return app_data, sr_data, ret_data

    def sep_srdata(self, data):
        """
        データをアプローチ、応力緩和、リトラクションのデータに分割する関数。
        Parameters
        ----------
        data : arr_like
            分割するデータ
        Returns
        -------
        app_data, sr_data, ret_data:arr_like
            アプローチ、応力緩和、リトラクションのデータ
        """
        app_data = data[:, :self.app_length]
        sr_data = data[:, self.app_length:self.app_length + self.sr_length]
        ret_data = data[:, self.app_length + self.sr_length:]
        return app_data, sr_data, ret_data

    def sep_srdatas(self):
        """
        deflection, 押し込み量, zsensor, forceを応力緩和、リトラクションのデータに分割する関数。
        """
        self.def_app, self.def_sr, self.def_ret = self.sep_srdata(self.deflection)
        self.delta_app, self.delta_sr, self.delta_ret = self.sep_srdata(self.delta)
        self.z_app, self.z_sr, self.z_ret = self.sep_srdata(self.zsensor)
        self.force_app, self.force_sr, self.force_ret = self.sep_srdata(self.force)

    def set_cp_delta_force(self, cp):
        """
        コンタクトポイントからのデータに0殻設定したもの

        Parameters
        ----------
        cp :
            コンタクトポイント
        Note
        self.delta~に全部設定される。
        """
        delta_cp = np.array([d[cp] for d, cp in zip(self.delta_app, cp)]).reshape(-1, 1)
        force_cp = np.array([f[cp] for f, cp in zip(self.force_app, cp)]).reshape(-1, 1)
        self.delta_app -= delta_cp
        self.force_app -= force_cp
        self.delta_app = np.array([d[cp:] for d, cp in zip(self.delta_app, cp)], dtype=object)
        self.force_app = np.array([f[cp:] for f, cp in zip(self.force_app, cp)], dtype=object)
        self.delta_sr -= delta_cp
        self.delta_ret -= delta_cp
        self.force_sr -= force_cp
        self.force_ret -= force_cp

    def hertz_Et(self):
        """
        Hertzモデルから算出したE(t)
        """
        p = (4 / 3) * (self.R**(1 / 2)) / (1 - self.v**2)
        Et = self.force_sr / (p * self.delta_sr**(3 / 2))
        return Et

    def power_low(self, x, e_inf, e0, alpha):
        """
        べき乗理論
        Parameters
        ----------
        t : arr_like
            時間
        e_inf : float
            Einf
        e0 : float
            E0
        alpha : float
            alpha

        Returns
        -------
        e_inf + (e0-e_inf)*(t**(-1*alpha))
            べき乗理論
        """
        # return e_inf + (e0-e_inf)*((x/self.dt+1)**(-1*alpha))
        return e_inf + (e0 - e_inf) * ((x)**(-1 * alpha))

    def power_low_e0fixed(self, x, e_inf, alpha):
        """
        べき乗理論
        Parameters
        ----------
        t : arr_like
            時間
        e_inf : float
            Einf
        e0 : float
            E0
        alpha : float
            alpha

        Returns
        -------
        e_inf + (e0-e_inf)*(t**(-1*alpha))
            べき乗理論
        """
        return e_inf + (self.e0_ - e_inf) * ((x / self.dt + 1)**(-1 * alpha))

    def residuals(self, y, y_pred, type="square"):
        """
        残差計算の関数

        Parameters
        ----------
        y : np.ndarry
            実データ
        y_pred : np.ndarry
            予測計算したデータ
        type : str, optional
            誤差計算の方法, by default "square"
            "square"    : 二乗
            "abs"       : 絶対値
            "dy_square" : 誤差率の二乗
            "dy_abs"    : 誤差率の絶対値
        Returns
        -------
        [type]
            [description]
        """
        if type == "square":
            res = np.mean((y - y_pred)**2)
        elif type == "abs":
            res = np.mean(np.abs(y - y_pred))
        elif type == "dy_square":
            res = np.mean(((y - y_pred) / (np.abs(y) + 1e-16))**2)
        elif type == "dy_abs":
            res = np.mean(np.abs((y - y_pred) / (np.abs(y) + 1e-16)))
        return res

    @data_statistics_deco(ds_dict={"data_name": ("e_inf_res", "e0_res",
                          "alpha_res", "res", "e_fit"), "skip_idx": [3, 4]})
    def fit_power_low(
            self,
            sr_time=1,
            sr_time_offset=0,
            data_range=(
                200,
                20000),
            e0_fixed=False,
            verbose=10,
            is_trial_img=False,
            is_fitting_img=False):
        """
        べき乗理論にフィッティングする関数

        Parameters
        ----------
        sr_time : int, optional
            応力緩和の時間, by default 1
        data_range : int, optional
            データの解析開始点数, by default 0
        n_trials : int, optional
            最適化の回数, by default 1000
        res_type : string, optional
            残差の種類, by default "square"
        e0_fixed : bool, optional
            e0をデータ点数のはじめにするかどうか。by default False
        verbose : int,
            時間表示のデータ, by default 10
        is_trial_img : bool, optional
            残差の試行回数ごとのプロット表示, by default False
        is_fitting_img : bool, optional
            フィッティング画像表示, by default False
        values_summary : bool, optional
            残差のグラフなど表示するか。by default False

        et_path : string, optional
            E(t)の保存先のパス。by default ""

        Returns
        -------
        e_inf_res, e0_res, alpha_res, res:
            E_inf, E0, alpha, residuals,e_fit
        """
        self.num_data = len(self.Et)

        e_fit = np.zeros((self.num_data, self.sr_length))
        res = np.zeros(self.num_data)
        alpha_res = np.zeros(self.num_data)
        e_inf_res = np.zeros(self.num_data)
        e0_res = np.zeros(self.num_data)
        self.dt = 1 / self.sr_length
        self.sr_time = np.arange(self.dt, 1 + self.dt, self.dt)[data_range[0]:data_range[1]]
        self.sr_time_all = np.arange(0 + sr_time_offset, 1 + sr_time_offset, self.dt)
        tk = TimeKeeper(self.num_data)

        if os.path.isfile(os.path.join(self.save_path, "complement_num.txt")):
            complement_num = np.loadtxt(os.path.join(self.save_path, "complement_num.txt"))
        else:
            complement_num = []
        for i, et in enumerate(self.Et):

            if not(i in complement_num):
                et_row = et
                et = et[data_range[0]:data_range[1]]
                if not e0_fixed:
                    param_bounds = ((0.0, 0.0, 0.0), (800, 3000 * (self.invols / 200), 1))
                    try:
                        popt, pcov = curve_fit(self.power_low, self.sr_time, et, bounds=param_bounds)
                    except ValueError:
                        with open(os.path.join(self.save_path, "valueErrorInsrFit.txt"), "a") as f:
                            print("\n\n{}".format(datetime.now()), file=f)
                            print(i, file=f)
                        popt = [0, 0, 0]
                    e_inf = popt[0]
                    e0 = popt[1]
                    alpha = popt[2]
                    y_pred = self.power_low(self.sr_time_all, e_inf, e0, alpha)
                    res_ = np.mean((et_row[y_pred < 1000000] - y_pred[y_pred < 1000000])**2)

                else:
                    param_bounds = ((0.0, 0.0), (800, 1))

                    self.e0_ = et_row[0]
                    popt, pcov = curve_fit(self.power_low_e0fixed, self.sr_time, et, bounds=param_bounds)
                    e0 = et[0]
                    e_inf = popt[0]
                    alpha = popt[1]
                    y_pred = self.power_low_e0fixed(self.sr_time_all, e_inf, alpha)

                    res_ = np.mean((et_row - y_pred)**2)
            else:
                res_ = 0
                e_inf = 0
                y_pred = np.zeros(50000)
                alpha = 0

            res[i] += res_
            e_inf_res[i] += e_inf
            e0_res[i] += y_pred[0]
            alpha_res[i] += alpha
            e_fit[i] += y_pred
            all_time = tk.timeshow(i)

            if is_fitting_img:
                os.makedirs(os.path.join(self.save_path, "fit_sr"), exist_ok=True)
                self.sr_fitting_img(self.sr_time_all, y_pred, et_row, e0, e_inf, alpha, res_, i)

        with open(self.savefile2savepath("fit_time"), "w") as f:
            print(all_time, file=f)
        return e_inf_res, e0_res, alpha_res, res, e_fit

    def fit_sr(self, fc_path, sr_fit_dict={}, complement=False):
        # FIXME: 変える 
        fsrfile = self.isfile_in_data_or_save("force_sr.npy")
        dsrfile = self.isfile_in_data_or_save("delta_sr.npy")
        if not fsrfile:
            start = time.time()
            self.logger.info("load data")

            fc_row_data = self.load_row_fc(fc_path=fc_path, complement=complement)
            if isinstance(fc_row_data, bool):
                return False
            self.deflection, self.zsensor = self.split_def_z(fc_row_data)
            del fc_row_data

            self.deflection = self.set_deflectionbase()
            self.delta = self.get_indentaion()

            self.force = self.def2force()

            self.sep_srdatas()
            fcp = FCApproachAnalyzer(save_path="./FitFCApproach.py", map_shape=(20, 20))

            self.E, self.line_fitted_data, self.cross_cp, self.topo_contact, self.cos_map = fcp.fit(
                self.delta_app, self.force_app)

            self.logger.info("fit approach")
            # np.save(self.savefile2savepath("grad_topo"),Habs)

            E_grad = self.gradient_adjasment(self.E)
            self.data_statistics(E_grad, "E_grad", stat_type=["map", "map_only", "hist", "plot"])
            self.set_cp_delta_force(self.contact)

            np.save(os.path.join(self.save_path, "force_sr"), self.force_sr)
            np.save(os.path.join(self.save_path, "delta_sr"), self.delta_sr)

        self.Et = self.isfile_in_data_or_save("et.npy")
        # self.hertz_Et()
        # np.save(os.path.join(self.save_path,"et"),self.Et)
        if not self.Et:
            self.Et = self.hertz_Et()
            np.save(os.path.join(self.save_path, "et"), self.Et)
        self.logger.info("start fitting")

        e_inf_res, e0_res, alpha_res, res, e_fit = self.fit_power_low(sr_fit_dict)
        self.fit_sr_all(self.sr_time_all, self.Et, e_fit, e0_res)

        e0_res_grad = self.gradient_adjasment(e0_res)

        self.data_statistics(e_fit[:, -1], "E1", stat_type=["map", "map_only", "hist", "plot"])
        self.data_statistics(np.log10(e_fit[:, -1]), "log_E1", stat_type=["map", "map_only", "hist", "plot"])

        ef2g = self.gradient_adjasment(e_fit[:, -1])
        self.data_statistics(ef2g, "E1_grad", stat_type=["map", "map_only", "hist", "plot"])
        self.data_statistics(np.log10(ef2g), "log_E1_grad", stat_type=["map", "map_only", "hist", "plot"])

        end = time.time() - start
        with open(os.path.join(self.save_path, "time.txt"), "w") as f:
            hour = int(end // 3600)
            minute = int((end - hour * 3600) // 60)
            second = (end - hour * 3600 - minute * 60) / 60
            self.logger.info("{:>3}h {:>3}m {:>3}s".format(hour, minute, second))
        return end / len(e0_res_grad)


if __name__ == "__main__":
    data_range = [(0, 50000), (500, 20000), (500, 50000)]
    sr_time_offset = [0.0002, 0.0004, 0.002, 0.004]
    sr_time_offset = [0.0006, 0.0008, 0.001, 0.0012]

    e0_fixed = [False, True]
    # data_path=f"../tsuboyama_0322/tsuboyama_0322/10/"
    save_path = "../応力緩和データ横堀/"
    save_path = "20210706_sr"
    srp = StressRelaxationPreprocessor(
        save_path=save_path, map_shape=(
            20, 20), zig=False, length_data=(
            12000, 12000), invols=305)
    srp.fit_sr(fc_path="../data_20210517/data_150818", sr_fit_dict={"is_fitting_img": True}, complement=True)
