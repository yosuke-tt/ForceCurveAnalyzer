import os
import sys
import warnings
warnings.simplefilter('ignore', UserWarning)


from glob import glob
from datetime import datetime, timedelta
import time
import re

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


from ..utils.fc_helper import *
from ..utils.decorators import data_statistics_deco
from ..parameters._afmparam import AFMParameters
from ..parameters._measurament import MeasuramentParameters
from ..parameters._iofilepath import IOFilePathes


class FCDataLoader():
    def __init__(self, 
                 meas_dict: MeasuramentParameters,
                 iofilePathes: IOFilePathes,
                 afm_parameters: AFMParameters=AFMParameters(),
                 invols=200,
                 resolution=10):

        self.meas_dict :dict= meas_dict
        self.filePathes: FilePathes = iofilePathes
        self.afm_prameters: AFMParameters = afm_parameters

        self.invols = invols
        self.K = self.invols * 1e-9 * self.afm_prameters.k
        self.Hertz = True

        self.resolution = resolution

    def check_numbering(self, data, complement=True):
        dd = 0
        if not os.path.isfile(os.path.join(self.filePathes.save_path, "data_info.txt")):
            with open(os.path.join(self.filePathes.save_path, "data_info.txt"), "w") as f:
                print(f"Data Length : {len(data)}", file=f)
                ld = int(re.findall("\\d+", os.path.basename(data[-1]))[0])
                print(f"last data   : {ld}", file=f)
                print(f"complement : {complement}", file=f)
                print("==== 欠損データ ====", file=f)
            if complement:
                sample_data = np.loadtxt(data[-1])
                data_length = re.findall("\\d+", os.path.basename(data[-1]))[0]
                comp_data = np.ones(sample_data.shape) * -100
                dir_name = os.path.dirname(data[-1])
            complement_num = []
            complement_shape = np.array([])
            for i, df in enumerate(data):
                df = int(re.findall("\\d+", os.path.basename(df))[0])
                if i + dd != int(df):
                    with open(os.path.join(self.filePathes.save_path, "data_info"), "a") as f:
                        print(f"{i}", file=f)
                    if complement:
                        np.savetxt(os.path.join(dir_name, "ForceCurve_{:>03}.lvm".format(i)), comp_data)
                    dd += 1
                    complement_num.append(i)
                    complement_shape = np.append(complement_shape, [True])
                else:
                    complement_shape = np.append(complement_shape, [False])
            np.savetxt(os.path.join(self.filePathes.save_path, "complement_num.txt"), complement_num)
            np.savetxt(os.path.join(self.filePathes.save_path, "complement_shape.txt"), complement_shape)
            self.complement_num = complement_num
        else:
            self.complement_num = np.loadtxt(os.path.join(self.filePathes.save_path, "complement_num.txt"))

    def load_row_fc(
            self,
            fc_path,
            prefix_lvmfile="ForceCurve_",
            map_shape=None,
            complement=True,
            allow_any_shape=False):
        """
        指定されたパスに含まれるlvmデータをすべてnumpy.ndarrayに入れる関数。

        Parameters
        ----------
        fc_path:str
            lvmファイルが含まれているファイル
        prefix_lvmfile
            lvmの番号の前の名前。番号の範囲を指定したい場合に固定。
        Return
        ------
        fc_row_data:numpy.ndarray
            lvmファイルのデータ
        """
        fc_row_data = self.filePathes.isfile_in_data_or_save("fc_row_data.npy")
        if isinstance(fc_row_data, bool):
            print("save data")
            all_fc = glob(os.path.join(fc_path, "ForceCurve", "*.lvm"))
            self.check_numbering(all_fc, complement)
            all_fc = glob(os.path.join(fc_path, "ForceCurve", "*.lvm"))

            fc_row_data = np.array([np.loadtxt(fc_path) for fc_path in all_fc])
            # メモリの関係でエラーが出るため、400は適当。
            np.save(os.path.join(self.filePathes.save_path, "fc_row_data.npy"), fc_row_data)
        else:
            self.complement_num = np.loadtxt(os.path.join(self.filePathes.save_path, "complement_num.txt"), dtype=object)
            if len(self.complement_num) > 0:
                self.complement_num = self.complement_num
        olength = int(np.median([len(f) for f in fc_row_data[::50]]))
        for i, f in enumerate(fc_row_data):
            assert len(f) == olength, f"length of ForceCurve {i} ({len(f)}) is not same as others ({olength})"

        assert isinstance(self.meas_dict["map_shape"], tuple), "map_shape is boolean"

        if len(fc_row_data) != self.meas_dict["map_shape"][0] * self.meas_dict["map_shape"][1]:
            am = self.meas_dict["map_shape"]
            self.meas_dict["map_shape"] = (int(np.sqrt(len(fc_row_data))), int(np.sqrt(len(fc_row_data))))
            ms = self.meas_dict["map_shape"]
            print(f"lenght of fc_row_data {len(fc_row_data)} is not same as map_shape {am}=>{ms}")

            allow_any_shape = True
            if self.meas_dict["map_shape"][0]**2 != len(fc_row_data) or not allow_any_shape:
                print("data shape is not square")
        return fc_row_data

    def direct_zig(self, data, data_length=1):
        numbering = np.arange(len(data), dtype=np.int).reshape(self.meas_dict["map_shape"])
        numbering[1::2] = numbering[1::2, ::-1]
        np.savetxt(self.savefile2savepath("nubering"), numbering)
        if data_length > 1:
            data = data.reshape(*self.meas_dict["map_shape"], data_length)
            data[1::2] = data[1::2, ::-1]
            return data.reshape(-1, data_length)

        else:
            data = data.reshape(self.meas_dict["map_shape"])
            data[1::2] = data[1::2, ::-1]
            return data.reshape(-1, self.meas_dict["map_shape"][0] * self.meas_dict["map_shape"][1])

    def split_def_z(self, fc_row_data, fc_img=False):
        """
        deflectionと、カンチレバーのZsensorのデータに分ける関数。
        Parameters
        ----------
        fc_row_data : numpy.ndarray
            lvmデータの配列
        Returns
        -------
        deflection, zsensor: np.ndarray
            デフレクション, zセンサー
        """
        deflection = fc_row_data[:, :self.meas_dict["all_length"]]
        zsensor = fc_row_data[:, self.meas_dict["all_length"]:self.meas_dict["all_length"] * 2] * 30e-6
        if fc_img:
            self.im_def_z_row(self.ioPathes.save_path,deflection, zsensor)
        return deflection, zsensor

    def set_deflectionbase(self):
        """
        デフレクションの基準値を端の最大値に合わせる
        """
        deflection = self.deflection - np.mean(self.deflection[:, :300], axis=1).reshape(-1, 1)
        return deflection

    def get_indentaion(self):
        """
        押し込み量を取得する関数。
        """
        delta = self.zsensor - self.deflection * self.invols * 1e-9
        return delta

    def def2force(self):
        """
        deflectionから力を求める関数。
        """
        force = self.deflection * self.K

        return force

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
        app_data = data[:, :self.meas_dict["app_points"]]
        sr_data =  data[:, self.meas_dict["app_points"]:self.meas_dict["app_points"] + self.meas_dict["sr_points"]]
        ret_data = data[:, self.meas_dict["app_points"] + self.meas_dict["sr_points"]:]
        return app_data, sr_data, ret_data

    def sep_srdatas(self):
        """
        deflection, 押し込み量, zsensor, forceを応力緩和、リトラクションのデータに分割する関数。
        """
        self.def_app, self.def_sr, self.def_ret       = self.sep_srdata(self.deflection)
        self.delta_app, self.delta_sr, self.delta_ret = self.sep_srdata(self.delta)
        self.z_app, self.z_sr, self.z_ret             = self.sep_srdata(self.zsensor)
        self.force_app, self.force_sr, self.force_ret = self.sep_srdata(self.force)

    @data_statistics_deco(ds_dict={"data_name": "topo_contact", "vmin": 0})
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
        # topo_trig = np.max(zsensor, axis=1).reshape(self.meas_dict["map_shape"])
        self.topo_contact = np.array([z[c] for z, c in zip(zsensor, contact)]).reshape(self.meas_dict["map_shape"])
        return self.topo_contact

    def outlier_value(self, values):
        """
        2σで外れ値を検出する関数。

        Parameters
        ----------
        values : arrlile
            データ
        Returns
        -------
        outlier_min, outlier_max:
            2sigmaの範囲
        """
        sd = np.nanstd(values)
        # 外れ値の基準点
        average = np.nanmean(values)
        outlier_min = average - (sd) * 2
        outlier_max = average + (sd) * 2
        return outlier_min, outlier_max

    def gradient_adjasment(self, data):
        """
        データを傾斜補正する。

        Parameters
        ----------
        data : array_like
            傾斜補正するデータ

        Returns
        -------
        data_ga:array_like
            傾斜補正したデータ
        """
        data_gra = np.array(data).reshape(self.meas_dict["map_shape"]) / self.cos_map

        return data_gra

    def split_app_ret(self, data):
        """
        データをアプローチ部分のデータと

        Parameters
        ----------
        data : array_like
            分割するデータ

        Returns
        -------
        data[:,:self.length_of_app],data[:,self.length_of_app:]:array_like
            アプローチ部分のデータ、リトラクションのデータ
        """
        return data[:, :self.meas_dict["app_points"]], data[:, self.meas_dict["app_points"]:]

    def load_data(self, fc_path, fc_type="fc", length=(12000, 12000)):
        fc_row_data = self.load_row_fc(fc_path=fc_path, complement=True)
        if isinstance(fc_row_data, bool):
            return False
        self.deflection, self.zsensor = self.split_def_z(fc_row_data)
        self.deflection_row = self.deflection
        del fc_row_data

        self.deflection = self.set_deflectionbase()
        self.delta = self.get_indentaion()
        self.force = self.def2force()
        if fc_type == "fc":
            self.length_of_app = length[0]
            delta_app, delta_ret = self.split_app_ret(self.delta)
            force_app, force_ret = self.split_app_ret(self.force)

            data = ((delta_app, delta_ret), (force_app, force_ret), self.zsensor)
        elif fc_type == "inv":
            def_app, def_ret = self.split_app_ret(self.deflection)
            z_app, z_ret = self.split_app_ret(self.zsensor)
            data = ((def_app, def_ret), (z_app, z_ret))
        elif fc_type == "sr":
            # 応力緩和用
            # def_app, def_sr, def_ret       = self.sep_srdata(self.deflection)
            # z_app, z_sr, z_ret             = self.sep_srdata(self.zsensor)
            delta_app, delta_sr, delta_ret = self.sep_srdata(self.delta)
            force_app, force_sr, force_ret = self.sep_srdata(self.force)
            data = ((delta_app, delta_sr, delta_ret), (force_app, force_sr, force_ret), self.zsensor)
        else:
            pass
        return data, self.complement_num
    def fit(self):
        pass
