from __future__ import annotations
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


from ..utils.fc_helper import *
from ..utils.decorators import data_statistics_deco

from ..fitting._base_analyzer import pathLike, FCBaseProcessor


class FCDataLoader(FCBaseProcessor):
    def __init__(self, 
                 save_path: pathLike,
                 measurament_dict: dict[str,float],
                 afm_param_dict: dict[str, float],
                 data_path: pathLike=None,
                 invols=200,
                 resolution=10,
                 logfile="logfile"):
        super().__init__(save_path, measurament_dict, afm_param_dict,data_path, logfile)
        self.invols = invols
        self.K = self.invols * 1e-9 * self.afm_param_dict["k"]
        self.Hertz = True
        self.resolution = resolution

    def check_numbering(self, 
                        numbering_str_list:list[str], 
                        complement :bool=True,
                        comp_value :int =-100,
                        data_length: int | None = None):
        """リスト内の文字列が順番になっているかの検証
        
        使用することがなかったので、バグあるかも
        
        Parameters
        ----------
        numbering_str_list : list[str]
            数値番号の入っている文字列のリスト
        complement : bool, optional
            Trueの時、欠番に対して、最後のデータ列と同形式でcomp_valueのデータを作成, by default True
        comp_value : int, optional
            欠番データを埋めるint
        data_length: int | None, optional
            データの数、Noneの場合、numbering_str_list内のデータの最大値+1(0始まり)になる
            設定しても、numbering_str_list内のデータの最大値+1の方が大きい場合、そちらが優先, by default None
        
        NOTE
        ----
        comp_valueで埋めるのでなく、両隣の平均をとるとかでもいいかも。
        分かりやすくするために、comp_valueを入れている。
        """
        dir_name = os.path.dirname(numbering_str_list[-1])
        
        if not os.path.isfile(os.path.join(self.save_path, "data_info.txt")):

            with open(os.path.join(self.save_path, "data_info.txt"), "w") as f:
                print(f"Data Length : {len(numbering_str_list)}", file=f)
                ld = int(re.findall("\\d+", os.path.basename(numbering_str_list[-1]))[0])
                print(f"last data   : {ld}", file=f)
                print(f"complement : {complement}", file=f)
                print("==== 欠損データ ====", file=f)
                
            #補完するためのデータ作成。
            if complement:
                sample_data = np.loadtxt(numbering_str_list[-1])
                data_length = int(re.findall("\\d+", os.path.basename(numbering_str_list[-1]))[0])
                comp_data = np.ones(sample_data.shape) * -100
            
            numbering_list = [ int(re.findall("\\d+", os.path.basename(ns))[0]) 
                  for ns in numbering_str_list]
            max_number = np.max(numbering_list)
            data_length = max([data_length-1, max_number])\
                            if data_length else max(max_number)
            if max_number != data_length:
                missing_number :set[int] = set(np.arange(max_number+1))-set(numbering_list)
                if complement:
                    [np.savetxt(os.path.join(
                                                dir_name, 
                                                "ForceCurve_{:>03}.lvm".format(i)
                                            ), comp_data) 
                        for i in missing_number]    
                       
                    missing_number_str_list = [
                                                os.path.join(dir_name, "ForceCurve_{:>03}.lvm".format(i)) 
                                                for i in missing_number
                                            ]       

                with open(os.path.join(self.save_path, "data_info.txt"), "a") as f:
                    print(missing_number,file=f)
            else:
                missing_number=[]
            np.savetxt(os.path.join(self.save_path, "missing_number.txt"), np.array(missing_number))
            numbering_list = np.sort(np.append(numbering_list,missing_number_str_list))
            return numbering_list
        else:
            missing_number = []
            if os.path.isfile(os.path.join(self.save_path, "missing_number.txt")):
                missing_number =  np.loadtxt(os.path.join(self.save_path, "missing_number.txt"))
                
        missing_number_str_list = [
                                    os.path.join(dir_name, "ForceCurve_{:>03}.lvm".format(i)) 
                                    for i in missing_number
                                    ]
        numbering_list = np.sort(np.append(numbering_str_list,missing_number_str_list))
        return numbering_list, missing_number
    
    @staticmethod
    def padding0filenames(filenames:list[pathLike]):
        dir_name = os.path.dirname(filenames[0])
        filenames_0padding = [re.sub(
                                "\d+",
                                "{:>03}".format(re.findall("\d+",os.path.basename(filename))[-1]),
                                os.path.basename(filename)
                                )
                              for filename in filenames
                            ]
        [
            os.rename(
                os.path.join(dir_name,filename),
                os.path.join(dir_name,filename_0padding)
            ) 
            for filename_0padding,filename in zip(filenames_0padding,filenames)
         ]
        
    @staticmethod
    def is_length_same(same_lenght_data, length_strict):
        if same_lenght_data.ndim==1:
            length_data = np.array([len(f) for f in same_lenght_data])
            med_length = np.median(length_data)
            diff_length_idx = np.where(length_data!=med_length)[0]
                
            diff_length_idx_str = ",".join(map(str,diff_length_idx))
            diff_length_str = ",".join(map(str,length_data[diff_length_idx]))
            err_str = f"length of ForceCurve {diff_length_idx_str} ({diff_length_str}) is not same as others ({med_length})"
            if length_strict:
                raise ValueError(err_str)
            else:
                print(err_str)

        else:
            return True
    
    def load_row_fc(self,
            fc_path        :pathLike,
            prefix_lvmfile :str           ="ForceCurve_",
            complement     :bool          =True,
            filename_0padding   : bool    = True,
            map_shape_square_strict:bool  =True,
            length_strict   :bool          =True):
        """
        指定されたパスに含まれるlvmデータをすべてnumpy.ndarrayに入れる関数。

        Parameters
        ----------
        fc_path:pathLike
        prefix_lvmfile

        Parameters
        ----------
        fc_path : pathLike
            lvmファイルが含まれているファイル
        prefix_lvmfile : str, optional
            lvmの番号の前の名前。, by default "ForceCurve_"
        complement : bool, optional
            欠損があった場合補完のデータを入れるかどうか, by default True
        filname_0padding   : bool         = True,
            file名のpaddingをするかどうか。
        map_shape_square_strict : bool, optional
            map_shapeを正方形かどうかの検証, by default True
        length_strict: bool, optional
            各データの長さが等しいかどうかの検証, by default True
            
        Return
        ------
        fc_row_data:numpy.ndarray
            lvmファイルのデータ
        """
        fc_row_data = self.isfile_in_data_or_save("fc_row_data.npy")
        if isinstance(fc_row_data, bool):
            all_fc = glob(os.path.join(fc_path, "ForceCurve", prefix_lvmfile+"*.lvm"))
            all_fc, self.missing_num =self.check_numbering(all_fc, complement)
            if filename_0padding:
                self.padding0filenames(all_fc)
            fc_row_data = np.array([np.loadtxt(fc_path) for fc_path in all_fc])
            if self.measurament_dict["zig"]:
                fc_row_data=self.direct_zig(fc_row_data, len(fc_row_data[0]))
            np.save(os.path.join(self.save_path, "fc_row_data.npy"), fc_row_data)
        else:
            if os.path.isfile(self.save_name2path("missing_num.txt")):
                self.missing_num = np.load(os.path.join(self.save_path,"missing_num.txt"))
            else:
                self.missing_num=[]
        self.length_same=self.is_length_same(fc_row_data, length_strict)
        if len(fc_row_data) != self.measurament_dict["map_shape"][0] * self.measurament_dict["map_shape"][1]:
            am = self.measurament_dict["map_shape"]
            self.measurament_dict["map_shape"] = (int(np.sqrt(len(fc_row_data))), int(np.sqrt(len(fc_row_data))))
            ms = self.measurament_dict["map_shape"]
            print(f"lenght of fc_row_data {len(fc_row_data)} is not same as map_shape {am}=>{ms}")
            if map_shape_square_strict and self.measurament_dict["map_shape"][0]**2 != len(fc_row_data):
                raise ValueError(f"data shape is not square. Data length : {len(fc_row_data)} ")

        return fc_row_data

    def direct_zig(self, data, data_length=1):
        print("direct_zig")
        numbering = np.arange(len(data), dtype=np.int).reshape(self.measurament_dict["map_shape"])
        numbering[1::2] = numbering[1::2, ::-1]
        np.savetxt(self.save_name2path("nubering"), numbering)
        if data_length > 1:
            data = data.reshape(*self.measurament_dict["map_shape"], data_length)
            data[1::2] = data[1::2, ::-1]
            return data.reshape(-1, data_length)

        else:
            data = data.reshape(self.measurament_dict["map_shape"])
            data[1::2] = data[1::2, ::-1]
            return data.reshape(-1, self.measurament_dict["map_shape"][0] * self.measurament_dict["map_shape"][1])

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
        if self.length_same: 
            deflection = fc_row_data[:, :self.measurament_dict["all_length"]]
            zsensor = fc_row_data[:, self.measurament_dict["all_length"]:self.measurament_dict["all_length"] * 2] * 30e-6
        else:
            deflection = np.array([ fc[:int(len(fc)/2)]for fc in fc_row_data],dtype=object)
            zsensor = np.array([ fc[int(len(fc)/2):int(len(fc))]for fc in fc_row_data],dtype=object)* 30e-6
        if fc_img:
            self.im_def_z_row(self.save_path,deflection, zsensor)
        return deflection, zsensor

    def set_deflectionbase(self,baseline_length=300):
        """
        デフレクションの基準値を端の最大値に合わせる
        """
        if self.length_same:
            deflection = self.deflection - np.mean(self.deflection[:, :baseline_length], axis=1).reshape(-1, 1)
        else:
            deflection = np.array([de - np.mean(de[:baseline_length]) for de in self.deflection ])

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
        app_data = data[:, :self.measurament_dict["app_points"]]
        sr_data =  data[:, self.measurament_dict["app_points"]:self.measurament_dict["app_points"] + self.measurament_dict["sr_points"]]
        ret_data = data[:, self.measurament_dict["app_points"] + self.measurament_dict["sr_points"]:]
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
        # topo_trig = np.max(zsensor, axis=1).reshape(self.measurament_dict["map_shape"])
        self.topo_contact = np.array([z[c] for z, c in zip(zsensor, contact)]).reshape(self.measurament_dict["map_shape"])
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
        data_gra = np.array(data).reshape(self.measurament_dict["map_shape"]) / self.cos_map

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
        data[:,:self.measurament_dict["app_length"]],data[:,self.measurament_dict["app_length"]:]:array_like
            アプローチ部分のデータ、リトラクションのデータ
        """
        if self.length_same:
            return data[:, :self.measurament_dict["app_points"]], data[:, self.measurament_dict["app_points"]:]
        else:
            
            app_data = [ d[:int(len(d)/2)] for d in data]
            ret_data = [ d[int(len(d)/2):] for d in data]
            return app_data, ret_data
            
    def load_data(self, fc_path, fc_type="fc",load_row_fc_kargs={}):

        if  not fc_type in ["fc","inv","sr"]:
            raise ValueError("fc_type must be  fc or inv or sr")
        
        map_shape_square_strict = fc_type != "inv" #良くない！
        self.measurament_dict["zig"]=  False if fc_type == "inv" else self.measurament_dict["zig"]

        fc_row_data = self.load_row_fc(fc_path=fc_path, 
                                       map_shape_square_strict=map_shape_square_strict,
                                       complement=True,
                                       **load_row_fc_kargs)
        if isinstance(fc_row_data, bool):
            return False
        self.deflection, self.zsensor = self.split_def_z(fc_row_data)
        self.deflection_row = self.deflection
        del fc_row_data

        self.deflection = self.set_deflectionbase()
        self.delta = self.get_indentaion()
        self.force = self.def2force()
        if fc_type == "fc":
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
        return data, self.missing_num, self.length_same
    def fit(self):
        pass
