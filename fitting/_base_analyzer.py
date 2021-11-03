from __future__ import annotations

import os
import re
from abc import ABCMeta, abstractmethod
from typing import Any, Union

import numpy as np

import logging
import logging.handlers
from logging import getLogger, Formatter, FileHandler, DEBUG, INFO


# config_dict = {'ystep': 20, 'xstep': 20, 'xlength': 3e-06, 'ylength': 3e-06, 'zig': False}
#?
#!

#//

pathLike = Union[str, bytes, os.PathLike, None]

 

    

class FCBaseProcessor(metaclass=ABCMeta):
    def __init__(self,
                 save_path       : pathLike,
                 measurament_dict: dict,
                 afm_param_dict  : dict[str,float],
                 data_path       : pathLike=None,
                 logfile         : str = 'fitlog.log'
                 ) -> None:

        self.measurament_dict:dict = measurament_dict
        self.save_path = save_path
        self.data_path = data_path
        self.is_data_path :bool = self.data_path and os.path.isdir(self.data_path)
        self.afm_param_dict : dict[str,float]= afm_param_dict
        self.logger = self.setup_logger(self.save_name2path(logfile))
    
        
    @abstractmethod
    def fit():
        pass

    def setup_logger(self, 
                     logfile: str,
                     modname: str = __name__) -> logging:
        """
        loggerの設定

        Parameters
        ----------
        logfile : str, optional
            ログファイル名, by default 'force_curve_analysis.log'
        modname : str, optional
            [description], by default __name__

        Returns
        -------
        logger
        """
        logger = getLogger("logger")  # logger名loggerを取得
        logger.setLevel(DEBUG)  # loggerとしてはDEBUGで

        fmt = '%(asctime)s  %(levelname)-8s  %(filename)-20s  %(lineno)-4s : %(message)s'
        # handler2を作成
        handler = FileHandler(filename=logfile)  # handler2はファイル出力
        handler.setLevel(DEBUG)  # handler2はLevel.WARN以上
        handler.setFormatter(Formatter(
            fmt, datefmt='%Y/%m/%d %H:%M:%S'))
        # handler.setFormatter(logging.Formatter(fmt,datefmt='%Y/%m/%d %H:%M:%S'))
        if not logger.handlers:
            logger.addHandler(handler)
        return logger
    
    @staticmethod
    def normalize( data: np.ndarray, 
                  norm_length: int = 10) -> np.ndarray:
        """平均化する関数
        
        0-100, 100-200, ... ごとに平均化する関数
        Parameters
        ----------
        data : np.ndarray
            平均化するデータ
        norm_length : int, optional
            平均化する幅, by default 10

        Returns
        -------
        data_med :np.ndarray
            平均化したデータ
        """
        data_reshape = data[int(len(data) - norm_length * (len(data) // norm_length)):].reshape(-1, norm_length)
        data_med = np.median(data_reshape, axis=1).reshape(1, -1)[0]
        return data_med

    @staticmethod
    def linefit( x : np.ndarray, 
                y : np.ndarray, 
                cp: int = 0, 
                d : int = 1):
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
        d : int or str, optional
            次元, dがeasy_fitの場合はじめの点と後ろの点のみで直線作成 by default 1

        Returns
        -------
        residuals:float
            残差
        [a,b]:list[float, float]
            線形回帰の係数
        """
        if d == "easy_fit":
            a = (y[cp:][-1] - y[cp:][0]) / (x[cp:][-1] - x[cp:][0])
            b = y[cp:][-1] - x[cp:][-1] * a
            coeffs = [a, b]
            if len(y[cp:])>2:
                residuals = np.mean(abs(y[cp:] - (x[cp:] * coeffs[0] + coeffs[1])))
                return [residuals, *coeffs]
            else:
                return coeffs

        else:
            coeffs = np.polyfit(x[cp:], y[cp:], d)
            if d == 1:
                # xのベクトルに0次元目と、xの高次元のベクトルに、coeffかけた方が、
                # 高次元に対応した残差計算ができる。
                residuals = np.mean(abs(y[cp:] - (x[cp:] * coeffs[0] + coeffs[1])))
                return residuals, [*coeffs]
            else:
                return [*coeffs]

    def save_name2path(self, filename: str) -> str:
        """
        save_path 内のパス

        Parameters
        ----------
        filename : str
            保存する時のファイル名

        Returns
        -------
        os.path.join(self.save_path, filename) : str
            保存したパス
        """
        return os.path.join(self.save_path, filename)
    
    def isfile_in_data_or_save(self, file_name: str) -> bool | np.ndarray:
        """
        self.save_pathか、self.data_pathのどちらかにfilenameがあるかの確認。

        Parameters
        ----------
        file_name : str
            ファイルの名前

        Returns
        -------
        issave_path or isdata_path
            存在するかどうか。
        path : str
            self.save_pathか、self.data_path内のfilenameがある場合そのパスを返す。
            ない場合、None
        """
        
        if self.is_data_path and os.path.isfile(os.path.join(self.data_path, file_name)):
            data:np.ndarray = np.load(os.path.join(self.data_path, file_name), allow_pickle=True)
        elif os.path.isfile(os.path.join(self.save_path, file_name)):
            data: np.ndarray = np.load(os.path.join(self.save_path, file_name), allow_pickle=True)
        else:
            data: np.ndarray = False
        return data


