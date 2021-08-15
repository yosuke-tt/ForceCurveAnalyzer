from __future__ import annotations

import os
import re
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

import logging
import logging.handlers
from logging import getLogger, Formatter, FileHandler, DEBUG, INFO

from ..parameters.afmparam import AFMParameters
from ..parameters.iofilepath import IOFilePathes
from ..parameters.measurament import MeasurantParameters
# config_dict = {'ystep': 20, 'xstep': 20, 'xlength': 3e-06, 'ylength': 3e-06, 'zig': False}
#?
#!
#//



class FCBaseProcessor(metaclass=ABCMeta):
    def __init__(self,
                 meas_dict: dict,
                 iofilePathes: IOFilePathes,
                 afmParam: AFMParameters = AFMParameters(),
                 logfile: str = 'fitlog.log') -> None:

        self.meas_dict:dict = meas_dict
        self.ioPathes : IOFilePathes = iofilePathes
        self.afmParam : AFMParameters= afmParam
        
        self.logger = self.setup_logger(self.ioPathes.save_name2path(logfile))

        
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

    def normalize(self, 
                  data: np.ndarray, 
                  norm_length: int = 10) -> np.ndarray:
        data_reshape = data[int(len(data) - norm_length * (len(data) // norm_length)):].reshape(-1, norm_length)
        data_med = np.median(data_reshape, axis=1).reshape(1, -1)[0]
        return data_med

    def linefit(self, 
                x : np.ndarray, 
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
        d : int, optional
            次元, by default 1

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
            residuals = np.mean(abs(y[cp:] - (x[cp:] * coeffs[0] + coeffs[1])))
            return [residuals, *coeffs]

        else:
            coeffs = np.polyfit(x[cp:], y[cp:], d)
            if d == 1:
                residuals = np.mean(
                    abs(y[cp:] - (x[cp:] * coeffs[0] + coeffs[1])))
                return residuals, [*coeffs]
            else:
                return [*coeffs]



