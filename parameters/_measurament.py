from __future__ import annotations
import os
import re

from ..utils.typing import pathLike

class MeasuramentParameters:
    config_dict_default: dict = {'zig': False, 
                                 'ret_points': 12000, 
                                 'app_points': 12000,
                                 'xlength': 3.0, 
                                 'ylength': 3.0,
                                 'xstep': 20,
                                 'ystep': 20,
                                 'xstep count': 0, 
                                 'ystep count': 0,
                                 'map_shape' : [20,20],
                                 "all_length":24000 
                                 }

    def __init__(self, config_overwite: dict = {}) -> None:
        self.config_overwite = config_overwite

    def str2goodtype(self, key: str, value: str) -> float | bool | list:
        if "時間" in key:
            time: int = int(value)
            h: int = time // 10000
            m: int = (time - h * 10000) // 100
            s: int = time - h * 10000 - m * 100
            time_list: list = [h, m, s]
            return time_list

        if value.upper() in ["TRUE", "FALSE"]:
            is_: bool = True if value.upper() == "TRUE" else False
            return is_

        if "." in value:
            if "length" in key:
                return float(value) * 1e-6
            else:
                return float(value)

        return int(value)

    def config_file2dict(self, file) -> dict:
        config_dict = MeasuramentParameters.config_dict_default
        for l in file.readlines():
            l = l[:-1]
            number = re.findall(r"[-+]?\d*\.\d+|\d+", l.strip("\n"))
            if len(l) == 0:
                continue
            elif len(number) > 0:
                key = re.sub(r"[-+]?\d*\.\d+|\d+|:|\u3000|_|", "", l.strip("\n"))
                value = number
            else:
                key = " ".join(l.split(" ")[:-1])
                value = l.split(" ")[-1]
            config_dict[key.rstrip(" ").lawer()] = self.str2goodtype(key, "".join(value))
            config_dict.update(self.config_overwite)
        config_dict["map_shape"] = (int(config_dict["xstep"]), int(config_dict["ystep"]))

        config_dict["all_length"] = int(config_dict["app_points"]) + int(config_dict['ret_points'])
        return config_dict
    
    def load_measurament_config(self, save_dir_name:pathLike, config_path: str = "config.txt") -> dict:
        
        import pickle
        if not os.path.isfile("config_dict.pkl"):
            try:
                with open(os.path.join(save_dir_name, config_path), "r", encoding="utf-8") as f:
                    config_dict = self.config_file2dict(f)
            except UnicodeDecodeError:
                with open(os.path.join(save_dir_name, config_path), "r", encoding="shift-jis") as f:
                    config_dict = self.config_file2dict(f)
            except FileNotFoundError:
                config_dict = self.config_dict_default
                
            with open(os.path.join(save_dir_name, "config_dict.pkl"), "wb") as tf:
                pickle.dump(config_dict, tf)
        else:
            with open(os.path.join(save_dir_name, "config_dict.pkl"), "wb") as tf:
                config_dict = pickle.load(tf)

        return config_dict
