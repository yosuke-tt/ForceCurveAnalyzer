from __future__ import annotations

import os
from glob import glob

import numpy as np

from ..utils.typing import pathLike

def search_dirs(
        fc_dir_path:pathLike,
        search_dir : str ="ForceCurve",
        save_path  :pathLike = "fc_dir_paths.txt"
    ):
    """
    fc_dir_path : pathLike 
        読みこむ親ファイル
    search_dir : str
        検索するファイル名
    
    save_path : pathLike or bool
        ファイルを保存する名前
        
    Returns
    -------
    fc_paths: pathLike
        fc_dir_path内で、search_dirがあるファイル一覧
    """
    
    fc_dirs = [ os.path.split(g)[0] 
                    for g in glob(os.path.join(fc_dir_path,"**",search_dir),
                                    recursive=True
                                )
               ]
    fc_paths = np.array([os.path.split(d)[0] for d in fc_dirs])
    np.savetxt(os.path.join(fc_dir_path,save_path), fc_paths, fmt="%s")
    return fc_paths

def load_fc_dirs( fc_dir_path:pathLike,
                  save_path  :pathLike = "fc_dir_paths.txt"
                  )->list[pathLike]:
    """特定のファイルからForceCurveのあるpathを保存する。

    Returns
    -------
    fc_pathes : list[pathLike]
        ForceCurveのあるpath
    """
    if not os.path.isfile(os.path.join(fc_dir_path, save_path)):
        fc_paths : list[pathLike] =search_dirs(fc_dir_path)
    else:
        fc_paths = np.loadtxt(os.path.join(fc_dir_path, save_path), dtype= "str")

    if fc_paths.ndim==0:
        fc_paths = np.array([fc_paths])

    return fc_paths

if __name__ == "__main__":
    # lf = LoadFCdirs("../data_20210309_new/")
    # print(lf.search_dirs())
    pass