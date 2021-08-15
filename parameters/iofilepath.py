from __future__ import annotations
import os
from dataclasses import dataclass


#! 
#?
#TODO

@dataclass 
class FilePathes:
    save_path: str= "tmp"
    data_path: str | bool = False
    def __post_init__(self):
        os.makedirs(self.save_path, exist_ok = True)
        self.is_data_path :bool = (not isinstance(self.data_path, bool)) and os.path.isdir(self.data_path)
        
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
        
        data_file_path:str = os.path.join(self.data_path, filname)
        if self.is_data_path and os.path.isfile(data_path):
            data:np.ndarray=np.load(os.path.join(self.data_path, filname), allow_pickle=True)
        elif os.path.isfile(os.path.join(self.save_path, filname)):
            data: np.ndarray = np.load(os.path.join(self.save_path, filname), allow_pickle=True)
        else:
            data: np.ndarray = False
        return data
    
