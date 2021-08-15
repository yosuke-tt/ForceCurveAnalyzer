import os
from glob import glob

import numpy as np

class LoadFCdirs:
    def __init__(self,fc_dir_path):
        self.fc_dir_path = os.path.abspath(fc_dir_path)

    def search_dirs( self, 
                    fc_parent_path = "../20201123_fc_row_data", 
                    search_dir = "ForceCurve", 
                    isdir=True):
        if isdir:
            search_dir += "/"
        return [ os.path.split(g)[0] for g in glob(os.path.join(fc_parent_path,"**",search_dir),recursive=True)]

    def load_fcdir_path(self):
        fc_dirs = self.search_dirs(self.fc_dir_path)
        fc_paths = np.array([os.path.split(d)[0] for d in fc_dirs])
        np.savetxt(os.path.join(self.fc_dir_path,"fc_dir_paths.txt"), fc_paths, fmt="%s")
        return fc_paths

    def load_fcdir_paths(self):
        print("start")
        if not os.path.isfile(os.path.join(self.fc_dir_path,"fc_dir_paths.txt")):
            fc_paths =self.load_fcdir_path()
        else:
            print("load")
            fc_paths = np.loadtxt(os.path.join(self.fc_dir_path,"fc_dir_paths.txt"), dtype= "str")
            print("finish loading")

        if fc_paths.ndim==0:
            fc_paths = np.array([fc_paths])

        if not os.path.isdir(fc_paths[0]):
            fc_paths =self.load_fcdir_path()
        if fc_paths.ndim==0:
            fc_paths = np.array([fc_paths])
            
        return fc_paths

if __name__ == "__main__":
    lf = LoadFCdirs("../data_20210309_new/")
    print(lf.load_fcdir_paths())
