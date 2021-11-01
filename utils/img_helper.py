from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .typing import pathLike
from ..fitting._base_analyzer import FCBaseProcessor


def cut_cross_section(img        :np.ndarray,
                      edge_points:list[list] = [[0,1],[0,1]],
                      is_plot    :bool       = False,
                      save_path  :pathLike   = "./",
                      img_name   :str        = "cross_section",
                      data_annot :bool       = False):
    """画像をある断面のデータを取得する関数。  
    Parameters
    ----------
    img : np.ndarray
        画像の配列(reshapeした2次元配列)
    edge_points : arrayLike
        取得したい断面の直線の短点
        [[x0,x1],[y0,y1]]の形式

    """
    if edge_points[0][0]!=edge_points[0][1]:
        coeffs = FCBaseProcessor.linefit(edge_points[0],edge_points[1],d="easy_fit")
        x_data = np.arange(0, len(img), 1 / (len(img)+1))
        y_data = x_data*coeffs[0]+coeffs[1]
    elif edge_points[0][0]==edge_points[0][1]:
        x_data = np.ones(len(img))*edge_points[0][0]
        y_data = np.arange(0,len(img))

    if is_plot:
        fig, ax = plt.subplots(1,1,figsize=(20,20))
        ax.set_xlim([-0.5,img.shape[0]-0.5])
        ax.set_ylim([-0.5,img.shape[1]-0.5])
        ax.plot(x_data,y_data)
        if data_annot:
            sns.heatmap(data=np.round(img,1), cmap="RdBu_r", annot=True, fmt=".2f",  square=True)
            ax.invert_yaxis()
        else:
            ax.imshow(img,origin="lower",cmap="Greys")
        plt.savefig(os.path.join(save_path,f"{img_name}_x_{edge_points[0][0]}__{edge_points[0][1]}_y_{edge_points[1][0]}_{edge_points[1][1]}"))
    
    idx_in_img : list[bool]= (x_data>=0) & (x_data<img.shape[0]) &  (y_data>=0) & (y_data<img.shape[1])
    
    x_data, y_data=np.int32(x_data[idx_in_img]), np.int32(y_data[idx_in_img])
    
    cross_section_idx : np.ndarray = np.unique(np.vstack([x_data, y_data]).T, axis=0)
    cross_section     : np.ndarray = img[cross_section_idx[:,1],cross_section_idx[:,0]]
    return cross_section, cross_section_idx


def get_region(data     :np.ndarray, 
               vrange   :list[float,float]             =None,
               hlines   :list[list[int,list[int,int]],]=[],
               vlines   :list[list[int,list[int,int]],]=[],
               drop_points:list[list[int,int]]         =None,
               add_points:list[list[int,int]]          =None,
               save_path :str                          = ""):
    """領域をとる関数。

    Parameters
    ----------
    data : np.ndarray
        全領域
    vrange : list[float,float], optional
        画像作成の際のデータ幅, by default None
    hlines : list[list[int,list[int,int]],], optional
        水平のラインでデータをとるデータ
        [[y, [水平の始まり,水平の終わり]]...]
        , by default []
    vlines : list[list[int,list[int,int]],], optional
        [[x, [垂直の始まり,垂直の終わり]]...]
        垂直のラインでデータをとるデータ, by default []
    drop_points : list[list[int]], optional
        [[x,y],..], by default None
    add_points : list[list[int]], optional
        [[x,y],..], by default None
    save_path : str, optional
        データを保存するパス(拡張子なし), by default ""
    """
    extract_idx=[]
    mask = np.zeros(data.shape,dtype=float)
    h_data = np.vstack([ np.vstack([np.arange(h_line[1][0],h_line[1][1]+1),
                                        np.ones(h_line[1][1]-h_line[1][0]+1)*h_line[0]
                                    ]).T 
                            for h_line in hlines
                        ])
    if len(h_data)>0 and len(extract_idx)==0:
        extract_idx = h_data
    v_data = np.vstack([ np.vstack([np.ones(v_line[1][1]-v_line[1][0]+1)*v_line[0],
                                        np.arange(v_line[1][0],v_line[1][1]+1)
                                  ]).T 
                            for v_line in vlines
                        ])
    if len(v_data)>0 and len(extract_idx)==0:
        extract_idx = v_data
    else:
        extract_idx = np.vstack([extract_idx, v_data])
    # print(extract_idx)
    
    if drop_points:
        drop_idx  = np.hstack([
                            (extract_idx[:,0] == drop_point[0]).reshape(-1,1)
                                & 
                            (extract_idx[:,1] == drop_point[1]).reshape(-1,1)
                            for drop_point in drop_points
                        ]).any(axis=1)
        drop_idx = [ not di for di in drop_idx]
        extract_idx = extract_idx[ drop_idx]
    if add_points:
        extract_idx=np.vstack(extract_idx,add_points)
    if not vrange:
        vrange = [np.min(data),np.max(data)]
    extract_idx = np.int32(np.unique(extract_idx, axis=1))
    mask[extract_idx[:,0],extract_idx[:,1]]=1
    extract_data = data[extract_idx]
    np.save("{}.npy".format(save_path),extract_data)
    
    plt.imshow(data,vmin=vrange[0],vmax=vrange[1], cmap="Greys",origin="lower")

    plt.imshow(data,vmin=vrange[0],vmax=vrange[1],origin="lower",alpha = mask, cmap="seismic")
    plt.legend()
    plt.savefig(save_path)
