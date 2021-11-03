from __future__ import annotations

import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import trace
import seaborn as sns

from .typing import pathLike
from ..fitting._base_analyzer import FCBaseProcessor


def cut_cross_section(img        :np.ndarray,
                      edge_points:list[list] = [[0,1],[0,1]],
                      is_plot    :bool       = False,
                      is_save    :bool       = False,
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
        y_data = np.arange(0,len(img))+0.5

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
    img_idx = (x_data>=np.min(edge_points[0]))& (x_data<=np.max(edge_points[0]))& (y_data<=np.max(edge_points[1]))& (y_data>=np.min(edge_points[1]))

    x_data = np.int32(x_data[img_idx]+0.5)
    y_data = np.int32(y_data[img_idx]+0.5)

    cross_section_idx : np.ndarray = np.unique(np.vstack([x_data, y_data]).T, axis=0)
    cross_section     : np.ndarray = img[cross_section_idx[:,1],cross_section_idx[:,0]]
    return cross_section, cross_section_idx

def cut_polygon2_triangle(num_poly_edge):
    num_poly_edge_idx = np.arange(num_poly_edge)
    return np.vstack([ np.append(0,num_poly_edge_idx[i+1:i+3]) for i in range(num_poly_edge-2)])

def extract_triangle(data, triangle_points):
    triangle_points_gen = itertools.combinations(triangle_points,2)

    triangle = [  cut_cross_section(data,
                    edge_points = [
                                    np.array(list(triangle_point))[:,0],
                                    np.array(list(triangle_point))[:,1]
                                    ]
                    )[1]
                for triangle_point in triangle_points_gen]
    
    triangle = np.vstack(triangle)
    triangle_by_x = np.unique(triangle[:,0])
    triangle_region = np.vstack([
     np.hstack([
            x_*np.ones(1+np.max(triangle[triangle[:,0]==x_][:,1])-np.min(triangle[triangle[:,0]==x_][:,1])).reshape(-1,1),
            np.arange(np.min(triangle[triangle[:,0]==x_][:,1]),
                        np.max(triangle[triangle[:,0]==x_][:,1])+1).reshape(-1,1)
         ])
        for x_ in triangle_by_x
     ])
    
    return triangle_region
    
    

def get_region(data     :np.ndarray, 
               vrange   :list[float,float]=None,
               region_alpha = 10,
               polygon_edge_points:list=[],
               hlines   :list[list[int,list[int,int]],]=[],
               vlines   :list[list[int,list[int,int]],]=[],
               drop_points:list[list[int,int]]         =[],
               add_points:list[list[int,int]]          =[],
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
    
    np.save(os.path.join(os.path.dirname(save_path),"polygon_edge_points"),polygon_edge_points)
    np.save(os.path.join(os.path.dirname(save_path),"hlines"),hlines)
    np.save(os.path.join(os.path.dirname(save_path),"vlines"),vlines)
    np.save(os.path.join(os.path.dirname(save_path),"drop_points"),drop_points)
    np.save(os.path.join(os.path.dirname(save_path),"add_points"),add_points)

    extract_idx=[]
    
    mask = np.zeros(data.shape,dtype=float)
    if len(polygon_edge_points)>0:
        for polygon_edge_point in polygon_edge_points:
            triangle_idx = cut_polygon2_triangle(len(polygon_edge_point))
            triangle_points = np.array([np.array(polygon_edge_point)[tidx] for tidx in triangle_idx])
            e_idx =np.vstack([
                                    extract_triangle(data, triangle_point) 
                                    for triangle_point in triangle_points
                                ])
            if len(extract_idx)>0:
                extract_idx = np.vstack([extract_idx,e_idx])
            else:
                extract_idx = e_idx
    print("polygon")
    print(extract_idx)


    if len(hlines)>0 :
        h_data = np.vstack([ np.vstack([
                                    np.arange(h_line[1][0],h_line[1][1]+1),
                                    np.ones(h_line[1][1]-h_line[1][0]+1)*h_line[0]
                                    ]).T 
                            for h_line in hlines
                        ])
        if len(extract_idx)==0:
            extract_idx = h_data
        
        else:
            extract_idx = np.vstack([extract_idx, h_data])
    print("hlines")
    print(extract_idx)

    if len(vlines)>0:
    
        v_data = np.vstack([ np.vstack([np.ones(v_line[1][1]-v_line[1][0]+1)*v_line[0],
                                        np.arange(v_line[1][0],v_line[1][1]+1)
                                  ]).T 
                            for v_line in vlines
                        ])
        if len(extract_idx)==0:
            extract_idx = v_data
        
        else:
            extract_idx = np.vstack([extract_idx, v_data])
    print("vlines")
    print(extract_idx)
    if len(drop_points)>0:
        drop_idx  = np.hstack([
                            (extract_idx[:,0] == drop_point[0]).reshape(-1,1)
                                & 
                            (extract_idx[:,1] == drop_point[1]).reshape(-1,1)
                            for drop_point in drop_points
                        ]).any(axis=1)
        drop_idx = [ not di for di in drop_idx]
        extract_idx = extract_idx[ drop_idx]
    print(add_points)
    if len(add_points)>0 and len(extract_idx)>0:
        extract_idx=np.vstack([extract_idx,add_points])
    else:
        extract_idx=add_points
    print("add_points")
    print(extract_idx)

        
    if not vrange:
        vrange = [np.nanmin(data),np.nanmax(data)]
    
    plt.imshow(data,vmin=vrange[0],vmax=vrange[1], cmap="Greys",origin="lower")
    # plt.imshow(data,vmin=vrange[0],vmax=vrange[1], cmap="Greys",origin="lower")
    print(extract_idx)
    if len(extract_idx)>0:
        extract_idx = np.int32(np.unique(extract_idx, axis=1))
        print(extract_idx)
        print(extract_idx[:,0])
        mask[extract_idx[:,1],extract_idx[:,0]]=10
        extract_data = data[extract_idx[:,1],extract_idx[:,0]]
        np.save("{}.npy".format(save_path),extract_data)
        plt.imshow(mask,vmin=vrange[0],
                   vmax=vrange[1],origin="lower",alpha = mask/region_alpha , cmap="Oranges")
        
        plt.legend()
    else:
        extract_data =[]
        
    plt.savefig(save_path)
    plt.close()

    plt.close()
    return extract_data,extract_idx
    