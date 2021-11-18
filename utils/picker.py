from __future__ import annotations

import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import trace
import seaborn as sns

from .typing import pathLike


def cut_cross_section(img        :np.ndarray,
                      edge_points:list[list] = [[0,1],[0,1]],
                      is_plot    :bool       = False,
                      is_save    :bool       = False,
                      save_path  :pathLike   = "./",
                      img_name   :str        = "cross_section",
                      data_annot :bool       = False):
    """画像をある断面のデータを取得する関数。  
    ***edge_pointsの形式ちがうごめんなさい。
    Parameters
    ----------
    img : np.ndarray
        画像の配列(reshapeした2次元配列)
    edge_points : arrayLike
        取得したい断面の直線の短点
        [[x0,x1],[y0,y1]]の形式

    """
    if edge_points[0][0]!=edge_points[0][1]:
        

        coeffs = linefit_easy_fit(edge_points[0],edge_points[1]) #FIXME:circular import 対策
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
        ax.set_xticklabels(ax.get_xticks(),fontsize=20)
        ax.set_yticklabels(ax.get_yticks(),fontsize=20)
        
        edge_points_s = [
            str(edge_points[0][0]).replace(".","p"),
            str(edge_points[0][1]).replace(".","p"),
            str(edge_points[1][0]).replace(".","p"),
            str(edge_points[1][1]).replace(".","p"),
                            ]
        plt.savefig(os.path.join(save_path,f"{img_name}_x_{edge_points_s[0]}__{edge_points_s[1]}_y_{edge_points_s[2]}_{edge_points_s[3]}"))
    img_idx = (x_data>=np.min(edge_points[0]))& (x_data<=np.max(edge_points[0]))& (y_data<=np.max(edge_points[1]))& (y_data>=np.min(edge_points[1]))

    x_data = np.int32(x_data[img_idx]+0.5)
    y_data = np.int32(y_data[img_idx]+0.5)

    cross_section_idx : np.ndarray = np.unique(np.vstack([x_data, y_data]).T, axis=0)
    cross_section_idx = cross_section_idx[
                                        (0<=cross_section_idx[:,0]) &\
                                        (img.shape[0]>cross_section_idx[:,0]) &\
                                        (0<=cross_section_idx[:,1]) &\
                                        (img.shape[1]>cross_section_idx[:,1])                                        
                                          ]

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
    FIXME:
        バグあり＾polygon_edge_points
        三角に切るときに距離順にする必要ある。今は、気休め距離
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
        import pandas as pd #numpyで簡単にやれればなあ
        
        for polygon_edge_point in polygon_edge_points:
            polygon_edge_point=np.array(polygon_edge_point)
            polygon_edge_point_sort_idx=np.argsort(np.sum(np.abs(polygon_edge_point-polygon_edge_point[0]),axis=1))
            polygon_edge_point = polygon_edge_point[polygon_edge_point_sort_idx]

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
    elif len(add_points)>0:
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
    
def linefit_easy_fit(x,y,cp=0):    
    a = (y[cp:][-1] - y[cp:][0]) / (x[cp:][-1] - x[cp:][0])
    b = y[cp:][-1] - x[cp:][-1] * a
    coeffs = [a, b]
    return coeffs



from ipywidgets import Output, Button

def pick_fc(main_imgs,
            plot_x,plot_y,
            zig=False):
    """

    Parameters
    ----------
    main_img : [type]
        [img, img,...] or img
        一番上の画像
    plot_x,plot_y : [type]
        プロットするデータ
        [arr,arr,arr,...]
    """
    out = Output()
    display(out)
    @out.capture(clear_output=True)
    def _onclick(event):
        nonlocal ax
        nonlocal last_ax
        px, py = event.xdata, event.ydata
        last_ax[0].remove()
        if px != None and py!=None:
            last_ax = ax[-1].plot(plot_x[int(px)+int(py)*20], plot_y[int(px)+int(py)*20])
        ax[-1].set_title("x: {} , y: {}, num : {}".format(int(px), int(py), int(px)+int(py)*20))

    if isinstance(main_imgs[0][0], float):
        main_imgs = [main_imgs]

    fig, ax = plt.subplots(1,1+len(main_imgs),figsize=(20,4))
    for i, main_img in enumerate(main_imgs):
        ax[i].imshow(main_img,cmap="Greys")
    last_ax = ax[-1].plot(plot_x[0],plot_y[0])
    
    cid = fig.canvas.mpl_connect('button_press_event', _onclick)
    
    
    


def pick_data(main_img,
              sub_img):
    """

    Parameters
    ----------
    main_img : [type]
        [img, img,...]
        一番上の画像
    sub_img : [type]
        順番んはmain_imgと同じ
        [[img,img,img..,],[img,img,img,...]]
    """
    def delete_botton(event):            
        nonlocal is_delete
        is_delete = not is_delete
    out = Output()
    button = Button(description='delete')
    button.on_click(delete_botton)
    display(out, button)
    @out.capture(clear_output=True)
    def _onclick(event):
        px, py = event.xdata, event.ydata
        px = int(np.rint(px))
        py = int(np.rint(py))
        rgb = ax_dict[event.inaxes][1][py,px]
        if not ("{}_{}".format(py,px) in list(picked_data.keys())) and not is_delete:    
            picked_data["{}_{}".format(py,px)] = []
            picked_ax["{}_{}".format(py,px)] = []
            for ax_ in ax[:, ax_dict[event.inaxes][0]]:        
                plot = ax_.plot(px,py,'wo')
                rgb=ax_dict[ax_][1][py,px]
                ax_.set_title('value:{}'.format(rgb))
                print(rgb)
                picked_data["{}_{}".format(py,px)].append(rgb)
                picked_ax["{}_{}".format(py,px)].append(plot[0])
                print(picked_data)
        elif ("{}_{}".format(py,px) in picked_data.keys()) and is_delete:
            print(picked_data)
            del picked_data["{}_{}".format(py,px)]
            for ax_ in picked_ax["{}_{}".format(py,px)]:        
                ax_.remove()
            print(picked_data)
    is_delete:bool = False
    
    picked_data = {}
    picked_ax   = {}
    img_num=len(main_img)
    img_types = len(sub_img)+1
    
    fig, ax = plt.subplots(img_types, img_num,figsize=(10,10))
    ax_dict:dict = {}
    
    for i, img in enumerate(main_img):        
        ax[0,i].imshow(img, interpolation='nearest',origin='lower',alpha=1)
        ax_dict[ax[0,i]]=(i,img)
    for i, imgs in enumerate(sub_img):
        for j, img in enumerate(imgs):
            ax[i+1, j].imshow(img, interpolation='nearest',origin='lower',alpha=1)
            ax_dict[ax[i+1, j]]=(j, img)
    
    
    cid = fig.canvas.mpl_connect('button_press_event', _onclick)
    return picked_data