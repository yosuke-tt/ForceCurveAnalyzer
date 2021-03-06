from __future__ import annotations

import argparse
import os
import logging 
import re

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib import gridspec 
from scipy import interpolate 

from ..utils import decorators 

class GradientAdjsutment:
    def __init__(self,
                 xystep_length=(3e-6,3e-6),
                 resolution = 10,
                 map_shape=(20,20),
                 save_path="./"):
        self.xystep_length = xystep_length
        self.resolution = resolution
        self.save_path = save_path
        self.map_shape = map_shape

    def slice_3d_img(self,
                     x,y,z,
                     gx,gy,
                     habs,theta,cos_map,
                     xfit,yfit,zfit,
                     spline=True):
        """結果の表示"""
        fig = plt.figure(figsize=(30,30))
        gs = gridspec.GridSpec(ncols=len(x)+10, nrows=len(y)+10)
        plt.subplot(gs[:z.shape[0], :z.shape[1]])
        plt.quiver(np.arange(self.map_shape[0]), np.arange(self.map_shape[1]),gx,gy,angles="xy",headwidth=3,scale=20,color="red")
        plt.imshow(z, cmap="Greys")
        if spline:
            for i in range(len(x)):
                plt.subplot(gs[i, z.shape[1]:])
                plt.plot(xfit, -zfit[int(self.resolution/2+i*self.resolution),:])
                plt.scatter(x,-z[i,:])

            for i in range(len(y)):
                plt.subplot(gs[z.shape[0]:, i])
                plt.plot(zfit[:,int(self.resolution/2+i*self.resolution)],yfit)
                plt.scatter(z[:,i],y)
        else:
            for i in range(len(x)):
                plt.subplot(gs[i, z.shape[1]:])
                # plt.plot(xfit,-zfit[0][int(self.resolution/2+i*self.resolution),:])
                plt.plot(xfit,-zfit[0][i,:])
                plt.scatter(x,-z[i,:])

            for i in range(len(y)):
                plt.subplot(gs[z.shape[0]:, i])
                plt.plot(zfit[1][:,i],yfit)
                plt.scatter(z[:,i],y)
        if spline:
            plt.savefig(os.path.join(self.save_path,"split_3d"))
        else:
            plt.savefig(os.path.join(self.save_path,"split_3d_multi"))
        plt.close()

        fig, ax = plt.subplots(1,3,figsize=(40,10))
        m1 = ax[0].imshow(habs, cmap="Greys")
        ax[0].set_title("H")
        plt.colorbar(m1, ax=ax[0])

        ax[1].set_title(r"$\theta$")
        m2 = ax[1].imshow(np.degrees(theta), cmap="Greys")
        plt.colorbar(m2, ax=ax[1])
        cont = ax[1].contour(np.degrees(theta),levels=[30,40,50],linewidths=8)
        cont.clabel(fmt='%1.1f', fontsize=14)

        ax[2].set_title(r"$cos(\theta)$")
        m3 = ax[2].imshow(cos_map, cmap="Greys")
        plt.colorbar(m3, ax=ax[2])
        if spline:
            plt.savefig(os.path.join(self.save_path,"cos_map"))
        else:
            plt.savefig(os.path.join(self.save_path,"cos_map_multi"))
        plt.close()

        return cos_map
    @staticmethod
    def fit_topo_multidimension(topo:np.ndarray,
                                plots:np.ndarray,
                                fit_plots:np.ndarray,
                                dim:int=8)->tuple(np.ndarray,np.ndarray):
        """8次元フィッティング(平面でなく、それぞれの方向で行う。)

        Parameters
        ----------
        topo : np.ndarray
            トポグラフィー像(z)
        plots : np.ndarray
            x, y
        fit_plots : np.ndarray
            フィッティングするx,y
        dim : int, optional
            フィッティングのディメンション, by default 8

        Returns
        -------
        x_fit, y_fit:np.ndarray
            フィッティングした後z xfit: x方向のfit,yfit: y方向のfit
        """
        x_fit = np.zeros((len(plots[0]),len(fit_plots[1])))
        y_fit = np.zeros((len(fit_plots[0]),len(plots[1])))
        xx = np.vstack([ fit_plots[0]**i for i in range(dim+1) ][::-1])
        yy = np.vstack([ fit_plots[1]**i for i in range(dim+1) ][::-1])

        for i, t in enumerate(topo):
            x_fit[i] += np.dot(xx.T,np.polyfit(plots[0], t, dim))
        for i, t in enumerate(topo.T):
            y_fit[:,i] += np.dot(yy.T,np.polyfit(plots[1], t, dim)).T
        return x_fit, y_fit
    
    @staticmethod
    def fit_topo_spline(
                        topo:np.ndarray,
                        plots:np.ndarray,
                        fit_plots:np.ndarray
                        )->np.ndarray:
        """スプライン補完をする関数。

        Parameters
        ----------
        topo : np.ndarray
            トポグラフィー像(z)
        plots : np.ndarray
            x,y
        fit_plots : np.ndarray
            フィッティング対称のx,y

        Returns
        -------
        z_fit:np.ndarray
            フィッティングした後のz
        """
        f = interpolate.interp2d(plots[0], plots[1], topo, kind="quintic")
        z_fit = f(fit_plots[0],fit_plots[1])

        return z_fit

    @staticmethod
    def isMultiorSpline(methods:str):
        """スプライン補完と多次元フィッティングの識別を行うための関数。

        Parameters
        ----------
        methods : str
            multiとspline **Noteに注意書き。

        Returns
        -------
            methods :multi =>["multi",int(dim)]
            methods :spline =>["spline"]
        
        Note
        ----
        multiとsplineがmethodsに入っていなければ、methods_8と同じになる。
        """
        if  "multi" in methods:
            dim = methods.split("_")[1]
            if len(dim)==0:
                print("Dimension is not defined")
                print("Dim -> 8 ")
                dim = 8
            return ["multi",int(dim)]
        elif "spline" == methods:
            return ["spline"]
        else:
            print("Methods needs to be multi or spline")
            print(f"{methods}->multi_8")
            return ["multi",int(8)]
        
    def edge_filter(self,
                    img:np.ndarray
                   ):
        """端は、隣のデータを入力することで補完する。
        Parameters
        ----------
        img: np.ndarray
        
        Returns
        -------
        img : np.ndarray
            補正後のイメージ
        """
        img[:,0] = img[:,1]
        img[:,-1] = img[:,-2]

        img[0,:] = img[1,:]
        img[-1,:] = img[-2,:]

        img[0,0] = np.mean([img[0,1],img[1,0],img[1,1]])
        img[-1,0] = np.mean([img[-1,1],img[-2,0],img[-2,1]])
        img[0,-1] = np.mean([img[0,-2],img[1,-1],img[1,-2]])
        img[-1,-1] = np.mean([img[-1,-2],img[-2,-1],img[-2,-2]])
        return img

    def fit(self,
            topo:np.ndarray,
            methods:str = "multi_8",
            is_plot:bool =False)->np.ndarray:
        """トポグラフィー像から、勾配を求める関数。

        Parameters
        ----------
        topo : np.ndarray
            トポグラフィー像
        methods : str, optional
            フィッティングの方法, by default "multi_8"
        Returns
        -------
        cos_map**(5/2) :np.ndarray
            コサインのマップ
        """
        # zsensorから高さに合わせるために逆にする。
        topo : np.ndarray=np.max(topo)-topo
        
        
        methods_r :list[str,int | None]= self.isMutiorSpline(methods)
        
        #x,yの真の点(z)をその値の中心から始まるようにする。
        x = np.arange(self.xystep_length[0]/2,self.xystep_length[0]*self.map_shape[0]+self.xystep_length[0]/2,self.xystep_length[0])
        y = np.arange(self.xystep_length[1]/2,self.xystep_length[1]*self.map_shape[1]+self.xystep_length[1]/2,self.xystep_length[1])

        #fittingしたときに補完するためのステップ幅。
        xystep_length_fit = (self.xystep_length[0]/self.resolution,self.xystep_length[1]/self.resolution)

        #fittingしたときに補完するためのステップ幅。
        x_fit = np.arange(0,(xystep_length_fit[0])*self.map_shape[0]*self.resolution+xystep_length_fit[0]/2,xystep_length_fit[0])
        y_fit = np.arange(0,(xystep_length_fit[1])*self.map_shape[1]*self.resolution+xystep_length_fit[1]/2,xystep_length_fit[1])
        
        #シンプルなフィッティング
        if methods_r[0]=="multi":
            #フィッティング
            x_fit_mul, y_fit_mul = self.fit_topo_multidimension(topo, plots=(x,x), 
                                                                fit_plots=(x_fit,y_fit),
                                                                dim=methods_r[1]
                                                                )
            #微分
            x_fit_mul_grad = np.gradient(x_fit_mul,xystep_length_fit[0],axis=1)
            y_fit_mul_grad = np.gradient(y_fit_mul,xystep_length_fit[1],axis=0)
            
            gx_d = x_fit_mul_grad[:, int(self.resolution/2):-int(self.resolution/2):self.resolution]
            gy_d = y_fit_mul_grad[int(self.resolution/2):-int(self.resolution/2):self.resolution, :]
            spline = False
            z_fit=(x_fit_mul, y_fit_mul)

        if methods_r[0]=="spline":
            # フィッティング
            z_fit = self.fit_topo_spline(topo, plots=(x,y), fit_plots=(x_fit,y_fit))
            gy, gx = np.gradient(z_fit,*xystep_length_fit)
            spline = True
            gx_d =gx[int(self.resolution/2):-int(self.resolution/2):self.resolution,int(self.resolution/2):-int(self.resolution/2):self.resolution]
            gy_d = gy[int(self.resolution/2):-int(self.resolution/2):self.resolution,int(self.resolution/2):-int(self.resolution/2):self.resolution]

        habs    = np.sqrt(gx_d**2+gy_d**2)
        theta   = np.arctan(habs)
        cos_map = np.cos(theta)
        
        #描画
        if is_plot:
            self.slice_3d_img(x,y,
                            topo,
                            gx_d,gy_d,
                            habs,theta,cos_map
                            ,x_fit,y_fit,z_fit,
                            spline=spline)
        return cos_map**(5/2) #=>returnどっちがいいかなあ



