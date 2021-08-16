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
    def __init__(self,xystep_length=(3e-6,3e-6), resolution = 10, map_shape=(20,20),save_path="./"):
        self.xystep_length = xystep_length
        self.resolution = 10
        self.save_path = save_path
        self.map_shape = map_shape

    def slice_3d_img(self,x,y,z,gx,gy,habs,theta,cos_map,xfit,yfit,zfit,spline=True):
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

    def fit_topo_multidimension(self,topo, plots, fit_plots, dim=8):
        x_fit = np.zeros((len(plots[0]),len(fit_plots[1])))
        y_fit = np.zeros((len(fit_plots[0]),len(plots[1])))
        xx = np.vstack([ fit_plots[0]**i for i in range(dim+1) ][::-1])
        yy = np.vstack([ fit_plots[1]**i for i in range(dim+1) ][::-1])

        for i, t in enumerate(topo):
            x_fit[i] += np.dot(xx.T,np.polyfit(plots[0], t, dim))
        for i, t in enumerate(topo.T):
            y_fit[:,i] += np.dot(xx.T,np.polyfit(plots[1], t, dim)).T
        return x_fit, y_fit

    def fit_topo_spline(self,topo,plots, fit_plots):

        f = interpolate.interp2d(plots[0], plots[1], topo, kind="quintic")
        z_fit = f(fit_plots[0],fit_plots[1])

        return z_fit

    def isMutiorSpline(self, methods):
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
    def edge_filter(self,img):
        img[:,0] = img[:,1]
        img[:,-1] = img[:,-2]

        img[0,:] = img[1,:]
        img[-1,:] = img[-2,:]

        img[0,0] = np.mean([img[0,1],img[1,0],img[1,1]])
        img[-1,0] = np.mean([img[-1,1],img[-2,0],img[-2,1]])
        img[0,-1] = np.mean([img[0,-2],img[1,-1],img[1,-2]])
        img[-1,-1] = np.mean([img[-1,-2],img[-2,-1],img[-2,-2]])
        return img

    def fit(self,topo, methods = "multi_8"):
        topo=np.max(topo)-topo

        methods_r = self.isMutiorSpline(methods)
        x = np.arange(self.xystep_length[0]/2,self.xystep_length[0]*self.map_shape[0]+self.xystep_length[0]/2,self.xystep_length[0])
        y = np.arange(self.xystep_length[1]/2,self.xystep_length[1]*self.map_shape[1]+self.xystep_length[1]/2,self.xystep_length[1])

        xystep_length_fit = (self.xystep_length[0]/self.resolution,self.xystep_length[1]/self.resolution)
        x_fit = np.arange(0,(xystep_length_fit[0])*self.map_shape[0]*self.resolution+xystep_length_fit[0]/2,xystep_length_fit[0])
        y_fit = np.arange(0,(xystep_length_fit[1])*self.map_shape[1]*self.resolution+xystep_length_fit[1]/2,xystep_length_fit[1])

        if methods_r[0]=="multi":
            x_fit_mul, y_fit_mul = self.fit_topo_multidimension(topo, plots=(x,x), 
                                                                fit_plots=(x_fit,y_fit)
                                                                ,dim=methods_r[1])
            x_fit_mul_grad = np.gradient(x_fit_mul,xystep_length_fit[0],axis=1)
            y_fit_mul_grad = np.gradient(y_fit_mul,xystep_length_fit[1],axis=0)
            gx_d = x_fit_mul_grad[:,int(self.resolution/2):-int(self.resolution/2):self.resolution]
            gy_d = y_fit_mul_grad[int(self.resolution/2):-int(self.resolution/2):self.resolution,:]
            spline = False
            z_fit=(x_fit_mul, y_fit_mul)
        if methods_r[0]=="spline":
            z_fit = self.fit_topo_spline(topo, plots=(x,y), fit_plots=(x_fit,y_fit))
            gy, gx = np.gradient(z_fit,*xystep_length_fit)
            spline = True
            gx_d =gx[int(self.resolution/2):-int(self.resolution/2):self.resolution,int(self.resolution/2):-int(self.resolution/2):self.resolution]
            gy_d = gy[int(self.resolution/2):-int(self.resolution/2):self.resolution,int(self.resolution/2):-int(self.resolution/2):self.resolution]

        habs    = np.sqrt(gx_d**2+gy_d**2)
        theta   = np.arctan(habs)
        cos_map = np.cos(theta)


        cos_map = self.slice_3d_img(x,y,
                                    topo,
                                    gx_d,gy_d,
                                    habs,theta,cos_map
                                    ,x_fit,y_fit,z_fit,
                                    spline=spline)
        return cos_map**(5/2)



if __name__ == '__main__':
    # img =  np.arange(0,100).reshape(10,10)
    # print(img)
    # ga = GradientAdjsutment().edge_filter(img)
    # print(ga)
    args=get_argparser()
    abs_fcdirpath = os.path.abspath(args.fcdirpath)

    print(abs_fcdirpath)
    dirname = os.path.basename(abs_fcdirpath)
    os.makedirs("20210629", exist_ok=True)
    if not os.path.isfile(os.path.join(abs_fcdirpath,"forcecurve.npy")):
        fc_dirs = common.search_dirs(abs_fcdirpath)
        fc_paths = [os.path.split(d)[0] for d in fc_dirs]
        np.save(os.path.join(abs_fcdirpath,"forcecurve.npy"), fc_paths)
    else:
        fc_paths = np.load(os.path.join(abs_fcdirpath,"forcecurve.npy"))

    for i,fc_path in enumerate(fc_paths):
        print(fc_path)
        if os.path.isfile(os.path.join(fc_path,"topo_contact.npy")):
            topo = np.load(os.path.join(fc_path,"topo_contact.npy"))
            ga = GradientAdjsutment().gradient_topo(topo,methods="spline")