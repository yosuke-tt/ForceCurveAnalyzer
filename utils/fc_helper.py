import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as mplstyle

mplstyle.use('fast')
class FCHelper():
    def __init__(self, savepath):
        self.save_path = savepath
    def im_def_z_row(self,deflection, zsensor):
        for i, (d,z) in enumerate(zip(deflection, zsensor)):
            fig, ax = plt.subplots(1,2,figsize=(24,12))
            ax[0].plot(d)
            ax[0].set_title("deflection")
            ax[1].plot(zsensor)
            ax[1].set_title("z")
            fig.suptitle(f"ForceCurve {i}")
            fig.savefig(os.path.join(self.save_path,"ForceCurve","ForceCurve_{:>03}".format(i)))
            plt.close()

    def topo_img(self,topo_c, topo_t, compare = False):
        if compare == True:
            fig, ax = plt.subplots(1, 3, figsize=(50,12))
            ax[0].imshow(topo_c, vmin=np.min( topo_t), vmax=np.max( topo_t), cmap = "Greys")
            ax[0].set_title(f"topo(contact)")
            m = ax[1].imshow(topo_t, vmin=np.min( topo_t ), vmax=np.max(topo_t), cmap = "Greys")
            ax[1].set_title(f"topo(trig)",fontsize=21)
            plt.colorbar(m, ax=ax[1])
            m1=ax[2].imshow(topo_t-topo_c, cmap = "Greys")
            ax[2].set_title(f"topo diff(trig-contact)",fontsize=21)
            plt.colorbar(m1, ax=ax[2])
            fig.savefig(os.path.join(self.save_path,"topo_compare"))
            plt.close()
        else:
            plt.imshow(topo_c, vmin=np.min(topo_c), vmax=np.max(topo_c), cmap = "Greys")
            plt.colorbar()
            plt.title(f"topo(contact) {np.min(topo_c)} / {np.max(topo_c)}")
            plt.savefig(os.path.join(self.save_path,"topo_contact"))
            plt.close()



    def im_all_fitted_approach(self,  force_app, delta_app, contact, app_coef_lin):

        fig, ax = plt.subplots(self.map_shape[0],self.map_shape[1], figsize=(120,100))
        if (self.map_shape[0]!=1) and (self.map_shape[1]!=1):
            for i, (af, ad, fm, ac) in enumerate(zip(force_app, delta_app, contact, app_coef_lin)):
                x, y = i//self.map_shape[0], i%self.map_shape[1]
                ax[x,y].set_xlim(0,1.e-5)
                ax[x,y].set_ylim(0,3.5e-6)
                ax[x,y].plot(ad, af**(2/3))
                ax[x,y].plot(ad[contact:], ad[contact:]*ac[0]+ac[1], c = "red")
                # ax[x,y].vlines(-ac[1]/ac[0], ymin = np.min(ad)-1e-9, ymax=np.max(af), label="contact ", color = "red")
            fig.savefig(os.path.join(self.save_path,"all_app_fitted"))
            plt.close()

    def arrow_imshow(self,topo_contact,gx,gy):
        plt.imshow(topo_contact, cmap = "Greys")
        plt.quiver(np.arange(self.map_shape[0]), np.arange(self.map_shape[1]),gx,gy,angles="xy",headwidth=3,scale=20,color="red")
        plt.title("topo map and gradient", fontsize=21)
        plt.savefig(os.path.join(self.save_path, "topo_grad_vector"))
        plt.close()


    def crop_image(self, im, top=None, left=None, bottom=None, right=None, imgname=None):
        if top is None:
            top = 0
        if left is None:
            left = 0
        if bottom is None:
            bottom = im.shape[0]
        if right is None:
            right = im.shape[1]


        cmin =np.min(im)
        cmax =np.max(im)
        fig, ax = plt.subplots(1, 3,figsize=(40,10))

        cm = ax[0].imshow(im, vmin =cmin, vmax =cmax, cmap ="Greys")

        r = patches.Rectangle(xy=(left-0.6, top-0.6), width=right-left+0.6, height=bottom-top+0.6, ec="#E11225", fill=False,linewidth=10)
        ax[0].add_patch(r)

        imcrop=im[top:bottom,left:right]
        cm1 = ax[1].imshow(imcrop, vmin =cmin, vmax =cmax, cmap ="Greys")
        ax[1].set_title(f"top={top},left={left},bottom={bottom},right={right}",fontsize=25)
        plt.colorbar(cm1, ax = ax[1])

        cm2 =ax[2].imshow(imcrop, cmap ="Greys")
        plt.colorbar(cm2, ax = ax[2])

        if imgname is None:
            plt.savefig(os.path.join(savepath,f"cropped_{top}_{left}_{bottom}_{right}"))
        else:
            plt.savefig(os.path.join(savepath, imgname+f"_cropped_{top}_{left}_{bottom}_{right}"))


    def image_3d(self, topomap_path, imgname=None, e_path = None):
        """

        3d画像を出す関数(仮)

        Parameters
        ----------
    　　topomap_path :  str
            AFM2から出たtopo像のまっぷファイルのパス
        
        savepath : str ,  defualt None
            保存先のパス
            Noneの場合、topomap_pathのあるディレクトリに保存
        
        imgname : str,  defualt None
            保存する画像の名前。
            Noneの場合、img_3d


        """
        if savepath is None:
            savepath = topomap_path.split("\\")[-2]

        im = np.loadtxt(topomap_path)

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d' )
        x = np.arange(0, im.shape[0], 1)
        y = np.arange(0, im.shape[1], 1)
        xx,yy = np.meshgrid(x,y)


        surf = ax.plot_surface(xx, yy,1- im, cmap=plt.cm.coolwarm,
                            linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig("./3d")
        if imgname is None:
            plt.savefig(os.path.join(savepath,"img_3d"))
        else:
            plt.savefig(os.path.join(savepath, imgname))


    def trial_img(self,v, study, i):
        plt.ylim([np.min(v)-1e-4,np.min(v)+1e-3])
        plt.plot([t.value for t in study.trials])
        plt.savefig(os.path.join(self.save_path, "trials","{:>03}".format(i)))
        plt.close()

    def sr_fitting_img(self,sr_time,y_pred,et,e0,e_inf,alpha,res,i):
        fig, ax = plt.subplots(1,2,figsize=(20,10))
        ax[0].plot(sr_time,et)
        ax[0].plot(sr_time,y_pred, label=" e_inf  : {}\n e0    : {}\n alpha : {}\n residual : {}".format(e_inf, e0, alpha,res))
        ax[0].set_title(self.save_path+"{:>03}".format(i))
        ax[1].plot(np.log10(sr_time),np.log10(et))
        ax[1].plot(np.log10(sr_time),np.log10(y_pred), label=" e_inf  : {}\n e0    : {}\n alpha : {}\n residual : {}".format(e_inf, e0, alpha,res))
        ax[1].legend(fontsize=10)

        plt.savefig(os.path.join(self.save_path,"fit_sr","{:>03}".format(i)))
        plt.close()

    def data_statistics(self,data,data_name,vmin=True,stat_type=["map","map_only","hist","plot"]):
        np.save(os.path.join(self.save_path,data_name),data)
        num_plot=len(stat_type)
        min_idx = np.argsort(data.reshape(-1,1))[1]

        data_std = np.std(data)
        if vmin==True:
            vmin = np.mean(data)-2*data_std
        vmax = np.mean(data)+2*data_std

        if "map_only" in stat_type:

            plt.imshow(data.reshape(self.map_shape),vmin=vmin,vmax=vmax,cmap="Greys")
            plt.colorbar()
            plt.savefig(os.path.join(self.save_path,data_name))
            plt.close()
            num_plot-=1

            plt.imshow(data.reshape(self.map_shape),vmin=vmin,cmap="seismic")
            plt.colorbar()
            plt.savefig(os.path.join(self.save_path,data_name+"_colored"))
            plt.close()

        fig, ax = plt.subplots(1,num_plot,figsize=(8*num_plot,5))
        i=0
        data_max= np.max(data)
        data_min=np.min(data)
        data_mean = np.mean(data) 
        data_median = np.median(data)

        if "map" in stat_type:
            mapple = ax[i].imshow(data.reshape(self.map_shape),vmin=vmin,cmap = "Greys")
            plt.colorbar(mapple,ax=ax[i])
            i+=1
        try:
            if "hist" in stat_type:
                ax[i].hist(data.reshape(-1,1))
                i+=1
            if "plot" in stat_type:
                ax[i].plot(data.reshape(-1,1))
                i+=1
        except ValueError:
            print("Value Error")
        ax[i//2].set_title("{} ,max : {:6.2f}, min : {:6.2f}, mean : {:6.2f}, median : {:6.2f}".format(data_name,data_max, data_min, data_mean, data_median))
        plt.savefig(os.path.join(self.save_path, data_name+"stat"))
        plt.close()
        # exit()

    def zoom_data(self,data,data_name, img_split_shape=(8,8), num_plot=(4,4)):
        data_shape = data.shape
        fig, ax = plt.subplots(*num_plot, figsize=(16,16))


        h_step = int((data_shape[0]/img_split_shape[0])*2)
        w_step = int((data_shape[1]/img_split_shape[1])*2)

        for h in range(4):
            for w in range(4):
                data_tmp = data[h*4:h*4+h_step, w*4:w*4+w_step]
                mapple = ax[h,w].imshow(data_tmp, cmap = "Greys")
                plt.colorbar(mapple, ax=ax[h,w])
        plt.savefig(os.path.join(self.save_path,data_name+"_zoom.png"))
        plt.close()

    def plot_contact_img(self,force_app, force_ret, delta_app, delta_ret, coeffs,contact,E):
        fig, ax = plt.subplots(*self.map_shape, figsize=(40,40))
        cmap = plt.get_cmap("Reds")
        cn = cmap.N
        erange=np.nanmax(E)-np.nanmin(E)
        back_ground = cn*E/erange

        for i, (fa,fr,da,dr, coeff,c,bc) in enumerate(zip(force_app, force_ret, delta_app, delta_ret,coeffs,contact, back_ground)):
            h = int(i//self.map_shape[0])
            w = int(i%self.map_shape[0])
            ax[h,w].set_title(str(i))
            ax[h,w].plot(np.concatenate([da,dr]),np.concatenate([fa,fr])**(2/3))
            ax[h,w].plot(da[c:],da[c:]*coeff[0]+coeff[1])
            ax[h,w].scatter(da[c],fa[c]**(2/3),color="red")
            try:
                b = cmap(int(bc))
            except ValueError:
                b = cmap(int(0))
            b = (*b[:-2],0.5)
            ax[h,w].set_facecolor(b)

        plt.savefig(os.path.join(self.save_path,"contact_all"))
        plt.close()

    def fit_sr_all(self,x,y,y_pred,value):
        np.save("data",(x,y,y_pred,value,self.save_path,self.map_shape))
        fig, ax = plt.subplots(self.map_shape[0],self.map_shape[1], figsize=(90,90))
        cmap = plt.get_cmap("Reds")
        cn = cmap.N
        vrange=np.nanmax(value)-np.nanmin(value)
        back_ground = cn*value/vrange
        for i, (yy,yp,v,bc) in enumerate(zip( y, y_pred, value,back_ground)):
            h = int(i//self.map_shape[0])
            w = int(i%self.map_shape[0])

            ax[h,w].plot(x[::100],yy[::100])
            ax[h,w].plot(x[::100],yp[::100],color="red")
            ax[h,w].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)

            try:
                b = cmap(int(bc))
            except ValueError:
                b = cmap(int(0))
            b = (*b[:-2],0.5)
            ax[h,w].set_facecolor(b)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        plt.savefig(os.path.join(self.save_path,"contact_all"))

        plt.close()

    def plot_3d(self, x,y,z):
        X,Y = np.meshgrid(x,y)

        fig = plt.figure(dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,z,cmap="coolwarm")
        ax.set_box_aspect((1, 1,1))
        plt.savefig(os.path.join(self.save_path,"3d_topo.png"),dpi=130)
        plt.close()


    def slice_3d(self,x,y,z,xfit,yfit,zfit,xystep,resolution):
        fig = plt.figure(figsize=(30,30))
        gs = gridspec.GridSpec(ncols=len(x)+10, nrows=len(y)+10)

        plt.subplot(gs[:z.shape[0], :z.shape[1]])
        plt.imshow(z, cmap="seismic")
        for i in range(len(x)):
            plt.subplot(gs[i, z.shape[1]:])
            plt.plot(xfit, zfit[int(resolution/2+i*resolution),:])

            plt.scatter(x,z[i,:])

        for i in range(len(y)):
            plt.subplot(gs[z.shape[0]:, i])
            plt.plot(zfit[:,int(resolution/2+i*resolution)],yfit)
            plt.scatter(z[:,i],y)

        plt.savefig(os.path.join(self.save_path,"split_3d"))
        plt.close()

        fig = plt.figure(figsize=(30,30))
        plt.imshow(-zfit, cmap="Greys")
        plt.savefig(os.path.join(self.save_path,"zfit"))
        plt.close()


    def fitting_ting(self,d,f,d_fit,f_fit,fitted_result,r, resi, index):
        if len(r)==2:
            plt.title("e0 {:>7.2} alpha {:>7.2} err : {:>7.2}".format(r[0],r[1],resi))
        else:
            plt.title("e0 {:>7.2} alpha {:>7.2} e_inf {:>7.2} err : {:>7.2}".format(float(r[0]),float(r[1]),float(r[2]),float(resi)))
        plt.plot(d,f, label = "all data")
        plt.plot(d_fit,f_fit[:len(d)], label = "fit data")
        plt.plot(d,fitted_result, label = "fitted plot")
        plt.legend()
        plt.savefig(self.savefile2savepath("fit_img/{:>03}".format(index)))
        plt.close()
    
    def imshow_ax(self, data, ax, title):
        m = np.median(data)
        st = np.std(data)
        mapple = ax.imshow(data,vmin=np.max([0,m-2*st]),vmax=m+2*st,cmap="Greys")
        ax.set_ylabel(title, fontsize=20)
        plt.colorbar(mapple,ax=ax)
        return ax

    def plot_ting_summury(self, topo, young, result, e1, res ):
        plt.close()
        fig, ax = plt.subplots(len(result.T)+4,1, figsize=(10,40))
        ax[0]=self.imshow_ax(topo.reshape(self.map_shape), ax[0], "topo")
        ax[1]=self.imshow_ax(np.array(young).reshape(self.map_shape), ax[1], "Young")
        j=2
        title = ["E0","alpha","E inf"]
        for i, (r, t) in enumerate(zip(result.T,title)):
            ax[j]=self.imshow_ax(r.reshape(self.map_shape), ax[j], t)
            j+=1
        ax[j]=self.imshow_ax(e1.reshape(self.map_shape), ax[j],"E1")
        j+=1
        try:
            res = np.array([ np.mean(r) for r in res])
            ax[j]=self.imshow_ax(res.reshape(self.map_shape), ax[j], "residuals")
        except ValueError:
            print(res.shape)
        plt.savefig(self.savefile2savepath("summury.png"))
        plt.close()
    
    def plot_preprocessed_img(self):
        os.makedirs(os.path.join(self.save_path, "processed_img"),exist_ok=True)
        for i,(d,f,dd,ff) in enumerate(zip(self.delta_data,self.force_data,self.delta, self.force)):
            fig, ax= plt.subplots(1,2,figsize=(15,5))
            ax[0].plot(dd[::10],ff[::10])
            ax[1].plot(d[::10],f[::10])
            plt.savefig(os.path.join(self.save_path,"processed_img","{:>03}".format(i)))
            plt.close()
    def ret_contact_img(self):
        os.makedirs(self.savefile2savepath("ret_contact"),exist_ok=True)
        for i, (dr, da, fa,fr,c,cr) in enumerate(zip(self.delta_ret,self.delta_app,self.force_app,self.force_ret,self.contact,self.ret_contact)):
            plt.plot(da,fa,zorder=2)
            plt.scatter(da[c],fa[c],color="blue",zorder=3)
            plt.plot(dr,fr,zorder=1)
            plt.scatter(dr[cr],fr[cr],color="red",zorder=4)
            plt.savefig(self.savefile2savepath("ret_contact/{:>03}".format(i)))
            plt.close()

    def plot_set_base(self,force_app_base,force_ret_base):
        os.makedirs(self.savefile2savepath("base_plot"),exist_ok=True)
        for i, (dr, da, fa,fr,c,cr,fab,frb) in enumerate(zip(self.delta_ret,self.delta_app,self.force_app,self.force_ret,self.contact,self.ret_contact,force_app_base,force_ret_base)):
            plt.plot(da,fa,color="red")
            plt.plot(dr,fr,color="red")
            plt.plot(da,fab,color="blue")
            plt.plot(dr,frb,color="blue")
            plt.savefig(self.savefile2savepath("base_plot/{:>03}".format(i)))
            plt.close()

if __name__ == "__main__":
    x,y,y_pred,value,self.save_path,self.map_shape=np.load("data.npy",allow_pickle=True)
    fit_sr_all(x,y,y_pred,value,self.save_path)


