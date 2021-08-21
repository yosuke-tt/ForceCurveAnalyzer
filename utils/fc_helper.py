import os


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.style as mplstyle

mplstyle.use('fast')


def im_def_z_row(save_path,deflection, zsensor):
    for i, (d,z) in enumerate(zip(deflection, zsensor)):
        fig, ax = plt.subplots(1,2,figsize=(24,12))
        ax[0].plot(d)
        ax[0].set_title("deflection")
        ax[1].plot(zsensor)
        ax[1].set_title("z")
        fig.suptitle(f"ForceCurve {i}")
        fig.savefig(os.path.join(save_path,"ForceCurve","ForceCurve_{:>03}".format(i)))
        plt.close()







"""stress relaxation"""
def fit_sr_all(save_path,map_shape,x,y,y_pred,value):
    """fit in sr"""
    
    np.save("data",(x,y,y_pred,value,save_path,map_shape))
    fig, ax = plt.subplots(map_shape[0],map_shape[1], figsize=(90,90))
    cmap = plt.get_cmap("Reds")
    cn = cmap.N
    vrange=np.nanmax(value)-np.nanmin(value)
    back_ground = cn*value/vrange
    for i, (yy,yp,v,bc) in enumerate(zip( y, y_pred, value,back_ground)):
        h = int(i//map_shape[0])
        w = int(i%map_shape[0])

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

    plt.savefig(os.path.join(save_path,"contact_all"))

    plt.close()

def data_statistics(save_path,map_shape,data,data_name,vmin=True,stat_type=["map","map_only","hist","plot"]):
    np.save(os.path.join(save_path,data_name),data)
    num_plot=len(stat_type)
    min_idx = np.argsort(data.reshape(-1,1))[1]

    data_std = np.std(data)
    if vmin==True:
        vmin = np.mean(data)-2*data_std
    vmax = np.mean(data)+2*data_std

    if "map_only" in stat_type:

        plt.imshow(data.reshape(map_shape),vmin=vmin,vmax=vmax,cmap="Greys")
        plt.colorbar()
        plt.savefig(os.path.join(save_path,data_name))
        plt.close()
        num_plot-=1

        plt.imshow(data.reshape(map_shape),vmin=vmin,cmap="seismic")
        plt.colorbar()
        plt.savefig(os.path.join(save_path,data_name+"_colored"))
        plt.close()

    fig, ax = plt.subplots(1,num_plot,figsize=(8*num_plot,5))
    i=0
    data_max= np.max(data)
    data_min=np.min(data)
    data_mean = np.mean(data) 
    data_median = np.median(data)

    if "map" in stat_type:
        mapple = ax[i].imshow(data.reshape(map_shape),vmin=vmin,cmap = "Greys")
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
    plt.savefig(os.path.join(save_path, data_name+"stat"))
    plt.close()
        # exit()
def sr_fitting_img(save_path,sr_time,y_pred,et,e0,e_inf,alpha,res,i):
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    ax[0].plot(sr_time,et)
    ax[0].plot(sr_time,y_pred, label=" e_inf  : {}\n e0    : {}\n alpha : {}\n residual : {}".format(e_inf, e0, alpha,res))
    ax[0].set_title(save_path+"{:>03}".format(i))
    ax[1].plot(np.log10(sr_time),np.log10(et))
    ax[1].plot(np.log10(sr_time),np.log10(y_pred), label=" e_inf  : {}\n e0    : {}\n alpha : {}\n residual : {}".format(e_inf, e0, alpha,res))
    ax[1].legend(fontsize=10)

    plt.savefig(os.path.join(save_path,"fit_sr","{:>03}".format(i)))
    plt.close()




    






    

"""fit2ting"""
def plot_preprocessed_img(save_path,delta_data,force_data,delta, force):
    """determine_data_range in fit2ting"""
    os.makedirs(os.path.join(save_path, "processed_img"),exist_ok=True)
    for i,(d,f,dd,ff) in enumerate(zip(delta_data,force_data,delta, force)):
        fig, ax= plt.subplots(1,2,figsize=(15,5))
        ax[0].plot(dd[::10],ff[::10])
        ax[1].plot(d[::10],f[::10])
        plt.savefig(os.path.join(save_path,"processed_img","{:>03}".format(i)))
        plt.close()
def ret_contact_img(save_path,
                    delta_ret,
                    delta_app,
                    force_app,
                    force_ret,
                    contact,
                    ret_contact):
    """determine_data_range in fit2ting"""
    os.makedirs(os.path.join(save_path,"ret_contact"),exist_ok=True)
    for i, (dr, da, fa,fr,c,cr) in enumerate(zip(delta_ret,
                                                 delta_app,
                                                 force_app,
                                                 force_ret,
                                                 contact,
                                                 ret_contact)):
        plt.plot(da,fa,zorder=2)
        plt.scatter(da[c],fa[c],color="blue",zorder=3)
        plt.plot(dr,fr,zorder=1)
        plt.scatter(dr[cr],fr[cr],color="red",zorder=4)
        plt.savefig(os.path.join(save_path,"ret_contact/{:>03}".format(i)))
        plt.close()

def plot_set_base(save_path,
                    delta_app,
                    delta_ret,
                    force_app,
                    force_ret,
                    contact,
                    ret_contact,
                    force_app_base,
                    force_ret_base):
    """base2zero in fit2ting """
    os.makedirs(os.path.join(save_path,"base_plot"),exist_ok=True)
    for i, (dr, da, fa,fr,c,cr,fab,frb) in enumerate(zip(delta_ret,
                                                            delta_app,
                                                            force_app,
                                                            force_ret,
                                                            contact,
                                                            ret_contact,
                                                            force_app_base,
                                                            force_ret_base)):
        plt.plot(da,fa,color="red")
        plt.plot(dr,fr,color="red")
        plt.plot(da,fab,color="blue")
        plt.plot(dr,frb,color="blue")

        plt.savefig(os.path.join(save_path,"base_plot/{:>03}".format(i)))
        plt.close()
    
def imshow_ax(data, ax, title):
    m = np.median(data)
    st = np.std(data)
    mapple = ax.imshow(data,vmin=np.max([0,m-2*st]),vmax=m+2*st,cmap="Greys")
    ax.set_ylabel(title, fontsize=20)
    plt.colorbar(mapple,ax=ax)
    return ax

def plot_ting_summury(save_path,map_shape, topo, young, result, e1, res ):
    """fit in fit2ting"""
    plt.close()
    fig, ax = plt.subplots(len(result.T)+4,1, figsize=(10,40))
    ax[0]=imshow_ax(topo.reshape(map_shape), ax[0], "topo")
    ax[1]=imshow_ax(np.array(young).reshape(map_shape), ax[1], "Young")
    j=2
    title = ["E0","alpha","E inf"]
    for i, (r, t) in enumerate(zip(result.T,title)):
        ax[j]=imshow_ax(r.reshape(map_shape), ax[j], t)
        j+=1
    ax[j]=imshow_ax(e1.reshape(map_shape), ax[j],"E1")
    j+=1
    try:
        res = np.array([ np.mean(r) for r in res])
        ax[j]=imshow_ax(res.reshape(map_shape), ax[j], "residuals")
    except ValueError:
        print(res.shape)
    plt.savefig(os.path.join(save_path,"summury.png"))
    plt.close()


def fitting_ting(save_path,d,f,d_fit,f_fit,fitted_result,r, resi, index):
    """fir in fit2ting"""
    if len(r)==2:
        plt.title("e0 {:>7.2} alpha {:>7.2} err : {:>7.2}".format(r[0],r[1],resi))
    else:
        plt.title("e0 {:>7.2} alpha {:>7.2} e_inf {:>7.2} err : {:>7.2}".format(float(r[0]),float(r[1]),float(r[2]),float(resi)))
    plt.plot(d,f, label = "all data")
    plt.plot(d_fit,f_fit[:len(d)], label = "fit data")
    plt.plot(d,fitted_result, label = "fitted plot")
    plt.legend()
    os.makedirs(os.path.join(save_path,"fit_img"),exist_ok=True)
    plt.savefig(os.path.join(save_path,"fit_img/{:>03}".format(index)))
    plt.close()


"""cp.py"""
def plot_contact(save_path,
                i: int,
                delta_app: np.ndarray,
                force_app: np.ndarray,
                line_fitted_data: list,
                cross_cp: list,
                cps: list):
    """
    search_cp in cp.py

    """
    save_dir = os.path.join(save_path, "plot_contact")
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))

    dr_force = (np.max(force_app) - np.min(force_app)) / 10
    frange = [np.min(force_app) - dr_force, np.max(force_app) + dr_force]
    dr_delta = (np.max(delta_app) - np.min(delta_app)) / 10
    drange = [np.min(delta_app) - dr_delta, np.max(delta_app) + dr_delta]
    ax[0].set_ylim(frange)
    ax[0].set_xlim(drange)
    ax[0].plot(delta_app, force_app, zorder=0)
    ax[0].plot(delta_app[cps[0]:cps[1]], force_app[cps[0]:cps[1]], zorder=0)

    ax[0].plot(delta_app, line_fitted_data[1] * delta_app +
                line_fitted_data[2], zorder=1, color="red")
    ax[0].plot(delta_app, cross_cp[1] * delta_app +
                cross_cp[2], zorder=1, color="green")

    ax[0].scatter(delta_app[line_fitted_data[0]],
                    force_app[line_fitted_data[0]], label="young", c="red", zorder=2)
    ax[0].scatter(delta_app[cross_cp[0]], force_app[cross_cp[0]],
                    label="cross", c="green", zorder=2)
    ax[0].legend()

    dr_force = (np.max(force_app) - np.min(force_app)) / 10
    dr_delta = (np.max(delta_app) - np.min(delta_app)) / 10
    drange = [np.min(delta_app) - dr_delta, np.max(delta_app) + dr_delta]

    ax[0].set_ylim(frange)
    frange = [0, np.max(force_app)]
    ax[1].set_ylim(frange)

    ax[1].plot(delta_app[cps[0]:], force_app[cps[0]:], zorder=0)

    ax[1].plot(delta_app[cps[0]:], line_fitted_data[1] *
                delta_app[cps[0]:] + line_fitted_data[2], zorder=1, color="red")
    # ax[1].plot(delta_app[cps[0]:],cross_cp[1]*delta_app[cps[0]:]+cross_cp[2],zorder=1, color="green")

    ax[1].scatter(delta_app[line_fitted_data[0]],
                    force_app[line_fitted_data[0]], label="young", c="red", zorder=2)
    ax[1].scatter(delta_app[cross_cp[0]], force_app[cross_cp[0]],
                    label="cross", c="green", zorder=2)
    ax[1].legend()

    plt.savefig(os.path.join(save_dir, "{:>03}".format(i)))
    plt.close()



