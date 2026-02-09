#!/usr/bin/env python
# -*- coding:utf-8 -*-

# from cmath import log
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import analysis_data_distribution as add
import galaxy_models as gm
# import action_state_samples as asa
# import transformation_some as ts
# import KDTree_python as kdtp
import fit_galaxy_wrapper as fgw
# import plot_galaxy_wrapper as pgw
import RW_data_CMGD as rdc
# import triaxialize_galaxy as tg

colors = ["red", "orange", "olive", "green", "cyan", "blue", "purple", "pink", "gray", "black"]
pointsize_preset    = 2.
fontsize_preset     = 6.

def plot_corse_grainedly_data_from_file(
    datapack, dim, xinfo, yinfo, zinfo=None, nameinfo=None, 
    plotlines=None, is_save=True, is_show=True
): #the main
    pointsize = 0.2
    fontsize = 6.0
    figsize = None
    dpi = 400
    fig = plt.figure(figsize=figsize, dpi=dpi)
    projection = None
    if dim==3:
        projection = "3d"

    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.grid(True)

    dpmin = [np.inf, np.inf, np.inf]
    dpmax = [-np.inf, -np.inf, -np.inf]
    # idx = 0 #only one group of data provided now
    for idx in np.arange(len(datapack)):
        dp = datapack[idx] #store the set({x,y,z}) of a curve
        if plotlines is not None:
            dpmin[0] = add.select_min(dpmin[0], np.min(dp[0]))
            dpmax[0] = add.select_max(dpmax[0], np.max(dp[0]))
            dpmin[1] = add.select_min(dpmin[1], np.min(dp[1]))
            dpmax[1] = add.select_max(dpmax[1], np.max(dp[1]))
            if dim==3:
                dpmin[2] = add.select_min(dpmin[2], np.min(dp[2]))
                dpmax[2] = add.select_max(dpmax[2], np.max(dp[2]))
        if dim==3:
            ax.scatter(dp[0], dp[1], dp[2], s=pointsize, label=dp[-1])
        else:
            ax.scatter(dp[0], dp[1], s=pointsize, label=dp[-1])
    
    add.DEBUG_PRINT_V(1, dpmin, dpmax, "dpminmax")
    if plotlines is not None:
        for pl in plotlines:
            if pl[0]=="x":
                if dim==3:
                    ax.plot([pl[1], pl[1]], [dpmin[1], dpmax[1]], [dpmin[2], dpmax[2]], lw=pointsize*3., label=None, color="k")
                else:
                    ax.plot([pl[1], pl[1]], [dpmin[1], dpmax[1]], lw=pointsize*3., label=None, color="k")
            if pl[0]=="y":
                if dim==3:
                    ax.plot([dpmin[0], dpmax[0]], [pl[1], pl[1]], [dpmin[2], dpmax[2]], lw=pointsize*3., label=None, color="k")
                else:
                    ax.plot([dpmin[0], dpmax[0]], [pl[1], pl[1]], lw=pointsize*3., label=None, color="k")
            if pl[0]=="z":
                if dim==3:
                    ax.plot([dpmin[0], dpmax[0]], [dpmin[1], dpmax[1]], [pl[1], pl[1]], lw=pointsize*3., label=None, color="k")

    if xinfo[0] is not None:
        ax.set_xlim(xinfo[0])
    if yinfo[0] is not None:
        ax.set_ylim(xinfo[0])
    if dim==3:
        if zinfo[0] is not None:
            ax.set_zlim(xinfo[0])
    ax.set_xlabel(xinfo[-1], fontsize=fontsize)
    ax.set_ylabel(yinfo[-1], fontsize=fontsize)
    ax.tick_params(labelsize=fontsize/2.) #size of the number characters
    # to set logscale expect for z ??
    if dim==3:
        ax.set_zlabel(zinfo[-1], fontsize=fontsize)

    pathname    = "./"
    figurename  = "figurename.png"
    textname    = "sometexts"
    if nameinfo is not None:
        pathname    = nameinfo[0]
        figurename  = nameinfo[1]
        textname    = nameinfo[2]
    plt.suptitle(figurename+"\n"+textname, fontsize=fontsize)
    # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc=0)

    fig_tmp = plt.gcf()
    if is_save:
        fig_tmp.savefig("savefig/"+pathname+"/"+figurename+".png", format="png", dpi=dpi, bbox_inches='tight')
    if is_show==True:
        plt.show()
    plt.close("all")
    return 0

def plot_parameters_mg_list(pr, label_list=None, sn="", is_show=True): #xplot ??
    N_pr = len(pr)
    plt.figure(dpi=300)
    for i in np.arange(N_pr):
        plt.subplot(add.integer_dividing_ceil_int(N_pr, 4), 4, i+1)
        lb = None
        if label_list is not None:
            lb = label_list[i]
        yplot = pr[i]
        xplot = np.arange(0, len(yplot))
        print("pr[%d]: "%(i), yplot, len(yplot))
        plt.grid()
        plt.scatter(xplot, yplot, label=lb, s=pointsize_preset)
        plt.plot(xplot, yplot, label=lb, lw=pointsize_preset/5)
        plt.legend(fontsize=fontsize_preset)
    plt.suptitle(sn)
    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/compare_mg_parameters"+sn+".png", format="png")
    if is_show:
        plt.show()
    plt.close("all")
    return 0

def plot_1d_DF_by_bins(aa, N_bins=100, is_show=True):
    n,m = np.shape(aa)
    binsxy = list(range(m+1))
    for i in np.arange(m+1):
        plt.subplot(m+1,1,i+1)
        hx = None
        lb = None
        if i==m:
            hx = np.sum(aa, axis=1)
            lb = "DF of total columns"
        else:
            hx = aa[:,i]
            lb = "DF of column %d (index is from 0)"%(i)
        # add.DEBUG_PRINT_V(1, np.min(hx), np.max(hx), np.shape(hx), "hx")
        # xbins = add.logspace_1d_by_boundaries(np.min(hx), np.max(hx), N_bins+1)
        xbins = np.linspace(np.min(hx), np.max(hx), N_bins+1)
        h, xedge = np.histogram(hx, bins=xbins)
        xbins_DF = xbins[:-1]
        h_DF = h/(xbins[1:]-xbins[:-1])
        plt.step(xbins_DF, h_DF, label=lb, where='post')
        # plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        binsxy[i] = [xbins_DF, h_DF]
    if is_show:
        plt.show()
    plt.close("all")
    return binsxy

class Plot_model_fit:
    def __init__(self):
        # plot options
        self.is_plotlog = 0
        self.color = ["purple", "blue", "green",
                      "orange", "red"]  # color value
        self.figsize = (20, 16)
        self.dpi = 100
        self.pointsize = 2.
        self.fontsize = 30.
        self.axis_scale = "log"
        self.title = ""
        # declaration
        self.x = 0.
        self.y = 0.
        self.xerr = 0.
        self.yerr = 0.
        self.funcfit = 0.
        self.params = 0.
        self.standard = 0.
        self.ll = 0.
        self.datalist = 0.
        self.fig_all = 0.

    def load(self, *datalist):
        self.ll = len(datalist)
        self.datalist = datalist

    def set_figure(self, **args):
        pass

    def plot(self, name="", xl="", yl="", text="", is_relative=0, id_relative_compare=1, dpi=300, is_show=0):
        self.xlabel = xl
        self.ylabel = yl
        self.fig_all = plt.figure(figsize=self.figsize, dpi=self.dpi)
        l_index = 0
        compare_funcfit = self.datalist[id_relative_compare][6]
        compare_x = self.datalist[id_relative_compare][0]
        compare_params = self.datalist[id_relative_compare][7]
        compare = compare_funcfit(compare_x, *compare_params)
        for l in self.datalist:
            # load each group of data
            x = l[0]
            y = l[1]
            xerr = l[2]
            yerr = l[3]
            xp = l[4]
            yp = l[5]
            funcfit = l[6]
            params = l[7]
            others = l[8]
            label = l[9]
            # plot each
            # plt.subplot(self.ll,l)
            ym = funcfit(x, *params)
            ym_plot = ym
            y_plot = y
            self.standard = [add.norm_l(ym, y, axis=0) / len(y)]
            # self.standard = others
            # if l_index == id_relative_compare:
            #     compare = ym
            if is_relative == 1:
                ym_plot = ym/compare-1.
                y_plot = y/compare-1.
            if l_index == 0:
                plt.scatter(xp, y_plot, s=self.pointsize,
                            color="k", label="data generated by DICE\n"+text)
            plt.scatter(xp, ym_plot, s=self.pointsize,
                        label=label+", ye=%e" % (self.standard[0]))
            l_index += 1
        # plt.xscale(self.axis_scale)
        # plt.yscale(self.axis_scale)
        plt.rcParams["legend.markerscale"] = self.pointsize*10.
        plt.tick_params(labelsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        plt.xlabel(self.xlabel, fontsize=self.fontsize)
        plt.ylabel(self.ylabel, fontsize=self.fontsize)
        if is_relative != 1:
            plt.xlim(0., 250.) #log
            plt.ylim(-30., 0.)
        plt.title(self.title, fontsize=self.fontsize)
        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/funcfit/"+name+"_compare_isrelative%d.png" %
                        (is_relative), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        plt.close("all")
        print("Fig ... Done.")

    def plot3d(self):
        pass

    def plot_scatter_2d_or_3d_with_color(self, 
        plot_list, f_list=None, label_list=None, 
        dim=3, xyzlogscale=None, xyzlim=None, xyztitle=None, scalevalue=None, 
        view_angles=None,
        pathname="", figurename="", textname="", is_save=False, is_show=True
    ): #the main
        pointsize = 0.2
        fontsize = 3.0
        figsize = None
        dpi = 400
        print("Start to plot...")
        fig = plt.figure(figsize=figsize, dpi=dpi)
        projection = None
        if dim==3:
            projection = "3d"

        lx = len(plot_list)
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.grid(True)
        if view_angles is not None:
            ax.view_init(elev=view_angles[0], 
                azim=view_angles[1])
        # ax.set_axis_off()
        for i in range(lx):
            f = None
            cm = None
            if f_list is not None:
                f = f_list[i]
                cm = plt.cm.get_cmap("gist_rainbow") #rainbow

            lb = None
            if label_list is not None:
                lb = label_list[i]

            curve_one = plot_list[i]
            x = curve_one[0]
            y = curve_one[1]
            if dim==3:
                z = curve_one[2]
                if xyzlogscale is not None:
                    if xyzlogscale[0]:
                        x = np.log10(x)
                    if xyzlogscale[1]:
                        y = np.log10(y)
                    if xyzlogscale[2]:
                        z = np.log10(z)
                axsc = ax.scatter(x, y, z, s=pointsize, label=lb, c=f, cmap=cm)
            else:
                axsc = ax.scatter(x, y, s=pointsize, label=lb, c=f, cmap=cm)
            if f_list is not None:
                plt.colorbar(axsc)

        # if dim==3:
        #     if scalevalue[0] is not None:
        #         ax.scatter([scalevalue[0]], [0.], [0.], s=pointsize, color="k")
        #     if scalevalue[1] is not None:
        #         ax.scatter([0.], [scalevalue[0]], [0.], s=pointsize, color="k")
        #     if scalevalue[2] is not None:
        #         ax.scatter([0.], [0.], [scalevalue[0]], s=pointsize, color="k")
        # else:
        #     if scalevalue[0] is not None:
        #         ax.scatter([scalevalue[0]], [0.], s=pointsize, color="k")
        #     if scalevalue[1] is not None:
        #         ax.scatter([0.], [scalevalue[0]], s=pointsize, color="k")

        if (xyzlogscale is not None) and (dim!=3): #?? wrong of plot 3d logscale
            if xyzlogscale[0]:
                ax.set_xscale("log")
            if xyzlogscale[1]:
                ax.set_yscale("log")
            if dim==3:
                if xyzlogscale[2]:
                    ax.set_zscale("log")

        if xyzlim is not None:
            if xyzlim[0] is not None:
                ax.set_xlim(xyzlim[0][0], xyzlim[0][1])
            if xyzlim[1] is not None:
                ax.set_ylim(xyzlim[1][0], xyzlim[1][1])
            if dim==3:
                if xyzlim[2] is not None:
                    ax.set_zlim(xyzlim[2][0], xyzlim[2][1])

        if xyztitle is not None:
            ax.set_xlabel(r"%s"%(xyztitle[0]), fontsize=fontsize)
            ax.set_ylabel(r"%s"%(xyztitle[1]), fontsize=fontsize)
            if dim==3:
                ax.set_zlabel(r"%s"%(xyztitle[2]), fontsize=fontsize)
        else:
            ax.set_xlabel(r"coordinate1", fontsize=fontsize)
            ax.set_ylabel(r"coordinate2", fontsize=fontsize)
            if dim==3:
                ax.set_zlabel(r"coordinate3", fontsize=fontsize)
        
        plt.tick_params(labelsize=fontsize)
        plt.suptitle(figurename+"\n"+textname, fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        ax.legend(fontsize=fontsize, loc=0)

        fig_tmp = plt.gcf()
        if is_save:
            fig_tmp.savefig("savefig/"+pathname+"/"+figurename+".png", format="png", dpi=dpi, bbox_inches='tight')
        if is_show==True:
            plt.show()
        plt.close("all")

        return 0 #function plot_scatter_2d_or_3d_with_color() end

    def plot_actions_Comb_NDF(self, ddl, nm="", text="", lim=[False], is_show=1):

        pointsize = 0.06
        fontsize = 6.0
        dpi = 500
        fig = plt.figure(dpi=300)
        # ax=fig.add_subplot(111, projection='3d') # ax = Axes3D(fig)
        ax=fig.add_subplot(111)
        ax.grid(True) # ax.set_axis_off()

        for i in range(len(ddl)):
            x = ddl[i][0]
            y = ddl[i][1]
            lb = ddl[i][-1]
            ax.scatter(x, y, s=pointsize, label=lb, marker="+")
        ax.legend(fontsize=fontsize, loc=0)

        ax.set_xlabel(r"x", fontsize=fontsize)
        ax.set_ylabel(r"y", fontsize=fontsize)
        ax.set_title(r"actions_Comb_NDF_"+"%s"%(text), fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        
        if(lim[0]):
            ax.set_xlim(lim[1], lim[2])
            ax.set_ylim(lim[3], lim[4])

        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/actions_Comb_NDF_%s.png"%(nm), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        plt.close("all")
        print("Fig ... Done.")
        return 

    def plot_actions_Comb_NDF_subplot(self, datapack, 
        xl=[], yl=[], lim=None, bd_much=None, 
        suppertitle="", savename="plt_NDF", is_show=True
    ):
        pointsize = 0.6
        fontsize = 12.0
        # dpi = dpi
        # fig = plt.figure()
        fig = plt.figure(figsize=(16,10), dpi=200)
        
        #: each fitmodel j is out of this supplot, j->l->i->k
        for l in range(len(datapack)): #type subplot
            datapack_subplot = datapack[l]
            ax = fig.add_subplot(int(np.ceil(len(datapack)/2.)), 2, l+1)
            ax.grid(True)

            for i in range(len(datapack_subplot)): #each combmodel i
                ddcomb = datapack_subplot[i]
                # add.DEBUG_PRINT_V(1, ddcomb, "ddcomb")
                for k in range(len(ddcomb)): #ydata or yfit
                    fitcompare = ddcomb[k]
                    x = fitcompare[0] #xdata
                    y = fitcompare[1] #ydata
                    lb = fitcompare[2] #labels
                    plottag = fitcompare[-1]
                    ax.scatter(x, y, s=pointsize, label=lb, marker="+")
                    # ax.scatter(x, y, s=pointsize, label=lb, color=colors[i+1], marker="+")
                    
                    #debug: main range
                    if bd_much is not None:
                        bd_y = [np.min(y), np.max(y)]
                        ax.plot([bd_much[0], bd_much[0]], bd_y, lw=pointsize, color="k")
                        ax.plot([bd_much[1], bd_much[1]], bd_y, lw=pointsize, color="k")
            
            ax.legend(fontsize=fontsize/2)#, loc=0)
            ax.set_xscale("log")
            ax.set_xlabel(xl[l], fontsize=fontsize)
            ax.set_ylabel(yl[l], fontsize=fontsize)
            # ax.set_title(r"actions_Comb_NDF_"+"%s"%(text), fontsize=fontsize)
            # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
            if lim is not None:
                if l==1:
                    plt.ylim(-6., 6.)
                else:
                    # plt.xlim(lim[0][0], lim[0][1])
                    plt.ylim(lim[1][0], lim[1][1])
            
        plt.suptitle(suppertitle)
        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/%s.png"%(savename), format='png', bbox_inches='tight')
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")
        return 

    def plot_x_scatter3d_dd(self, ddl, nm="", text="", is_lim=False, bd=None, k_median=0., is_show=1):

        fig = plt.figure(dpi=300)
        pointsize = 0.2
        fontsize = 6.0
        dpi = 500
        ax=fig.add_subplot(111,projection='3d') # ax = Axes3D(fig)
        ax.grid(True) # ax.set_axis_off()

        for i in range(len(ddl)):
            x = ddl[i][0]
            f = ddl[i][1]
            lb = ddl[i][-1]
            # cm = plt.cm.get_cmap('RdYlBu') #red-yellow-blue
            cm = plt.cm.get_cmap('gist_rainbow') #rainbow
            axsc = ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb, c=f, cmap=cm)
            plt.colorbar(axsc)
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
        # position=fig.add_axes([0.1, 0.3, 0.02, 0.5]) #x pos, ypos, width, shrink
        # fig.colorbar(surf, cax=position, aspect=5)
        # # ax.arrow(-lim,0, 2*lim,0) #only 2d??
        # ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=pointsize)
        # ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=pointsize)
        # ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=pointsize)
        ax.legend(fontsize=fontsize, loc=0)

        ax.set_xlabel(r"x", fontsize=fontsize)
        ax.set_ylabel(r"y", fontsize=fontsize)
        ax.set_zlabel(r"z", fontsize=fontsize)
        ax.set_title(r"scatter3d_dd_"+"%s"%(text), fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        
        if is_lim:
            if bd!=None:
                ax.set_xlim(0., bd[0])
                ax.set_ylim(0., bd[1])
                ax.set_zlim(0., bd[2])
            else:
                # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.6, 1., 0.6, 1.]))
                # klim = 0.9
                # lim=200.
                # ax.set_xlim(-lim, lim)
                # ax.set_ylim(-lim, lim)
                # ax.set_zlim(-lim, lim)
                klim = k_median
                lim = np.median(x,axis=0)*klim
                minx = np.min(x,axis=0)
                # ax.set_xlim(-lim[0]*0., lim[0])
                # ax.set_ylim(-lim[1]*0., lim[1])
                # ax.set_zlim(-lim[2]*0., lim[2])
                ax.set_xlim(minx[0], lim[0])
                ax.set_ylim(minx[1], lim[1])
                # ax.set_zlim(minx[2], lim[2])

        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_zscale("log")

        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/scatter3d_dd_action_%s.png"%(nm), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")

    def plot_histogram(self, x_list, bins_list=None, label_list=None, 
        name="", is_log=False, limit_list=None, xyzlabel_list=None, is_save=False
    ):
        pointsize = 0.2
        fontsize = 6.0
        dpi = 400
        fig = plt.figure()
        for j in range(len(x_list)):
            ax = fig.add_subplot(len(x_list),1,j+1)
            for i in range(len(x_list[j])):
                x = x_list[j][i]
                xbins = np.linspace(-np.min(x), np.max(x), 20)
                if bins_list!=None:
                    xbins = bins_list[j][i]
                lb = None
                if label_list!=None:
                    lb = label_list[j][i]
                h, xedge = np.histogram(x, bins=xbins)
                ax.step(xbins[:-1], h, label=lb, where='post')

            text = r"scatter3d_"+"%s"%(name)
            ax.legend(fontsize=fontsize, loc=0)
            ax.set_title(text, fontsize=fontsize)
            # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)

        fig_tmp = plt.gcf()
        if is_save:
            fig_tmp.savefig("savefig/x_scatter/"+text+".png", format='png', dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close("all")
        return 

    def plot_x_scatter3d_general(self, x_list, f_list=None, label_list=None, 
        name="", is_log=False, limit_list=None, xyzlabel_list=None, is_save=False
    ):
        pointsize = 0.2
        fontsize = 6.0
        dpi = 400
        fig = plt.figure(figsize=None, dpi=dpi)
        ax=fig.add_subplot(1,1,1, projection='3d')
        ax.grid(True) #ax.set_axis_off()

        for i in range(len(x_list)):
            x = x_list[i]
            lb = None
            if label_list!=None:
                lb = label_list[i]
            if f_list!=None:
                f = f_list[i]
                # f = f_list[-1]
                cm = plt.cm.get_cmap('gist_rainbow') #rainbow
                axsc = ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb, c=f, cmap=cm)
                plt.colorbar(axsc)
            else:
                axsc = ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb)
            
        if limit_list!=None:
            ax.set_xlim(limit_list[0], limit_list[1])
            ax.set_ylim(limit_list[2], limit_list[3])
            ax.set_zlim(limit_list[4], limit_list[5])
            ax.plot([limit_list[0], limit_list[1]], [0,0], [0,0], lw=pointsize*2, color="black")
            ax.plot([0,0], [limit_list[2], limit_list[3]], [0,0], lw=pointsize*2, color="black")
            ax.plot([0,0], [0,0], [limit_list[4], limit_list[5]], lw=pointsize*2, color="black")

        if xyzlabel_list!=None:
            ax.set_xlabel(r"%s"%(xyzlabel_list[0]), fontsize=fontsize)
            ax.set_ylabel(r"%s"%(xyzlabel_list[1]), fontsize=fontsize)
            ax.set_zlabel(r"%s"%(xyzlabel_list[2]), fontsize=fontsize)
        else:
            ax.set_xlabel(r"coordinate1", fontsize=fontsize)
            ax.set_ylabel(r"coordinate2", fontsize=fontsize)
            ax.set_zlabel(r"coordinate3", fontsize=fontsize)

        if is_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_zscale("log")

        text = r"scatter3d_"+"%s"%(name)
        ax.legend(fontsize=fontsize, loc=0)
        ax.set_title(text, fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)

        fig_tmp = plt.gcf()
        if is_save:
            fig_tmp.savefig("savefig/x_scatter/"+text+".png", format='png', dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close("all")

    def plot_x_scatter3d_xxx(self, ddl, nm="", text="", is_lim=False, k_median=8., is_show=1):

        fig = plt.figure(dpi=300)
        pointsize = 0.2
        fontsize = 6.0
        dpi = 500
        ax=fig.add_subplot(111,projection='3d') # ax = Axes3D(fig)
        ax.grid(True) # ax.set_axis_off()

        for i in range(len(ddl)):
            x = ddl[i][0]
            lb = ddl[i][-1]
            # cm = plt.cm.get_cmap('gist_rainbow') #rainbow
            # ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb, c=f, cmap=cm)
            # plt.colorbar(axsc)
            ax.scatter(x[:,0], x[:,1], x[:,2], s=pointsize, label=lb)
        ax.legend(fontsize=fontsize, loc=0)

        name = r"scatter3d_xxx_"+"%s"%(text)
        ax.set_xlabel(r"x", fontsize=fontsize)
        ax.set_ylabel(r"y", fontsize=fontsize)
        ax.set_zlabel(r"z", fontsize=fontsize)
        ax.set_title(name, fontsize=fontsize)
        # ax.text3D(-10.,-10.,-10., r'O', fontsize=fontsize)
        
        if is_lim:
            klim = k_median
            lim = np.median(x,axis=0)*klim
            minx = np.min(x,axis=0)
            # ax.set_xlim(-lim[0]*0., lim[0])
            # ax.set_ylim(-lim[1]*0., lim[1])
            # ax.set_zlim(-lim[2]*0., lim[2])
            ax.set_xlim(minx[0], lim[0])
            ax.set_ylim(minx[1], lim[1])
            # ax.set_zlim(minx[2], lim[2])

        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_zscale("log")

        fig_tmp = plt.gcf()
        fig_tmp.savefig("savefig/scatter3d_xxx_action_%s.png"%(nm), format='png', dpi=dpi, bbox_inches='tight')
        if is_show == 1:
            plt.show()
        print("Fig ... Done.")
        plt.close("all")



def plot_x_scatter(x, projection_surface="xy", lim=150., name="", text="", dpi=300, is_show=0):
    # fig, ax = plt.figure(dpi=dpi) #, facecolor=(0.0, 0.0, 0.0))
    fig = plt.figure(dpi=dpi) #, facecolor=(0.0, 0.0, 0.0))
    fontsize = 10.
    pointsize = 0.16
    k = 0.9
    plt.plot([-lim*k,lim*k],[0.,0.], color="k", linewidth=pointsize)
    plt.plot([0.,0.],[-lim*k,lim*k], color="k", linewidth=pointsize)
    if projection_surface=="xz":
        plt.scatter(x[:, 0], x[:, 1], s=pointsize, label=text)
    elif projection_surface=="yz":
        plt.scatter(x[:, 1], x[:, 2], s=pointsize, label=text)
    else: #xy
        plt.scatter(x[:, 0], x[:, 1], s=pointsize, label=text)
    plt.legend(fontsize=fontsize, loc=2)
    plt.rcParams["legend.markerscale"] = pointsize*10.
    plt.tick_params(labelsize=fontsize)
    # plt.xscale("log")
    # plt.yscale("log")
    plt.axis('scaled')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    # plt.text([x0,y0,z0],"the text")
    plt.xlabel(r"$x(\mathrm{kpc})$", fontsize=fontsize)
    plt.ylabel(r"$y(\mathrm{kpc})$", fontsize=fontsize)
    plt.title(r"position projection of particles on $z=0$", fontsize=fontsize)
    # plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0., hspace=0, wspace=0)
    # plt.margins(0, 0)
    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/x_scatter/"+name+".png", format='png', dpi=dpi, bbox_inches='tight')
    if is_show == 1:
        plt.show()
    plt.close()

def plot_x_scatter3d(x, projection_surface="xy", lim=200., name="", text="", dpi=300, is_show=0):
    # fig, ax = plt.figure(dpi=dpi) #, facecolor=(0.0, 0.0, 0.0))
    fig = plt.figure(dpi=dpi) #, facecolor=(0.0, 0.0, 0.0))
    pointsize = 0.2
    fontsize = 20.0
    k = 0.9
    ax = Axes3D(fig)
    ax.grid(True)
    # ax.set_axis_off() #remove all relevent axis
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.scatter(x[:,0], x[:,1], x[:,2], color="blue", s=pointsize)
    # ax.arrow(-lim,0, 2*lim,0) #only 2d??
    ax.plot3D([-lim,lim],[0,0],[0,0], color="red", linewidth=pointsize)
    ax.plot3D([0,0],[-lim,lim],[0,0], color="red", linewidth=pointsize)
    ax.plot3D([0,0],[0,0],[-lim,lim], color="red", linewidth=pointsize)
    ax.set_xlabel(r'$x\,\, (\mathrm{kpc})$', fontsize=fontsize)
    ax.set_ylabel(r'$y\,\, (\mathrm{kpc})$', fontsize=fontsize)
    ax.set_zlabel(r'$z\,\, (\mathrm{kpc})$', fontsize=fontsize)
    ax.set_title(r"position projection of particles on $z=0$", fontsize=fontsize)
    # ax.text3D(-10.,-10.,-10., r'O', fontsize=20)
    ax.legend(fontsize=fontsize, loc=2)

    # plt.rcParams["legend.markerscale"] = pointsize*10.
    # plt.tick_params(labelsize=fontsize)
    # # plt.xscale("log")
    # # plt.yscale("log")
    # plt.axis('scaled')
    # # plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0., hspace=0, wspace=0)
    # # plt.margins(0, 0)

    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/x_scatter/"+name+".png", format='png', dpi=dpi, bbox_inches='tight')
    if is_show == 1:
        plt.show()
    print("Fig ... Done.")
    plt.close("all")



def plot_one_snapshot(snapshot_Id, galaxymodel_name, MG, WF, is_fit=1, projection_surface="xy", is_show=0):

    print("snapshot_%03d: " % (snapshot_Id))
    print("\n\n\n")
    MG = MG
    WF = WF
    name = "rho_snapshot_"+"%03d" % (snapshot_Id)

    RD = rdc.Read_data_galaxy(MG, 2, gmn=galaxymodel_name)
    doc = RD.data_secondhand_snapshot(snapshot_Id)
    x, y = RD.data_sample_screen(doc, is_logy=True)
    xr, yr = RD.data_sample_combination(x, y, "radius")
    xplot, yplot = xr, yr
    xerr = x*0.
    yerr = y*0.1
    dos0 = RD.data_original_simulationoOutput(snapshot_Id)
    xv = dos0[:,0:6]

    if is_fit==0:
        # plot_x_scatter(xv, projection_surface=projection_surface, name=name, text=name, is_show=is_show)
        plot_x_scatter3d(xv, projection_surface=projection_surface, name=name, text=name, is_show=is_show)
        return 0, 0, 0

    P_DG = np.array(WF.reference_list)

    P_MC, E_MC = [0.9401185520037684, 3.211754159203118, 0.0008978270962895497, 
                  23.894278631802607, -54.143804306504165], [0.0026085736587127986]
    MF = fgw.MCMC_fit_galaxy(WF, WF.N_paramsfit, x, y, yerr, nw=60, ns=500,
                         nb=400, nth=1, nd=10, sd=250)
    # MF = MCMC_fit_galaxy(WF, nf, x, y, yerr, nw=60, ns=5000,
    #                      nb=4000, nth=10, nd=1000, sd=250)
    MF.run()
    P_MC, E_MC = MF.display(name=name, is_show=0)
    P_MC = P_MC[0:-1]

    CF = fgw.Curve_fit_galaxy(WF, WF.nf, x, y, yerr, tag0=0, mf=5000)
    add.DEBUG_PRINT_V(1, "before CF.run()")
    CF.run()
    add.DEBUG_PRINT_V(1, "after CF.run()")
    P_CF, E_CF = CF.display()

    dl_DG = [x, y, xerr, yerr, xplot, yplot, WF.funcfit, P_DG, 0., "expected function"]
    dl_MC = [x, y, xerr, yerr, xplot, yplot, WF.funcfit, P_MC, E_MC, "fit by MCMC"]
    dl_CF = [x, y, xerr, yerr, xplot, yplot,
             WF.funcfit, P_CF, E_CF, "fit by curve_fit"]
    dl = [dl_DG, dl_CF, dl_MC]  # dl_DG

    id_relative_compare = 1
    xl = "xlabel"
    yl = "ylabel"
    text = "text"
    # xl = r"radius: $r (\mathrm{kpc})$"
    # yl = r"logrithmic mass density: $\log(\frac{\rho(r)}{\mathrm{1e10\,\, M_{sun}\,\, kpc^-3}})\,\, (\times 1)$"
    # text = name+", the curve_fit result: \n"\
    #     +r"$n_\mathrm{Einasto}=%e$, "%(P_CF[0])+"\n"\
    #     +r"$\rho_0=%e$, "%(P_CF[1])+"\n"\
    #     +r"$r_0=%e$"%(P_CF[2]) #"snapshot_%03d"%(snapshot_Id)
    PLOT = Plot_model_fit()
    PLOT.load(*dl)
    PLOT.plot(name=name, xl=xl, yl=yl, text=text, is_relative=0, is_show=0)
    # PLOT.plot(name=name, xl=xl, yl=r"relative error of "+yl, is_relative=1,
    #           id_relative_compare=id_relative_compare, is_show=0)

    # plot_x_scatter(xv, projection_surface=projection_surface, name=name, text=text, is_show=0)
    plot_x_scatter3d(xv, projection_surface=projection_surface, name=name, text=name, is_show=is_show)
    m = [ np.median(abs(xv), axis=0)]
    return P_CF, E_CF, m

def plot_params_t(snapshot_sequence, fit_s_p, pe, gmn, params_name="", dpi=300, is_show=0, *pother):
    if not len(fit_s_p==2):
        exit(0)
    ns, nparams = fit_s_p.shape
    savedata = np.hstack((np.array([snapshot_sequence]).T, fit_s_p))
    savepath = "savefig/params_t_"+gmn+".txt"
    np.savetxt(savepath, savedata, fmt="%e")
    print("Save data to %s" %(savepath))

    fig = plt.figure(dpi=dpi)
    fontsize = 10.
    # pointsize = 0.16
    one = np.ones(ns)
    plt.plot(snapshot_sequence, one, color="k")
    for i in np.arange(nparams):
        plt.plot(snapshot_sequence, fit_s_p[:,i]/pe[i], \
            label="("+params_name[i]+")/(%e Unit)"%(pe[i]))
    plt.legend(fontsize=fontsize)
    # plt.rcParams["legend.markerscale"] = pointsize*10.
    plt.tick_params(labelsize=fontsize)
    plt.xlabel("snapshot Id (100 snapshot ~ 1 Gyr)", fontsize=fontsize)
    plt.ylabel("fit/expected rate", fontsize=fontsize)
    plt.legend()

    fig_tmp = plt.gcf()
    fig_tmp.savefig("savefig/params_t_"+gmn+".png", format='png', dpi=dpi, bbox_inches='tight')
    if is_show == 1:
        plt.show()
    plt.close()



if __name__ == '__main__':

    # filename = "/home/darkgaia/workroom/0prog/GroArnold_framework/GDDFAA/"+\
    #     "step3_actions/step2_had_all_AGAMA/Agama-master/exe/data/lyapunov.txt"
    # data = np.loadtxt(filename) #Julia M ??
    # xcoor = data[:,0:3]
    # vcoor = data[:,3:6]
    # r = add.norm_l(xcoor, axis=1)
    # vr = add.norm_l(vcoor, axis=1)
    # lyap = data[:,6]
    # add.DEBUG_PRINT_V(1, data.shape, r.shape, lyap.shape)
    
    # plot_dim = 2
    # datapack = [[r, lyap, "label"]]
    # xinfo = [None, None, "radius (unit kpc??)"]
    # yinfo = [None, None, "lyapunov_ord6Alg (unit One)"]
    # nameinfo = ["./", "r~l", "sometexts"]
    # plot_corse_grainedly_data_from_file(datapack, plot_dim, xinfo, yinfo, nameinfo=nameinfo)
    
    # plot_dim = 2
    # datapack = [[vr, lyap, "label"]]
    # xinfo = [None, None, "velocity_norm (unit kpc??)"]
    # yinfo = [None, None, "lyapunov_ord6Alg (unit One)"]
    # nameinfo = ["./", "vr~l", "sometexts"]
    # plot_corse_grainedly_data_from_file(datapack, plot_dim, xinfo, yinfo, nameinfo=nameinfo)
    
    # plot_dim = 3
    # datapack = [[r, vr, lyap, "label"]]
    # xinfo = [None, None, "radius (unit kpc??)"]
    # yinfo = [None, None, "velocity_norm (unit kpc??)"]
    # zinfo = [None, None, "lyapunov_ord6Alg (unit One)"]
    # nameinfo = ["./", "r~vr~l", "sometexts"]
    # plot_corse_grainedly_data_from_file(datapack, plot_dim, xinfo, yinfo, zinfo, nameinfo=nameinfo)
    

    '''
    filename = "/home/darkgaia/workroom/0prog/GroArnold_framework/GDDFAA/"\
        "step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general/aa/snapshot_90.action.method_all.txt"
    RG = rdc.Read_galaxy_data(filename)
    RG.AAAA_set_particle_variables(
        col_particle_IDs=7-1, col_particle_mass=8-1
    )

    data = RG.data
    mass = RG.particle_mass
    IDs = RG.particle_IDs
    Dim = gm.Dim #3
    iast = 28
    adur = 10
    AA_TF_FP = data[:, iast+adur*0:iast+adur*0+adur]
    AA_OD_FP = data[:, iast+adur*1:iast+adur*1+adur] #none
    AA_GF_FP = data[:, iast+adur*2:iast+adur*2+adur] #none
    iast += adur*5 # = 78
    AA_TF_DP = data[:, iast+adur*0:iast+adur*0+adur]
    AA_OD_DP = data[:, iast+adur*1:iast+adur*1+adur]
    AA_GF_DP = data[:, iast+adur*2:iast+adur*2+adur] #none

    AA_method = AA_TF_DP
    Act = AA_method[:, 0:3]
    Ang = AA_method[:, 3+1:7]
    Fre = AA_method[:, 7:10]
    add.DEBUG_PRINT_V(1, AA_TF_DP.shape, Act.shape, Fre.shape)
    AA = np.hstack((Act, Fre))
    cols = [0,1,2]
    # bd = 2e2
    bd = 2e5
    AA_cl, cl, cln = add.screen_boundary_some_cols(AA, cols, 1./bd, bd, value_discard=bd*1e4)

    x = data[:,0:3]
    v = data[:,3:6]
    add.DEBUG_PRINT_V(1, np.median(x[cl], axis=1), np.median(add.norm_l(x[cl], axis=1)), "rclmed")
    import triaxialize_galaxy as tg
    L = tg.angularMoment(x, v)
    lnorm = add.norm_l(L, axis=1)
    potd = data[:,11]
    # aa = AA_cl
    aa = np.abs(AA_cl)
    a1 = aa[:,0]
    a2 = aa[:,1]
    a3 = aa[:,2]
    fq1 = aa[:,3]
    fq2 = aa[:,4]
    fq3 = aa[:,5]

    energytransl1 = (0.5*add.norm_l(v,axis=1)**2 + potd)[cl]
    energytransl = (energytransl1 + np.max(np.abs(potd)))
    lnorm = lnorm[cl]
    ac1 = gm.AA_combination_sum(aa)
    # actioncomb = gm.AA_combination_sumWeightFrequency(aa)
    actioncomb = gm.AA_combination_sumWeightFrequency_rateF1(aa)
    add.DEBUG_PRINT_V(1, np.shape(energytransl), np.shape(lnorm), np.shape(actioncomb), "var shape")
    add.DEBUG_PRINT_V(1, np.min(energytransl), np.min(actioncomb), "var min")
    add.DEBUG_PRINT_V(1, np.median(energytransl1), np.median(energytransl), np.median(fq1), np.median(ac1), np.median(actioncomb), "var median")

    plot_dim = 2
    # is_logP1_x = True
    is_logP1_x = False
    # is_logP1 = True
    is_logP1 = False
    added = 0.
    # added = 1.
    datapack = [
        # # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(potd, is_logP1=is_logP1),       "pord"], #meds
        # # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(lnorm, is_logP1=is_logP1),      "lnorm"], 
        # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(a3, is_logP1=is_logP1),         "a3"], 
        # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(a2, is_logP1=is_logP1),         "a2"], 
        # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(a1, is_logP1=is_logP1),         "a1"], 
        # # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(fq1, is_logP1=is_logP1),        "fq1"], 
        # # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(fq2, is_logP1=is_logP1),        "fq2"], 
        # # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(fq3, is_logP1=is_logP1),        "fq3"], 
        # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(ac1, is_logP1=is_logP1),        "ac1"], 
        # [add.frac_to_median(energytransl, is_logP1=is_logP1_x), add.frac_to_median(actioncomb, is_logP1=is_logP1), "actioncomb"], 
        # [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(potd, added=added),       "pord"], #pers
        [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(lnorm, added=added),      "lnorm"], 
        [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(a3, added=added),         "a3"], 
        [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(a2, added=added),         "a2"], 
        [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(a1, added=added),         "a1"], 
        # [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(fq1, added=added),        "fq1"], 
        # [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(fq2, added=added),        "fq2"], 
        # [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(fq3, added=added),        "fq3"], 
        [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(ac1, added=added),        "ac1"], 
        [add.log_abs_P1(-energytransl1, added=added), add.log_abs_P1(actioncomb, added=added),        "actioncomb1"]
        # [add.log_abs_P1(energytransl, added=added), add.log_abs_P1(actioncomb, added=added),        "actioncomb"]
    ]
    # ylim = [0., 10.]
    ylim = None
    xinfo = [None, None, "ln(-E) (unit unit)"]
    yinfo = [ylim, None, "ln(J_comb) (unit unit)"]
    nameinfo = ["./", "sometitle", "with percentile lines at [0.5, 20., 50., 80., 99.5]"]
    pers = [0.5, 20., 50., 80., 99.5]
    logbasis = np.exp(1.)
    # logbasis = 10.
    xs = add.log_abs_P1(-energytransl1, logbasis=logbasis, added=added)
    ys = add.log_abs_P1(actioncomb, logbasis=logbasis, added=added)
    add.DEBUG_PRINT_V(1, np.median(xs), np.median(ys), "var median2")
    import scipy.optimize as sciopt
    funcfit = gm.linear_curve1d_space1d
    opt, cov = sciopt.curve_fit(funcfit, ys, xs, p0=[1., 1.])
    print("lin1d1d opt: ", opt)
    
    fv = funcfit(ys, *opt)
    fv2 = np.log(np.exp(ys)**opt[0]*np.exp(opt[1]))
    # #[learn code] be careful that it is not [np.exp(opt[1])*ys**opt[0]]
    # plt.scatter(ys, xs, color="k")
    # plt.plot(ys, fv)
    # plt.plot(ys, fv2)
    # plt.show()
    # exit(0)
    datapack.append([fv, ys, "linefit"])
    plxs = np.percentile(xs, pers)
    plys = np.percentile(ys, pers)
    plotlines = [
        ["x", plxs[0]], ["y", plys[0]], 
        ["x", plxs[1]], ["y", plys[1]], 
        ["x", plxs[2]], ["y", plys[2]], 
        ["x", plxs[3]], ["y", plys[3]], 
        ["x", plxs[4]], ["y", plys[4]]
    ]
    is_show = False
    plot_corse_grainedly_data_from_file(
        datapack, plot_dim, xinfo, yinfo, nameinfo=nameinfo, 
        plotlines=plotlines, is_show=is_show
    )
    '''
