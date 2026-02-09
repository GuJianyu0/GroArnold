#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import imageio

def siglefigure_to_gif(image_list, gif_name):
 
    frames = []
    for image_name in image_list:
        if image_name.endswith('.png'):
            print(image_name)
            frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1) #save as gif

def gif_path_creat(path_singlefigure="./image_gif/", gif_name="./a.gif"):
 
    files = os.listdir(path_singlefigure)
    files.sort()
    # files.sort(key = lambda x:int(x[:-4]))
    image_list=[ path_singlefigure+img for img in files]
    siglefigure_to_gif(image_list, gif_name)
    print("Creat a gif to %s. Done."%(gif_name))



if __name__ == "__main__":

    gmn = "galaxymodel"
    path_singlefigure = "savefig/x_scatter/"
    gif_name = "savefig/params_t_"+gmn+".gif"
    gif_path_creat(path_singlefigure=path_singlefigure, gif_name=gif_name)