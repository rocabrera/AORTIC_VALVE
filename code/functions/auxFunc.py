import os
import numpy as np

import SimpleITK as sitk
import pandas as pd
from collections import Counter
from scipy.signal import convolve2d as conv

def createEdge(img, t=1):
    """
    Função que cria o contorno da máscara. 

    Parameters
    ----------
    img : numpy.array
        Imagem em numpy (2D).
    t : int
        Espessura do contorno.

    Returns
    -------
    grad : numpy.array (default = 2)
        Imagem do contorno, binário.

    """
    grad_x = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

    grad_y = np.array([[ 1, 1, 1],
                       [ 0, 0, 0],
                       [-1,-1,-1]])

    img_x = np.abs(conv(img, grad_x, mode='same'))
    img_y = np.abs(conv(img, grad_y, mode='same'))

    thicker_filter = np.ones((t, t))


    grad = ((img_y + img_x) > 0).astype(float)

    grad = np.abs(conv(grad, thicker_filter, mode='same'))

    grad[grad > 1] = 1.0

    return grad


def createImageDic(image_name):
	path_figures  = 'figures'
	imagemFolder = os.path.join(os.getcwd(), path_figures, image_name)

	if not os.path.exists(imagemFolder):#Se não existe folder cria ele.
		os.makedirs(imagemFolder)


def reduceSize(image_np, mask_np):

    d,l,c = mask_np.shape
    dim = [[],[],[]]
    for k in range(d):
        if mask_np[k,:,:].max() == 0:
            continue
        else:
            dim[0].append(k)
        for i in range(l):
            if mask_np[k,i,:].max() == 0:
                continue
            else:
                dim[1].append(i)
            for j in range(c):
                if mask_np[k,i,j] == 1:
                    dim[2].append(j)

    mask = mask_np[min(dim[0]):max(dim[0])+1, min(dim[1]):max(dim[1])+1, min(dim[2]):max(dim[2])+1]
    image = image_np[min(dim[0]):max(dim[0])+1, min(dim[1]):max(dim[1])+1, min(dim[2]):max(dim[2])+1]

    red_image_object = sitk.GetImageFromArray(image)
    red_mask_object = sitk.GetImageFromArray(mask)

    return red_image_object, red_mask_object