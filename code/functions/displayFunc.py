from collections import Counter
import os

import numpy as np
import cv2
import SimpleITK as sitk
import skimage.segmentation as seg
from skimage import exposure
import matplotlib.pyplot as plt
import imageio

import auxFunc 

def makeGifDir(segmentationFolder, fileGifPath):

    lstFiles = [file for file in os.listdir(segmentationFolder) if os.path.isfile(os.path.join(segmentationFolder, file))]
    dicioFiles = {i:int(i.split('.')[0]) for i in lstFiles}
    dicioFiles={k: v for k, v in sorted(dicioFiles.items(), key=lambda item: item[1])}

    with imageio.get_writer(fileGifPath, mode='I', duration=0.6) as writer:
        for filename in dicioFiles.keys():
            image = imageio.imread(os.path.join(segmentationFolder, filename))
            writer.append_data(image)

def makeImageContour(image_np, mask_np):
    
    # Normaliza imagem para visualização
    image_slice = (image_np - image_np.min())/(image_np.max() - image_np.min())
    
    # Cria a imagem RGB, as 3 camadas são iguais a imagem grayscale
    image_rgb = np.zeros((image_slice.shape[0], image_slice.shape[1], 3))
    image_rgb[:,:,0] = image_slice.copy()
    image_rgb[:,:,1] = image_slice.copy()
    image_rgb[:,:,2] = image_slice.copy()
    
    # Parte da máscara
    mask_edge = auxFunc.createEdge(mask_np)
    i_mask_edge = 1 - mask_edge
    
    # Cria o contorno chavoso
    image_rgb[:,:,2] = image_rgb[:,:,2] + mask_edge # R
    image_rgb[:,:,1] = image_rgb[:,:,1]*i_mask_edge # G 
    image_rgb[:,:,0] = image_rgb[:,:,0]*i_mask_edge # B
    
    # Correção do contorno
    image_rgb[image_rgb > 1] = 1 # Quando soma o contorno no canal vermelho, pode passar de 1
    image_rgb = (image_rgb*255).astype(int) # Passa para 255 antes de salvar
    
    return image_rgb

def make3DImgMask(image, mask, auxPathImage, auxPathMask):

    path_figures  = 'figures'
    segmentationFolder = os.path.join(os.getcwd(), path_figures, auxPathImage, auxPathMask)
    imageFolder = os.path.join(os.getcwd(), path_figures, auxPathImage)

    if not os.path.exists(segmentationFolder):#Se não existe folder cria ele.
        os.makedirs(segmentationFolder) #Cria o folder que contem os slices em png
        #########################################
        image_np = sitk.GetArrayFromImage(image)
        mask_np = sitk.GetArrayFromImage(mask)
        #########################################
        indice=mask_np.max(axis= (1,2)).astype(bool)#Seleciona em shape[0]
        image_np = image_np[indice,:,:]
        mask_np  = mask_np[indice,:,:]
        #########################################
        for i in range(sum(indice)):
            image_rgb = makeImageContour(image_np[i,:,:], mask_np[i,:,:])
            number = f'{i}.png' 
            cv2.imwrite(os.path.join(segmentationFolder, number), image_rgb)

        fileGifPath = imageFolder + os.sep + auxPathMask + '.gif'
        makeGifDir(segmentationFolder, fileGifPath)
    # Seleciona Slice

def make2DImgMask(image, mask, auxPathImage, auxPathMask):

    image_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)

    image_np = image_np[0,:,:]
    mask_np = mask_np[0,:,:]

    image_rgb = makeImageContour(image_np, mask_np)

    auxPathMask += '.png' 
    cv2.imwrite(os.path.join(os.getcwd(),'figures',auxPathImage, auxPathMask), image_rgb)

def makeImgMask(image, mask, auxPathImage, auxPathMask):
    shape_mask = mask.GetSize()
    dimension = sum([True if i != 1 else False for i in shape_mask])

    if dimension == 3:
    	make3DImgMask(image, mask, auxPathImage, auxPathMask)
    elif dimension == 2:
    	make2DImgMask(image, mask, auxPathImage, auxPathMask)


def dictSummary(dicio):
    print('DICT SUMMARY')
    print(f'Quantidade de imagens: {len(dicio)}')
    
    qtdSegImage_lst = [len(dicio[chave]) for chave in dicio.keys()]
    result = Counter(qtdSegImage_lst)
    
    for i in result.keys():
        print(f'Número de imagens com {i} segmentações: {result[i]}')
    print()