#import matplotlib
import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import cv2 as cv
from scipy.ndimage import zoom
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.mplot3d import Axes3D

import skimage.segmentation as seg
from skimage import exposure


import os

def zoomIn(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming in
    if zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def printSlices(data, rows=3, cols=3, start_with=10, show_every=1):
    fig,ax = plt.subplots(rows,cols,figsize=[20,20])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(data[:,:,ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

def resample(image, x,y,z , new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([x,y,z])

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def print3DImage(mask):	
    
    colors = np.empty(mask.shape, dtype=object)
    colors[mask == 1] = '#e19c2d30' 

    corErodeMask = ['#2ecd2350', '#2ecd2350', '#3a36ba70', '#e12d2d80', '#01040390']

    mask = mask.astype(float)

    kernel = np.ones((5,5))
    dataErosed = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
    fig = plt.figure(figsize=(20,20))
    ax = plt.axes(projection='3d')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim3d(215,285)
    ax.set_ylim3d(215,285)

    ax.view_init(elev=45, azim=35)
    aux = mask.copy()
    
    qtd_mask = mask.sum()

    for j in np.arange(1,6):
        for i in np.arange(0, mask.shape[2]):
            dataErosed[:,:,i] = cv.erode(mask[:,:,i], kernel, iterations=j)
        print(f"j: {j}\tArea: {dataErosed.sum()/qtd_mask}")

        aux += dataErosed.astype(float)

        colors[aux == j+1] = corErodeMask[j-1]

    ax.voxels(mask, facecolors = colors)
    plt.savefig('images/imagem3D.png', format = 'png')
    plt.show()

def print2DImage(data, qtdSlices=2):	
	kernel = np.ones((5,5))
	dataErosed = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
	fig = plt.figure(figsize=(20,20))

	#ax.set_xlabel('X axis')
	#ax.set_axis_off()
    
	for j in np.arange(1,qtdSlices+1):
		dataErosed[:,:,25] = cv.erode(data[:,:,25], kernel, iterations=j)
		
		#plt.imshow(dataErosed,  alpha=j/qtdSlices)
    
	plt.show()

def print2DSlices(data, mask, rows=3, cols=3, start_with=5, show_every=2):

	inverseMask = np.where(mask == 1, 0, 1)
	dataMaskaradeInverse = np.where(inverseMask == 1, data, -2048)
	fig,ax = plt.subplots(rows,cols,figsize=[20,20])

	for i in range(rows*cols):
		ind = start_with + i*show_every
		ax[int(i/rows),int(i % rows)].set_title('Slice %d' % ind)
		ax[int(i/rows),int(i % rows)].imshow(dataMaskaradeInverse[:,:,ind], cmap='gray')
		ax[int(i/rows),int(i % rows)].axis('off')
    #fig.savefig('images/slices2DPosition.png')
	plt.show()

def imageProfessor(data, mask, slice = 0):
	fig,ax = plt.subplots(2,3,figsize=[20,20])
	fig.suptitle('Slice %d' %slice)

	cmin = data.min()
	cmax = 2048

	ax[0,0].set_title('Image without mask')
	ax[0,0].imshow(data[:,:,slice], origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
	ax[0,0].axis('off')

	dataMaskarade = np.where(mask[:,:,slice] == 1, data[:,:,slice], cmin)

	ax[0,1].set_title('Image with mask')
	ax[0,1].imshow(dataMaskarade, origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
	ax[0,1].axis('off')

	ax[0,2].set_title('Zoom mask \n Number of pixels: {:5d} \n Area: {}%'.format(mask[:,:,slice].sum(), 100))
	ax[0,2].imshow(zoomIn(dataMaskarade,4.5), origin="lower",cmap='gray', vmin=cmin,vmax=cmax) 
	ax[0,2].axis('off')

	kernel = np.ones((5,5))

	for i in np.arange(3):
		erodeMask = cv.erode(mask[:,:,slice], kernel, iterations=i+1)
		dataErode = np.where(erodeMask == 1, data[:,:,slice], cmin)
		ax[1,i].imshow(zoomIn(dataErode,4.5), origin="lower", cmap='gray', vmin=cmin,vmax=cmax) 
		ax[1,i].set_title('Erode Mask: {}%\n Number of pixels: {:5d} \n Area: {:.2f}%'.format( (0.75-0.25*i ), erodeMask.sum(), erodeMask.sum()*100/mask[:,:,slice].sum() ) ) 
		ax[1,i].axis('off')

	fig.savefig('images/imageProfessor.png')

	fig, ax = plt.subplots(figsize=[5, 4])
	fig.suptitle('Slice %d' %slice)

	mask = np.where(mask == 1, 2048, 0)
	region = mask[:,:,slice] + data[:,:,slice]
	ax.imshow(region, origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
	ax.set_title('Mask Region')
	# Create inset of width 1.3 inches and height 0.9 inches
	# at the default upper right location
	axins = zoomed_inset_axes(ax, 2, bbox_to_anchor=(1.05, .6, .5, .4), bbox_transform=ax.transAxes, loc=1, borderpad=0)

	#axins.tick_params(labelleft=False, labelbottom=False)
	axins.imshow(dataMaskarade, origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
	axins.set_title('Zoom')
	axins.set_xlim(225, 290)
	axins.set_ylim(230, 290)

	for axis in ['top','bottom','left','right']:
		axins.spines[axis].set_linewidth(2)
		axins.spines[axis].set_color('r')
	mark_inset(ax, axins, loc1=4, loc2=2, fc="none", lw=1, ec='r', zorder = 3)
	#ax.indicate_inset_zoom(axins)
    #fig.savefig('images/imageProfessor2.png')
	plt.show()

def findLimitsSlice(mask):
    rows = mask.shape[0]
    columns = mask.shape[1]
    aux1 = 0
    aux2 = 0
    
    for i in np.arange(rows):
        for j in np.arange(columns):
            if(mask[i][j] == 1 and aux1 == 0):
                ymin = i-1
                aux1 = 1
                aux2 = 1
        if(np.all(mask[i]==0) and aux2 == 1):
                ymax = i+1
                break
            
    aux1 = 0
    aux2 = 0
    
    for i in np.arange(columns):
        for j in np.arange(rows):
            if(mask[j][i] == 1 and aux1 == 0):
                xmin = i-1
                aux1 = 1
                aux2 = 1
        if(np.all(mask[:,i]==0) and aux2 == 1):
                xmax = i+1
                break
    
    return ymin,ymax,xmin,xmax
            
def findLimitsVolume(mask):
    
    nSlice = mask.shape[2] -1
    ymin, ymax, xmin, xmax = findLimitsSlice(mask[:,:,0])
    for i in np.arange(1,nSlice):        
        nYmin, nYmax, nXmin, nXmax = findLimitsSlice(mask[:,:,i])
        if(ymin > nYmin):
            ymin = nYmin
        if(ymax < nYmax):
            ymax = nYmax
        if(xmin > nXmin):
            xmin = nXmin
        if(xmax < nXmax):
            xmax = nXmax
        
    return ymin, ymax, xmin, xmax

def makeGifDir(diretorio, name):
    
    PathImages = "images/{}".format(diretorio)
 
    lstFiles = [file for file in os.listdir(PathImages) if os.path.isfile(os.path.join(PathImages, file))]
    lstFiles.sort()
    with imageio.get_writer('{}/Gif/{}.gif'.format(PathImages,name), mode='I', duration=0.1) as writer:
        for filename in lstFiles:
            image = imageio.imread(os.path.join(PathImages, filename))
            writer.append_data(image)
                 
def plotImagesContour(data, mask):
    
    path = 'images/slicesTestes'
    data = data+2048
    
    for i in np.arange(0,data.shape[2]):
        contour = seg.mark_boundaries(data[:,:,i], mask[:,:,i], color=(1, 0, 0), mode='thick', outline_color=(1,0,0))
        p2, p98 = np.percentile(contour, (0.2, 99.8))
        img = exposure.rescale_intensity(contour, in_range=(p2, p98))
        plt.title('sliceNumber{}'.format(i+376))
        plt.imshow(img,cmap = 'gray', origin="lower")
        plt.savefig(os.path.join(path,'sliceNumber{}.png'.format(i+376)), format='png')

    makeGifDir('slicesTestes','slicesContourApplied')
        

def makeGifMaskApplied(data, mask):
    
    cmin = data.min()
    cmax = 2048
   
    
    path = 'images/slicesOriginalSegment/'
        
    f, axarr = plt.subplots(1,2,figsize=[20,20])
    dataMaskarade = []
    for i in  np.arange(0, data.shape[2]):
        dataMaskarade.append(np.where(mask[:,:,i] == 1, data[:,:,i], cmin))
        
        
    data = data+2048
    
    for i in np.arange(0, data.shape[2]):
        f.suptitle('Slice {}'.format(i+376), fontsize=40)
        
        contour = seg.mark_boundaries(data[:,:,i], mask[:,:,i], color=(1, 0, 0), mode='thick', outline_color=(1,0,0))
        
        p2, p98 = np.percentile(contour, (0.2, 99.8))
        img = exposure.rescale_intensity(contour, in_range=(p2, p98))
        axarr[0].imshow(img, origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
        axarr[0].axis('off')
        
        axarr[1].imshow(dataMaskarade[i], origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
        axarr[1].axis('off')
        plt.savefig(os.path.join(path,'sliceNumber{}.png'.format(i+376)), format='png')
        
    
    makeGifDir('slicesOriginalSegment','slices_Segment_Original_SideBySide')
    
    
def makeGifErodeMasksApplied(data, mask):
    
    ymin, ymax, xmin, xmax = findLimitsVolume(mask)
    
    cmin = data.min()
    cmax = 2048
   
    path = 'images/slicesErodeMaskApplied/'
    
    f, axarr = plt.subplots(2,2,figsize=[20,20])
    
    kernel = np.ones((5,5))
    
    for i in np.arange(0, data.shape[2]):
        f.suptitle('Slice {}'.format(i+376), fontsize=40)
        dataMaskarade = np.where(mask[:,:,i] == 1, data[:,:,i], cmin)
        axarr[0,0].imshow(dataMaskarade[ymin-5:ymax+5,xmin-5:xmax+5], origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
        axarr[0,0].axis('off')
        axarr[0,0].set_title('Original Mask',fontsize=26)
        
        for j in np.arange(1,4):
            erodeMask = cv.erode(mask[:,:,i], kernel, iterations=j)
            dataErode = np.where(erodeMask == 1, data[:,:,i], cmin)
            axarr[j//2,j%2].imshow(dataErode[ymin-5:ymax+5,xmin-5:xmax+5], origin="lower", cmap='gray', vmin=cmin,vmax=cmax)
            axarr[j//2,j%2].axis('off')
            axarr[j//2,j%2].set_title('Erode Mask: {}'.format(j),fontsize=26)
       
        plt.savefig(os.path.join(path,'sliceNumber{}.png'.format(i+376)), format='png')
        
    
    makeGifDir('slicesErodeMaskApplied','slicesErodeMaskApplied')
    
        
def makeMorphologyErodeMask(mask):
   
    colors = []
    colors.append((1,1,1)) #Branco
    colors.append((1,0.5,0)) #Laranja
    colors.append((0,1,0)) #Verde
    colors.append((0,0,1)) #Azul
    colors.append((1,0,0)) #Vermelho
    colors.append((0,0,0)) #Preto
    
    cmap_name = 'my_list'
    
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    
    ymin, ymax, xmin, xmax = findLimitsVolume(mask)
     
    path = 'images'+os.sep+'slicesMorphology'
    
    maskSlices = np.zeros(mask.shape)
    kernel = np.ones((5,5))
    for i in np.arange(0, mask.shape[2]):
        maskSlices[:,:,i] = maskSlices[:,:,i] + mask[:,:,i] # Esse seria o plot da mascara original
        plt.title('sliceNumber {}.png'.format(i+376))
        for j in np.arange(1,5):
            erodeMask = cv.erode(mask[:,:,i], kernel, iterations=j)
            maskSlices[:,:,i] = maskSlices[:,:,i] + erodeMask
     
        plt.imshow(maskSlices[ymin-5:ymax+5,xmin-5:xmax+5,i], origin="lower", vmin=0, vmax=5, cmap = cm)
        
        plt.savefig(os.path.join(path,'sliceNumber{}.png'.format(i+376)), format='png')
     

    makeGifDir('slicesMorphology','slicesMorphology')
     
def makeAllSlice(data):
    
    cmin = data.min()
    cmax = 2048
    
    path = 'images/slicesAll/'
    
    for i in np.arange(0,data.shape[0]):
        plt.title('sliceNumber {}.png'.format(i++376))
        plt.imshow(data[i,:,:], origin="lower", vmin = cmin, vmax = cmax, cmap='gray')
        plt.savefig(os.path.join(path,'sliceNumber{}.png'.format(i+376)), format='png')
         
    makeGifDir('slicesAll','allSlices')
    

def testeSlices(data,mask):

    for y in np.arange(data.shape[1]):
        for x in np.arange(data.shape[0]):
            if(mask[x,y,0] == 1):
                data[y,x,0] = -2048
               
    plt.imshow(data[:,:,0], origin="lower", cmap='gray')

    
    
    
    

   



    
    
    

    