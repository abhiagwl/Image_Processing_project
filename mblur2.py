from skimage.graph import route_through_array
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from matplotlib.widgets import Button
from skimage import morphology

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    sigma = np.std(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def mblur(inpts,image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = auto_canny(blurred)
    edges[-1,:] = 255
    edges[0,:] = 255
    edges[:,-1] = 255
    edges[:,0] = 255
    totpt = len(inpts)
    path = np.zeros_like(edges)
    H,W = edges.shape
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    for i in range(totpt):
        sigma2 = ((inpts[(i+1)%totpt][1]-inpts[i][1])**2 + (inpts[(i+1)%totpt][0]-inpts[i][0])**2 )/4
        gmask = np.exp(-((x-inpts[i][1])**2 + (y-inpts[i][0])**2)/(2.0*100.0))
        gmask += np.exp(-((x-inpts[(i+1)%totpt][1])**2 + (y-inpts[(i+1)%totpt][0])**2)/(2.0*sigma2))
        emask = np.multiply(gmask,edges)/gmask.max()
        emask = 255 - emask
        indices, weight = route_through_array(emask, inpts[i],inpts[(i+1)%totpt])
        indices = np.array(indices).T
        path[indices[0], indices[1]] = 1
    cent = ndimage.measurements.center_of_mass(path)
    h,w = path.shape
    mask = np.zeros((h+2,w+2),dtype=np.uint8)
    cv2.floodFill(path, mask, (int(cent[1]),int(cent[0])), 1)
    skeleton = morphology.skeletonize(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilim = cv2.dilate(path,kernel,iterations=10)
    erodim = cv2.erode(path,kernel,iterations=5)
    dilb = cv2.dilate(dilim,kernel) - dilim
    im2, contours, hierarchy = cv2.findContours(dilim,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    grabmask = np.zeros_like(path)
    # grabmask[y:y+h,x:x+w] = 2
    grabmask[dilim==1] = 3
    grabmask[dilb==1] = 2
    grabmask[skeleton==1] = 1
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    grabmask, bgdModel, fgdModel = cv2.grabCut(image,grabmask,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((grabmask==2) | (grabmask==0),0,1).astype('uint8')
    out1 = image*mask2[:,:,np.newaxis]
    mask2 = np.bitwise_or(mask2,erodim)
    out = image*mask2[:,:,np.newaxis]
    ksize = 31
    kernel = np.zeros((ksize,ksize))
    kernel[int((ksize-1)/2),:] = np.ones(ksize)
    kernel/=ksize
    bimg = cv2.filter2D(image,-1,kernel)
    bimg[mask2==1] = 0
    bimg = bimg + out
    plt.imsave("bImg",bimg)
    return bimg