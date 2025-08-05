"""
Filters for image processing

Authors
-------

    Jani <jkkoski@iki.fi>

Changelog
---------

    20240628 -- Added tis file from the old py_imageanalysis to this file
    20201109 -- Started implementing the filters
    20240628 -- Added more filters and morphological operations 

Contributors
------------

    Manuel Längle <manuel.laengle@univie.ac.at>
    Somar Dibeh <somar.dibeh@univie.ac.at>

"""
from scipy import fftpack
import numpy as np
from math import floor
import cv2
from skimage import exposure

def gaussian_blur():

    pass

def dgfilter(shape, params):
        """Returns the double Gaussian filter

        Arguments:

        params -- (sigma1, sigma2, weight)
        """

        sigma1, sigma2, weight = params
        s1 = sigma1*sigma1
        s2 = sigma2*sigma2

        yy_min = -shape[0]//2
        yy_max = shape[0]//2
        xx_min = -shape[1]//2
        xx_max = shape[1]//2
        yy, xx = np.meshgrid(np.linspace(yy_min, yy_max, shape[0]),
                                np.linspace(xx_min, xx_max, shape[1]),
                                indexing='ij')
        rr = np.square(xx/(shape[1]*0.5)) + np.square(yy/(shape[0]*0.5))

        filter = np.exp(-0.5*rr/s1) - (1.0-weight)*np.exp(-0.5*rr/s2)

        return filter

def double_gaussian(data, sigma1, sigma2, weight):
        """Returns image with the Double Gaussian filter applied [1] to an image

        Arguments:

        data -- 2D np array, image data
        sigma1 -- float, std for the positive Gaussian (fractional)
        sigma2 -- float, std for the negative Gaussian (fractional)
        weight -- float, weight of the second Gaussian

        [1] Krivanek et al., Nature 464m 571-574 [2010]
        """    

        fft_data = fft(data)

        filtered_fft_data = fft_data * dgfilter(data.shape, (sigma1, sigma2, weight))
        filtered_image = ifft(filtered_fft_data)
        
        return filtered_image

def fft(data):
    """Returns fft of an image

    Arguments:

    data -- 2D np array, image data
    """
    return fftpack.fftshift(fftpack.fft2(data))

def ifft(data):
    """Returns the real part of the inverse fft of an image

    Arguments:

    data -- 2D np array, image data
    """
    return fftpack.ifft2(fftpack.ifftshift(data)).real


def fft_filter(data, fftspots, size):
    """
    Returns a fourier filtered image as well as the filtered FFT where one set of FFT spots has been deleted
    
    Arguments: Image Series, FFT spots, size of overwritten area
    """
    filtered_data=[]
    filtered_fft=[]
    for image in data:
        data_fft = fft(image)    
        for i in range(len(fftspots)):
            data_fft[int(round(fftspots[i,1]))-size:int(round(fftspots[i,1]))+size,int(round(fftspots[i,0]))-size:int(round(fftspots[i,0]))+size] = np.average(data_fft)
        filtered_fft.append(data_fft)
        filtered_data.append(ifft(data_fft))
    return np.array(filtered_data), np.array(filtered_fft)
    



################################## Filters functions #####################################
def improve_contrast( image,percentile=10):
    low = np.percentile(image, percentile)
    high = np.percentile(image,100 - percentile)
    return exposure.rescale_intensity(image, in_range=(low, high))


def apply_clahe( image, clip_limit, tile_grid_size):
    """Ensures the image is 8-bit before applying CLAHE."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(image)


def apply_gamma_correction( image, gamma):
    """Applies gamma correction."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_sigmoid_contrast( image, alpha, beta):
    normalized = image.astype(np.float32) / 255
    sigmoid = 1 / (1 + np.exp(-alpha * (normalized - beta)))
    return (255 * sigmoid).astype(np.uint8)


def apply_log_transform( image):
    c = 255 / np.log(1 + np.max(image))
    return (c * np.log1p(image)).astype(np.uint8)


def apply_exp_transform( image, gamma):
    return (255 * (image/255) ** gamma).astype(np.uint8)


def apply_gaussian_blur(image, kernel_size, sigma=0):
    """Applies Gaussian blur to the image."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size (required by OpenCV)
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return image



# Modified morphological operations to use dynamic kernel size
def dilate( image, iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

def erode( image, iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def opening( image, iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

def closing( image, iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def gradient( image, iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)

def boundary_extraction( image, iterations, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return cv2.subtract(image, eroded)



