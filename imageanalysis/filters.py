"""
Filters for image processing

Authors
-------

    Jani <jkkoski@iki.fi>

Changelog
---------

    20240628 -- Added tis file from the old py_imageanalysis to this file
    20201109 -- Started implementing the filters

Contributors
------------

    Manuel Längle <manuel.laengle@univie.ac.at>

"""
from scipy import fftpack
import numpy as np
from math import floor


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
    
