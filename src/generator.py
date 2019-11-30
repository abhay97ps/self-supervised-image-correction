#code by Abhay Pratap Singh
# generate pseduo label images using set of images 
import numpy as np
from skimage.color import rgb2gray

# generate gray scale with all chanels = R = G = B = grayscale
# Y = 0.2125 R + 0.7154 G + 0.0721 B
# These weights are used by CRT phosphors as they better represent human perception of red, green and blue than equal weights.
# http://www.poynton.com/PDFs/ColorFAQ.pdf
def GrayScalePL(img):
    grayscale = rgb2gray(img)
    return np.stack((grayscale,)*3, axis=-1)

# generate patched images for pseudo labels (eg: image inpainting)
# patches can be made on individual channels 
# patch shape is square by default
# patch can either be all zero or 255
# rotate patch between [0,90)
def RandomPatchPL(img, channel, type, degree):
     