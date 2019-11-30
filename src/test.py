import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from generator import RandomGrayScalePL

original = imread('img2.jpeg')
grayscale = RandomGrayScalePL(original)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

print(grayscale.shape)
fig.tight_layout()
plt.show()
