import numpy as np
import skimage.data
from skimage.measure import compare_psnr
import matplotlib.pyplot as plt
import cv2

import pybm3d

noise_std_dev = 100
noisy_img = cv2.imread('img1.jpg')

out = pybm3d.bm3d.bm3d(noisy_img, noise_std_dev)

plt.figure(figsize=(16, 5))

plt.subplot(121)
plt.imshow(noisy_img, interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('Noisy image', fontsize=20)
plt.subplot(122)
plt.imshow(out, interpolation='nearest', vmin=0, vmax=5)
plt.axis('off')
plt.title('BM3D tranform', fontsize=20)


plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                    right=1)

plt.show()