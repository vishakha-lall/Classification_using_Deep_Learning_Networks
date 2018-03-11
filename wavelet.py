import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, color
from skimage.util import random_noise


noisy = cv2.imread('img1.jpg')

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 10),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=True, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print("Estimated Gaussian noise standard deviation = {}".format(sigma_est))

ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('Noisy')
ax[1].imshow(denoise_wavelet(noisy, multichannel=True))
ax[1].axis('off')
ax[1].set_title('Wavelet denoising')
ax[2].imshow(denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True))
ax[2].axis('off')
ax[2].set_title('Wavelet denoising\nin YCbCr colorspace')

fig.tight_layout()

plt.show()