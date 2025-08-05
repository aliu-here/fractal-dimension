import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import copy
import preprocessing

img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
print(img.shape)
f_r, f_g, f_b = np.fft.fft2(np.float64(img[:,:,0]) / 256), np.fft.fft2(np.float64(img[:,:,1]) / 256), np.fft.fft2(np.float64(img[:,:,2]) / 256)
fshift_r, fshift_g, fshift_b = np.fft.fftshift(f_r), np.fft.fftshift(f_g), np.fft.fftshift(f_b)
magnitude_spectrum_r, magnitude_spectrum_g, magnitude_spectrum_b = 20*np.log(np.abs(fshift_r)), 20*np.log(np.abs(fshift_g)), 20*np.log(np.abs(fshift_b))

rows, cols, depth = img.shape
filter_size = 1
center_row = rows // 2
center_col = cols // 2
transformed = [copy.deepcopy(fshift_r), copy.deepcopy(fshift_g), copy.deepcopy(fshift_b)]
for channel in transformed:
    mask = np.ones((rows, cols))
    mask[center_row - filter_size:center_row + filter_size + 1, center_col - filter_size : center_col + filter_size + 1] = 0
    channel[mask != 0] = 0

low_pass = np.zeros(img.shape)
low_pass[:,:,3] = 1
for i in range(3):
    f_ishift = np.fft.ifftshift(transformed[i])
    low_pass[:,:,i] = np.real(np.fft.ifft2(f_ishift))
fig, ax = plt.subplots(3, 2)
print(low_pass)

transformed = [copy.deepcopy(fshift_r), copy.deepcopy(fshift_g), copy.deepcopy(fshift_b)]
for channel in transformed:
    mask = np.zeros((rows, cols))
    mask[center_row - filter_size:center_row + filter_size + 1, center_col - filter_size : center_col + filter_size + 1] = 1
    channel[mask != 0] = 0

high_pass = np.zeros(img.shape)
high_pass[:,:,3] = 1
for i in range(3):
    f_ishift = np.fft.ifftshift(transformed[i])
    high_pass[:,:,i] = np.real(np.fft.ifft2(f_ishift))

fig, ax = plt.subplots(3, 2)

ax[0][0].set_title("original image")
ax[0][0].imshow(img)

ax[0][1].set_title("FFT of red channel")
ax[0][1].imshow(magnitude_spectrum_r, cmap='gray')

ax[1][0].set_title("FFT of green channel")
ax[1][0].imshow(magnitude_spectrum_g, cmap='gray')

ax[1][1].set_title("FFT of blue channel")
ax[1][1].imshow(magnitude_spectrum_b, cmap='gray')

ax[2][0].set_title("image after low-pass filter")
ax[2][0].imshow(low_pass)

ax[2][1].set_title("image after high-pass filter")
ax[2][1].imshow(preprocessing.high_pass_filter(img, 1))
plt.show()
