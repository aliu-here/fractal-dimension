import cv2
import math
import numpy as np
import copy

def readfile(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    image = image.astype(np.uint8)
    return image

def fillalpha(image):
    trans_mask = image[:,:,3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    return image

def tograyscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return image

def detect_edges(image, thresh1=0, thresh2=80):
    edges = cv2.Canny(image, thresh1, thresh2, 25)
    return edges

def detect_edges_sobel(img):
    img = cv2.GaussianBlur(img, (25, 25), 0)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    return grad_norm

def high_pass_filter(img, filter_size):
    rows, cols, depth = img.shape
    optimal_rows, optimal_cols = (cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols))
    tmp = np.zeros((optimal_rows, optimal_cols, depth))
    tmp[:rows,:cols] = img[:,:]
    img = tmp
    rows, cols = optimal_rows, optimal_cols
    f_r, f_g, f_b = np.fft.fft2(np.float64(img[:,:,0]) / 255), np.fft.fft2(np.float64(img[:,:,1]) / 255), np.fft.fft2(np.float64(img[:,:,2]) / 255)
    fshift_r, fshift_g, fshift_b = np.fft.fftshift(f_r), np.fft.fftshift(f_g), np.fft.fftshift(f_b)  

    center_row = rows // 2
    center_col = cols // 2
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
    return np.array(np.clip(high_pass * 255, 0, 255), dtype='uint8')

def boxcount(image, sizes, thresh=128):
    x = []
    y = []
    rows, cols = image.shape

    for box_size in sizes:
        black_box_count = 0
        for row in range(0, rows, box_size):
            for col in range(0, cols, box_size):
                in_box = False
                for box_row in range(0, min(box_size, rows - row)):
                    for box_col in range(0, min(box_size, cols - col)):
                        if (image[row + box_row][col + box_col] > thresh):
                            in_box = True
                            break
                    if (in_box):
                        break
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))
    return x, y

def boxcount_4x(image, sizes, thresh=128):
    x = []
    y = []
    rows, cols = image.shape

    for box_size in sizes:
        black_box_count = 0
        for row in range(0, rows, box_size):
            for col in range(0, cols, box_size):
                in_box = False
                for box_row in range(0, min(box_size, rows - row)):
                    for box_col in range(0, min(box_size, cols - col)):
                        if (image[row + box_row][col + box_col] > thresh):
                            in_box = True
                            break
                    if (in_box):
                        break
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))

        black_box_count = 0
        for row in range(0, rows, box_size):
            for col in range(cols - box_size - 1, -box_size, -box_size):
                in_box = False
                for box_row in range(0, min(box_size, rows - row)):
                    for box_col in range(0, min(box_size, cols - col)):
                        if (image[row + box_row][max(col + box_col, 0)] > thresh):
                            in_box = True
                            break
                    if (in_box):
                        break
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))

        black_box_count = 0
        for row in range(rows - box_size - 1, -box_size, -box_size):
            for col in range(cols - box_size - 1, -box_size, -box_size):
                in_box = False
                for box_row in range(0, min(box_size, rows - row)):
                    for box_col in range(0, min(box_size, cols - col)):
                        if (image[max(row + box_row, 0)][max(col + box_col, 0)] > thresh):
                            in_box = True
                            break
                    if (in_box):
                        break
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))

        black_box_count = 0
        for row in range(rows - box_size - 1, -box_size, -box_size):
            for col in range(0, cols, box_size):
                in_box = False
                for box_row in range(0, min(box_size, rows - row)):
                    for box_col in range(0, min(box_size, cols - col)):
                        if (image[max(row + box_row, 0)][col + box_col] > thresh):
                            in_box = True
                            break
                    if (in_box):
                        break
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))
    return x, y

def linregression(x, y):
    return np.polyfit(x, y, 1);
