import cv2
import math
import cupy
import numpy as np
import copy

def readfile(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    image = image.astype(cupy.uint8)
    return cupy.asarray(image)

def fillalpha(image):
    trans_mask = image[:,:,3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    return image

def tograyscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return image

def detect_edges(image, thresh1=0, thresh2=80):
    edges = cv2.Canny(image.get(), thresh1, thresh2, 5)
    return cupy.asarray(edges)

def detect_edges_GPU(image, detector):
    dstImg = detector.detect(image)
    return dstImg.download()

def detect_edges_sobel(img):
    img = cv2.GaussianBlur(img, (25, 25), 0)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    grad = cupy.sqrt(grad_x**2 + grad_y**2)
    grad_norm = (grad * 255 / grad.max()).astype(cupy.uint8)
    return grad_norm

def high_pass_filter(img, filter_size):
    rows, cols, depth = img.shape
    optimal_rows, optimal_cols = (cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols))
    tmp = cupy.zeros((optimal_rows, optimal_cols, depth))
    tmp[:rows,:cols] = img[:,:]
    img = tmp
    rows, cols = optimal_rows, optimal_cols
    f_r, f_g, f_b = cupy.fft.fft2(cupy.array(img[:,:,0], dtype=np.float32) / 255), cupy.fft.fft2(cupy.array(img[:,:,1], dtype=np.float32) / 255), cupy.fft.fft2(cupy.array(img[:,:,2], dtype=np.float32) / 255)
    fshift_r, fshift_g, fshift_b = cupy.fft.fftshift(f_r), cupy.fft.fftshift(f_g), cupy.fft.fftshift(f_b)  

    center_row = rows // 2
    center_col = cols // 2
    transformed = [copy.deepcopy(fshift_r), copy.deepcopy(fshift_g), copy.deepcopy(fshift_b)]
    for channel in transformed:
        mask = cupy.zeros((rows, cols))
        mask[center_row - filter_size:center_row + filter_size + 1, center_col - filter_size : center_col + filter_size + 1] = 1
        channel[mask != 0] = 0

    high_pass = cupy.zeros(img.shape)
    high_pass[:,:,3] = 1
    for i in range(3):
        f_ishift = cupy.fft.ifftshift(transformed[i])
        high_pass[:,:,i] = cupy.real(cupy.fft.ifft2(f_ishift))
    return cupy.array(cupy.clip(high_pass * 255, 0, 255), dtype='uint8')


def calc_sum_area(table, br, tl):
    return table[tl[0]][tl[1]] + table[br[0]][br[1]] - table[tl[0]][br[1]] - table[br[0]][tl[1]]

def boxcount(image, sizes):
    x = []
    y = []
    rows, cols = image.shape

    prefsum = cupy.zeros((rows + 1, cols + 1))
    prefsum[1:,1:] = image.cumsum(axis=0).cumsum(axis=1)
    for box_size in sizes:
        black_box_count = 0
        for row in range(0, rows, box_size):
            for col in range(0, cols, box_size):
                in_box = (prefsum[min(row + box_size, rows)][min(col + box_size, cols)] - prefsum[row][col]) > 0
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))
    return x, y

def boxcount_4x(image, sizes):
    x = []
    y = []
    rows, cols = image.shape

    prefsum = np.zeros((rows + 1, cols + 1))
    prefsum[1:,1:] = (image.cumsum(axis=0).cumsum(axis=1)).get()

    for box_size in sizes:
        black_box_count = 0
        for row in range(0, rows, box_size):
            for col in range(0, cols, box_size):
                in_box = (calc_sum_area(prefsum, (min(row + box_size, rows), min(col + box_size, cols)), (row, col))) > 0
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))

        black_box_count = 0
        for row in range(0, rows, box_size):
            for col in range(cols - box_size - 1, -box_size, -box_size):
                in_box = (calc_sum_area(prefsum, (min(box_size + row, rows), col + box_size), (row, max(col, 0)))) > 0
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))

        black_box_count = 0
        for row in range(rows - box_size - 1, -box_size, -box_size):
            for col in range(cols - box_size - 1, -box_size, -box_size):
                in_box = (calc_sum_area(prefsum, (box_size + row, box_size + col), (max(row, 0), max(col, 0)))) > 0
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))

        black_box_count = 0
        for row in range(rows - box_size - 1, -box_size, -box_size):
            for col in range(0, cols, box_size):
                in_box = (calc_sum_area(prefsum, (box_size + row, min(box_size + col, cols)), (max(row, 0), col))) > 0
                black_box_count += in_box
        x.append(math.log(1/box_size))
        y.append(math.log(black_box_count))
    return x, y

def linregression(x, y):
    return np.polyfit(x, y, 1);
