import cv2
import math
import numpy as np
import copy
import scipy

def readfile(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    image = image.astype(np.uint8)
    return np.asarray(image)

def fillalpha(image):
    trans_mask = image[:,:,3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    return image

def tograyscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return image

def detect_edges(image, thresh1=0, thresh2=80):
    edges = cv2.Canny(image, thresh1, thresh2, 5)
    return np.asarray(edges)

def detect_edges_GPU(image, detector):
    dstImg = detector.detect(image)
    return dstImg.download()

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
    
    for i in range(3):
        img[:,:,i] = scipy.fft.fftshift(scipy.fft.dctn(np.array(img[:,:,i], dtype=np.float32) / 255))

    center_row = rows // 2
    center_col = cols // 2
    for i in range(3):
        mask = np.zeros((rows, cols))
        mask[center_row - filter_size:center_row + filter_size + 1, center_col - filter_size : center_col + filter_size + 1] = 1
        img[:,:,i][mask != 0] = 0

    for i in range(3):
        f_ishift = scipy.fft.ifftshift(img[:,:,i])
        img[:,:,i] = scipy.fft.idctn(f_ishift)
    return np.array(np.clip(img * 255, 0, 255), dtype='uint8')


def calc_sum_area(table, br, box_size):
    tl = (max(br[0] - box_size, 0), max(br[1] - box_size, 0))
    return table[tl[0]][tl[1]] + table[br[0]][br[1]] - table[tl[0]][br[1]] - table[br[0]][tl[1]]

def boxcount(image, sizes):
    x = []
    y = []
    rows, cols = image.shape

    prefsum = np.zeros((rows + 1, cols + 1))
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
    prefsum[1:,1:] = (image.cumsum(axis=0).cumsum(axis=1))

    for box_size in sizes:
        boxlocs = [[(min(box_size + x, rows), min(box_size + y, cols)) for x in range(0, rows, box_size)] for y in range(0, cols, box_size)]
        black_box_count = sum(np.array([[calc_sum_area(prefsum, box, box_size) > 0 for box in row] for row in boxlocs]).flatten())
        try:
            x.append(math.log(1/box_size))
            y.append(math.log(black_box_count))
        except:
            pass

        boxlocs = [[(min(box_size + x, rows), box_size + y) for x in range(0, rows, box_size)] for y in range(cols - box_size - 1, -box_size, -box_size)]
        black_box_count = sum(np.array([[calc_sum_area(prefsum, box, box_size) > 0 for box in row] for row in boxlocs]).flatten())
        try:
            x.append(math.log(1/box_size))
            y.append(math.log(black_box_count))
        except:
            pass

        boxlocs = [[(box_size + x, box_size + y) for x in range(rows - box_size - 1, -box_size, -box_size)] for y in range(cols - box_size - 1, -box_size, -box_size)]
        black_box_count = sum(np.array([[calc_sum_area(prefsum, box, box_size) > 0 for box in row] for row in boxlocs]).flatten()) 
        try:
            x.append(math.log(1/box_size))
            y.append(math.log(black_box_count))
        except:
            pass

        boxlocs = [[(box_size + x, min(box_size + y, cols)) for x in range(rows - box_size - 1, -box_size, -box_size)] for y in range(0, cols, box_size)]
        black_box_count = sum(np.array([[calc_sum_area(prefsum, box, box_size) > 0 for box in row] for row in boxlocs]).flatten()) 
        try:
            x.append(math.log(1/box_size))
            y.append(math.log(black_box_count))
        except:
            pass
    return x, y

def linregression(x, y):
    return np.polyfit(x, y, 1);
