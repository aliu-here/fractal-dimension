import cv2
import math
import numpy as np

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
