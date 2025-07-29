import cv2
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def factors(val):
    out = {val}
    for i in range(1, math.ceil(val ** 0.5 + 1)):
        if (val % i == 0):
            out.add(int(val / i));
            out.add(int(i))
    return out;

image = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

trans_mask = image[:,:,3] == 0
image[trans_mask] = [255, 255, 255, 255]

height, width, channels = image.shape
#image = cv2.resize(image, (min(height, width), min(height,width)))
image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

rows, cols = image.shape

size_factor = factors(cols);
if (len(factors(rows)) > len(size_factor)):
    size_factor = factors(rows)

print(len(size_factor))

thresh = 127

x = []
y = []

print(size_factor)

for factor in size_factor:
    black_pix_count = 0;
    box_size = cols // factor;
    print(math.log(factor))
    for row in range(0, rows - 1, box_size):
        for col in range(0, cols - 1, box_size):
            in_box = False
            for box_row in range(0, min(box_size, rows - row)):
                for box_col in range(0, min(box_size, cols - col)):
                    if (image[row + box_row][col + box_col] < thresh):
                        in_box = True
                        break
                if (in_box):
                    break

            black_pix_count += in_box

            #count = summed_table[row][col] + summed_table[min(row + box_size, rows)][min(col + box_size, cols)] - summed_table[min(row + box_size, rows)][col] - summed_table[row][min(col + box_size, cols)];
            #if (count < 255 * (min(row + box_size, rows) - row) * (min(col + box_size, cols) - col)):
                #black_pix_count += 1
    print(math.log(black_pix_count))
    y.append(math.log(black_pix_count))
    x.append(math.log(factor))
    try:
        print(f"fd ~= {math.log(black_pix_count) / math.log(factor)}")
    except:
        print(f"fd = NaN")
    print()

m, b = np.polyfit(x, y, 1);
print(m)

fig, axs = plt.subplots(1);
axs.scatter(x, y);
axs.axline((0, b), slope=m);
plt.show()
