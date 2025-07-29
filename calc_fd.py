drimdb_path = "./DRIMDB/"
good_path = drimdb_path + "Good"
bad_path = drimdb_path + "Bad"
outlier_path = drimdb_path + "Outlier"

from preprocessing import *
import glob
import pandas as pd

paths = []
fds = []
quality = []

box_sizes = [2 ** n for n in range(1, 7)]

#for filename in glob.glob(outlier_path + "/*.jpg"):
    #    edges = detect_edges(tograyscale(fillalpha(readfile(filename))))
#    x, y = boxcount(edges, box_sizes)
#    print(x, y)
#    fd, _ = linregression(x, y)
#    fds.append(fd)
#    paths.append(filename)
#    quality.append("Outlier")

for filename in glob.glob(bad_path + "/*.jpg"):
    edges = (detect_edges(fillalpha(readfile(filename))))
    x, y = boxcount(edges, box_sizes)
    print(x, y)
    fd, _ = linregression(x, y)
    fds.append(fd)
    paths.append(filename)
    quality.append("Bad")


for filename in glob.glob(good_path + "/*.jpg"):
    edges = (detect_edges(fillalpha(readfile(filename))))
    cv2.imwrite("test.png", edges)
    x, y = boxcount(edges, box_sizes)
    print(x, y)
    fd, _ = linregression(x, y)
    fds.append(fd)
    paths.append(filename)
    quality.append("Good")



out = pd.DataFrame({"filename" : paths,
                    "fractal dimension": fds,
                    "quality" : quality})
out.to_csv("fds.csv")
