drimdb_path = "./DRIMDB/"
good_path = drimdb_path + "Good"
bad_path = drimdb_path + "Bad"
outlier_path = drimdb_path + "Outlier"

from preprocessing import *
import glob
import pandas as pd
from multiprocessing import Process, Manager


manager = Manager()
paths = manager.list()
fds = manager.list()
quality = manager.list()

box_sizes = [2 ** n for n in range(1, 7)]

def calc_fd(path, label, fds, paths, quality):
    edges = detect_edges(fillalpha(readfile(path)))
    x, y = boxcount_4x(edges, box_sizes)
    fd, _ = linregression(x, y)
    print(fd, path, label)
    fds.append(fd)
    paths.append(path)
    quality.append(label)

threads = []

for filename in glob.glob(bad_path + "/*.jpg"):
    #    calc_fd(filename, "Bad")
    t = Process(target=calc_fd, args=(filename, "Bad", fds, paths, quality))
    threads.append(t)

for filename in glob.glob(good_path + "/*.jpg"):
    #calc_fd(filename, "Good")
    t = Process(target=calc_fd, args=(filename, "Good", fds, paths, quality))
    threads.append(t)
    cv2.imwrite("test.png", detect_edges(fillalpha(readfile(filename))))

for t in threads:
    t.start()

for t in threads:
    t.join()

print(paths, fds, quality)

out = pd.DataFrame({"filename" : list(paths),
                    "fractal dimension": list(fds),
                    "quality" : list(quality)})
out.to_csv("fds_4xboxcount_sobelfilter.csv")
