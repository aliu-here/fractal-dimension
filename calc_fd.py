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

box_sizes = [2 ** n for n in range(1, 8)]

def calc_fd(files, labels, fds, paths, quality):
    for path, label in zip(files, labels):
        edges = detect_edges(high_pass_filter(fillalpha(readfile(path)), 0))
        edges = edges[:,:] > 127
        x, y = boxcount_4x(edges, box_sizes)
        fd, _ = linregression(x, y)
        print(fd, path, label)
        fds.append(fd)
        paths.append(path)
        quality.append(label)
thread_count = 16

divided_files = []
divided_labels = []
for i in range(thread_count):
    divided_files.append([])
    divided_labels.append([])
counter = 0

threads = []

for filename in glob.glob(bad_path + "/*.jpg"):
    divided_files[counter].append(filename)
    divided_labels[counter].append("Bad")
    counter += 1
    counter %= thread_count

for filename in glob.glob(good_path + "/*.jpg"):
    divided_files[counter].append(filename)
    divided_labels[counter].append("Good")
    counter += 1
    counter %= thread_count

for file_group, label_group in zip(divided_files, divided_labels):
    t = Process(target = calc_fd, args = (file_group, label_group, fds, paths, quality))
    threads.append(t)

for t in threads:
    t.start()

for t in threads:
    t.join()

print(paths, fds, quality)

out = pd.DataFrame({"filename" : list(paths),
                    "fractal dimension": list(fds),
                    "quality" : list(quality)})
out.to_csv("fds_4xboxcount_fft.csv")
