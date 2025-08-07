import pandas as pd
from preprocessing import *
from multiprocessing import Process, Manager
import time
import datetime

manager = Manager()
paths = manager.list()
fds = manager.list()
quality = manager.list()

starttime = 0
total_count = 0

box_sizes = [2 ** n for n in range(1, 8)]

def calc_fd(files, labels, fds, paths, quality):
    for path, label in zip(files, labels):
        edges = detect_edges(high_pass_filter(fillalpha(readfile(path)), 0))
        x, y = boxcount_4x(edges, box_sizes)
        fd, _ = linregression(x, y)
        fds.append(fd)
        paths.append(path)
        quality.append(label)

        elapsed = time.time() - starttime
        rate = (len(paths) - len(processed_files)) / elapsed
        print(f"file {path} done; {len(paths) - len(processed_files)}/{total_count} files finished; estimated {datetime.timedelta(seconds=total_count / rate - elapsed)} remaining")

try:
    already_processed = pd.read_csv("./EyeQ_fds.csv")
    processed_files = set(already_processed["filename"].to_list())

    paths += already_processed["filename"].to_list()
    fds += already_processed["fractal dimension"].to_list()
    quality += already_processed["quality"].to_list()
except:
    processed_files = set()

thread_count = 16

divided_files = []
divided_labels = []

for i in range(thread_count):
    divided_files.append([])
    divided_labels.append([])

eyeq_test_prefix = "./EyeQ/test/"
eyeq_train_prefix = "./EyeQ/train"

test_set = pd.read_csv("./EyeQ/test/Label_EyeQ_test.csv")
train_set = pd.read_csv("./EyeQ/train/Label_EyeQ_train.csv")

counter = 0

for filename, label in zip(test_set["image"].to_list(), test_set["quality"].to_list()):
    if (eyeq_test_prefix + filename in processed_files):
        continue
    divided_files[counter].append(eyeq_test_prefix + filename)
    divided_labels[counter].append(label)
    counter += 1
    counter %= thread_count
    total_count += 1

for filename, label in zip(train_set["image"].to_list(), train_set["quality"].to_list()):
    if (eyeq_train_prefix + filename in processed_files):
        continue
    divided_files[counter].append(eyeq_train_prefix + filename)
    divided_labels[counter].append(label)
    counter += 1
    counter %= thread_count
    total_count += 1

threads = []

for file_group, label_group in zip(divided_files, divided_labels):
    t = Process(target = calc_fd, args = (file_group, label_group, fds, paths, quality))
    threads.append(t)

starttime = time.time()

for t in threads:
    t.start()

try:
    for t in threads:
        t.join()
except:
    out = pd.DataFrame({"filename" : list(paths),
                        "fractal dimension": list(fds),
                        "quality" : list(quality)})
    out.to_csv("EyeQ_fds.csv")
