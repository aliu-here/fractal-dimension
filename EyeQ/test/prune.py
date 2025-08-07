import pandas
import glob
import os

eyeQ_files = pandas.read_csv("Label_EyeQ_test.csv")

eyeQ_file_dict = set(eyeQ_files["image"].to_list())

files = [file[2:] for file in glob.glob("./*.jpeg")]

del_list = []

for file in files:
    if file not in eyeQ_file_dict:
        del_list.append(file)

del_list = ["./" + file for file in del_list]

print(del_list)

for file in del_list:
    os.remove(file)
