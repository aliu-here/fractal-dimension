import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./fds_4xboxcount_fft.csv")

fig, ax = plt.subplots(2)

colors = {"Good" : (0, 1, 0), "Bad" : (1, 0, 0), "Outlier" : (1, 1, 0)}
ylocs = {"Good" : 0.1, "Bad" : -0.1, "Outlier" : 0}

good = []
bad = []

for index, rows in data.iterrows():
    if (rows.iloc[3] == "Good"):
        good.append(rows.iloc[2])
    else:
        bad.append(rows.iloc[2])

best_acc = 0
best_sens, best_spec = 0, 0
best_acc_cutoff, best_sens_cutoff, best_spec_cutoff = -1, -1, -1
for cutoff in (good + bad):
    tp, tn, fp, fn = 0, 0, 0, 0
    for val in good:
        if cutoff > val:
            fn += 1
        else:
            tp += 1
    for val in bad:
        if cutoff < val:
            fp += 1
        else:
            tn += 1

    acc = (tp + tn) / (tp + tn + fp + fn)
    try:
        sens = (tp) / (tp + fp)
    except:
        sens = 0

    try:
        spec = (tn) / (tn + fn)
    except:
        spec = 0

    if (acc > best_acc):
        best_acc = acc
        best_acc_cutoff = cutoff
        best_sens = sens
        best_spec = spec
        print(tp, fp, tn, fn)
    
print(f"cutoff with best accuracy: {best_acc_cutoff}; has accuracy {best_acc}, specificity {best_spec}, sensitivity {best_sens}")

good_hist, good_edges = np.histogram(good, 50, (1,2))
bad_hist, bad_edges = np.histogram(bad, 50, (1, 2))

ax[0].stairs(good_hist, good_edges, fill=True)
ax[1].stairs(bad_hist, bad_edges, fill=True)

ax[0].set_title("Fractal dimension of good images")
ax[1].set_title("Fractal dimension of bad images")

ax[0].axvline(x = best_acc_cutoff, c = (0, 1, 0))
ax[1].axvline(x = best_acc_cutoff, c = (0, 1, 0))


ax[0].set_xlim(1, 1.75)
ax[1].set_xlim(1, 1.75)

plt.show()
