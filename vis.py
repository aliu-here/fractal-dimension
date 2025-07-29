import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./fds.csv")

fig, ax = plt.subplots(1)

colors = {"Good" : (0, 1, 0), "Bad" : (1, 0, 0), "Outlier" : (1, 1, 0)}
ylocs = {"Good" : 0.1, "Bad" : -0.1, "Outlier" : 0}

for index, rows in data.iterrows():
    ax.scatter(rows.iloc[2], ylocs[rows.iloc[3]], c=colors[rows.iloc[3]])

plt.show()
