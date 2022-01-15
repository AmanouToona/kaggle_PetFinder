# show learning rate & score
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("output/SimpleModelSwin_001_fold_00_eval.csv", header=0, index_col=0)

fig, ax = plt.subplots()

ax.plot(df["learning_rate"], df["valid_eval"], label="valid")
ax.plot(df["learning_rate"], df["train_eval"], label="train")
ax.invert_xaxis()
fig.legend()

plt.show()
