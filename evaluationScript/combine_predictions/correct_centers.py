import pandas as pd

df = pd.read_csv("predictions_ALL.csv")
df["coordX"] -= 64
df["coordY"] -= 64
df["coordZ"] -= 64

df.to_csv("predictions_ALL.csv", index=False)
