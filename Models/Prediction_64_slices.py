
# coding: utf-8

from keras.models import load_model
import h5py
import numpy as np

# import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve, auc
import pandas as pd

import argparse
parser = argparse.ArgumentParser(description='Prediction on HOLDOUT subset',add_help=True)

parser.add_argument("--holdout", type=int, default=0, help="HOLDOUT subset for predictions")
parser.add_argument("--modelFileName", default="wrong_model_file.hdf5", help="Name of model file to load")
args = parser.parse_args()

model_file = args.modelFileName
HOLDOUT = args.holdout

model = load_model(model_file)

df = h5py.File("64x64x64-patch.hdf5", "r")
idx_subset = np.where(df["subsets"][:,0] == HOLDOUT)[0]

imgs = df["input"][idx_subset, :]
imgs_reshaped = imgs.reshape(len(idx_subset), 64, 64, 64, 1).swapaxes(1,3)
predictions = model.predict(imgs_reshaped, verbose=2)

y_true = df["output"][idx_subset, :]
cnf = confusion_matrix(y_true, np.round(predictions))
np.savetxt("confusion_matrix{}.txt".format(HOLDOUT), cnf, delimiter=',')

print("Average Precision score = {}".format( average_precision_score(y_true, predictions) ))
np.savetxt("precision{}.txt".format(HOLDOUT), average_precision_score(y_true, predictions))

print("Prediction shape = {}".format(predictions.shape)
print("True labels shape = {}".format(y_true.shape)

d = np.array([predictions, y_true])

dfp = pd.DataFrame(d[:,:,0].transpose())
dfp.columns = ["Prediction", "Truth"]
dfp.to_csv("predictions_truth_subset{}.csv".format(HOLDOUT))

df_out = pd.DataFrame((df["uuid"][idx_subset,:]).astype(str))
df_out.columns = ["seriesuid"]

df_out["coordX"] = df["centroid"][idx_subset, 0]
df_out["coordY"] = df["centroid"][idx_subset, 1]
df_out["coordZ"] = df["centroid"][idx_subset, 2]

df_out["probability"] = dfp["Prediction"]
df_out.to_csv("predictions_subset{}.csv".format(HOLDOUT), index=False)
