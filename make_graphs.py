from pathlib import Path
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from numpy import load
#Statistics functions
def RMSE(predicted, actual):
  return math.sqrt(sum((predicted - actual)**2)/len(predicted))

def MAE(predicted, actual):
  return sum(abs(predicted - actual))/len(predicted)

def MAPE( predicted,  actual):
  return sum(abs((actual - predicted)/ actual))/len(predicted)

#Creates a plot
#pred is the dataframe with the predicted data
#actual is the dataframe with the original data
    #For the best results, pred and actual should only contain one column each
#start and end determines the interval that will be plotted
#title, y, and x can be customized. They are just labels for the plot itself / axis
#extras is an array of data frames (similar to pred and actual). This is here if you want to plot something else alongside pred and actual
def plot(pred, actual, start, end, title="Prediction vs Actual", y="VWC (mm)", x="Time", extras=[]):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.plot(pred[start:end], label="Predicted")
    ax.plot(actual[start:end], label="Actual")
    for extra in extras:
        ax.plot(extra[start:end], label=extra.name)
    ax.legend()
    plt.show()


#Prints all the statistics.
#This works best if you pass in the specific interval you want statistics on
#For example df[column][0:100]
def statisics(pred, actual):
  if type(pred) != torch.Tensor:
    p = torch.tensor(pred.values)
  else:
    p = pred
  if type(actual) != torch.Tensor:
    a = torch.tensor(actual.values)
  else:
    a = actual

  rmse = RMSE(p, a)
  mae = MAE(p, a)
  mape = MAPE(p, a)
  if type(rmse) == torch.Tensor:
    rmse = rmse.item()
  if type(mae) == torch.Tensor:
    mae = mae.item()
  if type(mape) == torch.Tensor:
    mape = mape.item()
  print("RMSE: %s" % (rmse))
  print("MAE: %s" % (mae))
  print("MAPE: %s" % (mape))
#denormalizes the df using the std and mean
def denormalize(df, std, mean):
  return df * (std + np.finfo(float).eps) + mean
p = "SMPModel_2024-10-17_15-28-44_P3y"
path = f"experiments/{p}/predictions/best_predictions.npz"
pathnorm = f"experiments/{p}/normalization.pickle"
#actual is the data without the gap
#gaps is the data with the gap
actual = pd.read_pickle("data/OriginalSMPData.pkl")
gaps = pd.read_pickle("data/SMPSKLearnGap.pkl")
start = 1440
end = 4320
testSet = gaps[start:end]
title = "SkLearn Filled Data - Epoch 300, mean mask length 50, batch size 128, Window Length 350"
#data is the best predictions
data = load(path, allow_pickle=True)

#Norm contains all the std and means for denormization purposes
norm = load(pathnorm, allow_pickle=True)

#In best predictions, ther are several sections, and predictions is the name of the predictions that we want
pred = data['predictions']
print(pred.shape)
#The predictions are of shape num batches, batch size, window size, row size
#This converts that format into a standard df
dfList = []
for block in pred:
  for batch in block:
    for row in batch:
      dfList.append(row)

#This formats the df
df = pd.DataFrame(dfList)
df.columns = actual.columns

#The predictions do not control dates (because the indices need to be related to the windows, and dtaes could not be processed)
#This maps the predictions to their timestamps
dates = testSet.index
matching_rows = actual.loc[actual.index.isin(dates)]

#Denormizing each column
#Doing it all at once fails
for column in df.columns:
  df[column] = denormalize(df[column], norm['std'][column], norm['mean'][column])

#Sometimes the prediction is larger than the gap due to the transoformer filling in for extra space if there are not enough batches to fill the entire batch size
#this fixes the issue by trimming off the excess
df = df[:len(matching_rows)]
df.index = matching_rows.index

intervals = {"P2_VWC": [0, 2160-start]}
for col in intervals:
    statisics(df[col][intervals[col][0]:intervals[col][1]], matching_rows[col][intervals[col][0]:intervals[col][1]])
for col in intervals:
    plot(df[col], matching_rows[col], 0, len(df[col]), title="%s %s" % (title, "P2_VWC"))
for col in intervals:
    plot(df[col], matching_rows[col], intervals[col][0], intervals[col][1], title = "%s %s" % (title, "P2_VWC"))