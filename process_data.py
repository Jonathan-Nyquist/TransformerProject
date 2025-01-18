import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os


# Statistics functions
def RMSE(predicted, actual):
    return math.sqrt(sum((predicted - actual) ** 2) / len(predicted))


def MAE(predicted, actual):
    return sum(abs(predicted - actual)) / len(predicted)


def MAPE(predicted, actual):
    return sum(abs((actual - predicted) / actual)) / len(predicted)


# Creates a plot
# pred is the dataframe with the predicted data
# actual is the dataframe with the original data
# For the best results, pred and actual should only contain one column each
# start and end determines the interval that will be plotted
# title, y, and x can be customized. They are just labels for the plot itself / axis
# extras is an array of data frames (similar to pred and actual). This is here if you want to plot something else alongside pred and actual
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


# Prints all the statistics.
# This works best if you pass in the specific interval you want statistics on
# For example df[column][0:100]
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


# Adds statistics to the stats file
# Note this only works if a stats file is already pre
def addStats(methodName, pred, actual, start, end, path="stats.csv", desc="Default",
             dataset="SMPNotFilled"):
    if os.path.exists(path):
        stats = pd.read_csv(path, index_col=False)
    else:
        # Method is the name of the method. For example, SKLearn, MLP, etc
        # Description is for any misc info
        # Dataset is used if there are multiple datasets being worked on simultaneously.
        # There are 2 Gap Lengths: one for the number of rows, one for the time duration.
        # The rest are the statistics
        stats = pd.DataFrame(
            columns=["Method", "Description", "Dataset", "Gap Length (rows)", "Gap Length (time)", "RMSE", "MAE",
                     "MAPE"])

    method = methodName
    rmse = RMSE(pred[start:end], actual[start:end])
    mae = MAE(pred[start:end], actual[start:end])
    mape = MAPE(pred[start:end], actual[start:end])
    intervalLen = end - start
    timeLen = actual.index[end] - actual.index[start]
    stats.loc[len(stats)] = [method, desc, dataset, intervalLen, timeLen, rmse, mae, mape]

    stats.to_csv(path, index=False)


# denormalizes the df using the std and mean
def denormalize(df, std, mean):
    return df * (std + np.finfo(float).eps) + mean


# function used to locate the gap in a certain column
# Only really useful for test cases
def getGap(df):
    start = -1
    end = -1
    for i in range(len(df)):
        if math.isnan(df[i]):
            if start == -1:
                start = i
        if not math.isnan(df[i]) and start != -1:
            end = i
            break
    return start, end


def getGapAll(df):
    gaps = []  # To store start and end of each NaN gap
    start = -1  # Initialize start
    end = -1  # Initialize end

    for i in range(len(df)):
        # Check if the current element is NaN
        if math.isnan(df[i]):
            if start == -1:
                start = i  # Mark the start of a NaN gap
        # If the current element is not NaN and there's a gap recorded
        if not math.isnan(df[i]) and start != -1:
            end = i  # Mark the end of the NaN gap
            gaps.append([start, end - 1])  # Record the gap (end - 1 because it's non-NaN)
            start = -1  # Reset start for the next gap

    # If the loop ends and there's still an ongoing NaN gap
    if start != -1:
        gaps.append([start, len(df) - 1])  # Handle case where NaN gap ends at the last element

    return gaps


def getNonNanData(total_length, gaps):
    non_nan_ranges = []  # To store the ranges of non-NaN data
    current_index = 0  # Start checking from index 0

    for gap in gaps:
        start, end = gap  # Unpack the start and end of the NaN gap

        # If there's non-NaN data before the current gap, add the range
        if current_index < start:
            non_nan_ranges.append([current_index, start - 1])

        # Move current index to the end of the NaN gap
        current_index = end + 1

    # Add the final range if there's non-NaN data after the last gap
    if current_index < total_length:
        non_nan_ranges.append([current_index, total_length - 1])

    return non_nan_ranges

data = pd.read_pickle('data/SMPSKLearnGap.pkl')
# Find the gap. This will be used later for plotting
nan_gap = getGapAll(data["P2_VWC"])
print(nan_gap)

start = nan_gap[0][0]
end = nan_gap[0][1]
# Separate validation set from training set.
# I do not know why, but removing the test set from the validation set is necessary, otherwise the predictions are all over the place
valData = data[start:end]
data = data.drop(data.index[start:end])

# Write test and train sets to local directory
data.to_csv("data/SMPSKLearn/SMP_TRAIN.csv")
valData.to_csv("data/SMPSKLearn/SMP_TEST.csv")
