import numpy as np
import pandas as pd
import os
import pickle
import json

dir = "../3.4/"
sim_data = pickle.load(open(dir + "similarity_dict.dat", "rb"))
def find_nearest(cluster):
    cluster = [c[:-4] for c in cluster if c[-4:] == ".csv"]
    sum = {c : 0 for c in cluster}
    for c1 in cluster:
        for c2 in cluster:
            sum[c1] += sim_data[c1][c2][0]

    max, res = None, None
    for c1 in cluster:
        if max is None or sum[c1] > max:
            max, res = sum[c1], c1
    return res + '.csv'
if __name__== "__main__":

    data = pd.read_csv(dir + "3.6/performance-3.6.csv")
    clusters = json.load(open(dir + "file_clusters.txt", "r"))
    data["mean"] = [None] * len(data)
    data["std"] = [None] * len(data)
    data["mean5std"] = [None] * len(data)
    data.index = data["file"]
    for c in clusters:
        central = find_nearest(clusters[c])
        train_file = dir + "3.6/train-3.6" + central
        train_co_disp = pd.read_csv(train_file)["score"].values
        mean = np.mean(train_co_disp)
        std = np.std(train_co_disp)
        print(max(train_co_disp))
        threshold = mean + 5 * std
        for file in clusters[c]:
            test_file = dir + "3.6/test-3.6" + file
            test_data = pd.read_csv(test_file)
            test_co_disp = test_data["score"].values
            predict = [1 if d >= threshold else 0 for d in test_co_disp]
            data.loc[file, "mean"] = mean
            data.loc[file, "std"] = std
            data.loc[file, "mean5std"] = threshold
            test_data["5std"] = predict
            test_data.to_csv(test_file, index = False)
    data.to_csv(dir + "3.6/performance-3.6.csv", index = False)