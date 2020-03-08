import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def our_threshold(mean, std, score, timestamp):
    period = int(24 * 60 / ((timestamp[1] - timestamp[0]) / 60))
    threshold = mean + 2 * std
    while threshold < mean + 6 * std:
        predict = np.array([1 if s > threshold else 0 for s in score])
        if predict.sum() > len(predict) * 0.15:
            threshold += 0.3 * std
        else:
            break
    while threshold < mean + 6 * std:
        predict = np.array([1 if s > threshold else 0 for s in score])
        hours = []
        for i in range(len(score) // period):
            hours.append(predict[i * period: i * period + period].sum())
        hour_mean, hour_std = np.mean(hours), np.std(hours)
        count = 0
        for i in range(len(hours)):
            if hours[i] > hour_mean + 3 * hour_std and hours[i] > 30:
                count += 1
        if count < len(hours) // 3:
            threshold += 0.3 * std
        else:
            break
    return threshold

if __name__ == "__main__":
    dir = "../3.5-不聚类/"
    version = "6.1"
    perform = pd.read_csv(dir + version + "/performance-" + version + ".csv")
    perform.index = perform["file"]
    for file in os.listdir("../contest_data/"):
        train_data = pd.read_csv(dir + version + "/train-" + version + file)
        train_score = train_data["score"].values
        test_data = pd.read_csv(dir + version + "/test-" + version + file)
        test_data["time"] = pd.to_datetime(test_data["timestamp"], unit='s')
        plt.figure(figsize=(19.2, 13.4), dpi=100)
        plt.scatter(test_data["time"], test_data["score"], s=30,
                    label=version)
        plt.axhline(y=perform.loc[file, "best-threshold"], ls="-", c="green")
        #plt.axhline(y=perform.loc[file, "mean5std"], ls="-", c="red")
        #plt.axhline(y=perform.loc[file, "mean"] + 3 * perform.loc[file, "std"], ls="-", c="red")
        #plt.axhline(y=perform.loc[file, "mean"] + 1 * perform.loc[file, "std"], ls="-", c="red")
        threshold = our_threshold(np.mean(train_score), np.std(train_score), train_score, train_data["timestamp"].values)
        plt.axhline(y=threshold, ls="-", c="purple")
        plt.savefig("threshold_pic/" + file + ".jpg")