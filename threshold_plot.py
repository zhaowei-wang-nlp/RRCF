import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
def our_threshold(mean, std, score, timestamp):
    period = int(10080 / ((timestamp[1] - timestamp[0]) / 60))
    threshold = mean + std
    while True:
        while True:
            predict = np.array([1 if s > threshold else 0 for s in score])
            if predict.sum() > len(predict) * 0.15:
                threshold += 0.3 * std
            else:
                break
        week = []
        for i in range(len(score) // period):
            week.append(predict[i * period: i * period + period].sum())
        week_mean, week_std = np.mean(week), np.std(week)
        flag = False
        for i in range(len(score) // period):
            if week[i] > week_mean + week_std and week[i] > 10:
                threshold += 0.4 * std
                flag = True
        if not flag:
            break
    return threshold

if __name__ == "__main__":
    dir = "../3.4/3.6/"
    perform = pd.read_csv(dir + "performance-3.6.csv")
    perform.index = perform["file"]
    for file in os.listdir("../contest_data/"):
        data = pd.read_csv(dir + "test-3.6" + file)
        data["time"] = pd.to_datetime(data["timestamp"], unit='s')
        plt.figure(figsize=(19.2, 13.4), dpi=100)
        plt.scatter(data["time"], data["score"], s=30,
                    label="3.6")
        plt.axhline(y=perform.loc[file, "best-threshold"], ls="-", c="green")
        plt.axhline(y=perform.loc[file, "mean5std"], ls="-", c="red")
        plt.axhline(y=perform.loc[file, "mean"] + 3 * perform.loc[file, "std"], ls="-", c="red")
        plt.axhline(y=perform.loc[file, "mean"] + 1 * perform.loc[file, "std"], ls="-", c="red")
        threshold = our_threshold(perform.loc[file, "mean"], perform.loc[file, "std"], score = data["score"], timestamp= data["timestamp"])
        plt.axhline(y=threshold, ls="-", c="purple")
        plt.savefig("threshold_pic/" + file + ".jpg")