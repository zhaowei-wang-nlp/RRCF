import numpy as np
import pandas as pd
import os
from evaluation import label_evaluation, get_range_proba


def our_threshold3(mean, std, score, timestamp):
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
    back_up = pd.read_csv(dir + version + "/" + "performance-" + version + ".csv")
    perform = pd.DataFrame({"file": back_up["file"], "F1-score": [None] * len(back_up), "threshold": [None] * len(back_up),
         "precision": [None] * len(back_up), "recall": [None] * len(back_up)})
    perform.index = perform["file"]
    for file in os.listdir("../contest_data/"):
        train_data = pd.read_csv(dir + version + "/" + "train-" + version + file)
        train_score = train_data["score"].values
        mean = np.mean(train_score)
        std = np.std(train_score)
        threshold = our_threshold3(mean, std, train_score, train_data["timestamp"].values)

        test_data = pd.read_csv(dir + version + "/" + "test-6.1" + file)
        true_data = pd.read_csv("../contest_data/" + file)
        true_data.index = true_data["timestamp"]
        true_data = true_data.loc[test_data["timestamp"][0]:, :]

        predict = get_range_proba([ 1 if d > threshold else 0 for d in test_data["score"]], true_data["anomaly"].values)
        metrics = label_evaluation(predict, true_data["anomaly"].values)
        perform.loc[file, "F1-score"] = metrics["F1-score"]
        perform.loc[file, "precision"] = metrics["precision"]
        perform.loc[file, "recall"] = metrics["recall"]
        perform.loc[file, "threshold"] = threshold
        perform.to_csv(dir + version + "/" + "our-threshold" + ".csv", index=False)



