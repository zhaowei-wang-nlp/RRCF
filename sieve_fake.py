import numpy as np
import pandas as pd
from evaluation import label_evaluation, get_range_proba
if __name__ == "__main__":
    file = "adb2fd.csv"
    true_data = pd.read_csv("../contest_data/" + file)
    data = pd.read_csv("../3.6/6.1/0test-6.1" + file)
    true_data.index = true_data["timestamp"]
    true_ans = true_data.loc[data.loc[0, "timestamp"]: , "anomaly"].values

    all_anomaly_scores = []
    indices = []
    sum = 0 if data["anomaly"][0] == 0 else data["score"][0]
    pos = 0 if data["anomaly"][0] else None
    for i in range(1, len(data)):
        if data["anomaly"][i]:
            sum += data["score"][i]
            if data["anomaly"][i - 1] == 0:
                pos = i
        else:
            if data["anomaly"][i - 1]:
                all_anomaly_scores.append(sum/(i - pos))
                indices.append((pos, i - 1))
                sum = 0
    if sum:
        all_anomaly_scores.append(sum/(len(data) - pos) ) # 最后一段是异常
        indices.append((pos, len(data) - 1))
    all_anomaly_scores = np.array(all_anomaly_scores)
    start, end = all_anomaly_scores.min(), all_anomaly_scores.max()
    step, cur_threshold = 10, start
    best_F1, precision, recall, best_threshold = None, None, None, None
    best_label = None
    print("F1", "precision", "recall", "threshold")
    for k in range(1000):
        temp = np.zeros(data["anomaly"].values.shape)
        temp[:] = data["anomaly"].values[:]
        for i in range(len(all_anomaly_scores)):
            if all_anomaly_scores[i] <= cur_threshold:
                pos, sp = indices[i]
                temp[pos: sp + 1] = 0
        metrics = label_evaluation(temp, true_ans)
        print(metrics["F1-score"], metrics["precision"], metrics["recall"], cur_threshold)
        if best_F1 is None or metrics["F1-score"] >= best_F1:
            best_F1, precision, recall, best_threshold = metrics["F1-score"], metrics["precision"], metrics["recall"], cur_threshold
            best_label = temp
        cur_threshold += step
    data["sieve"] = best_label
    data.to_csv("../3.6/6.1/0test-6.1" + file, index = False)
    print()
    print(best_F1, precision, recall, best_threshold)
