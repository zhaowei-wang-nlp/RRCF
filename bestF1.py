import pandas as pd
from evaluation import label_evaluation
import os
import sys
import numpy as np
def compute_best_F1(ans_file, co_disp_file, reverse = False):
    # reverse 是True的话代表当前的方法的异常分数越小越可能是异常，否则是异常分越大越可能是异常
    ans_data = pd.read_csv(ans_file)
    co_disp_data = pd.read_csv(co_disp_file)

    ans_data.index = ans_data["timestamp"]
    true_ans = ans_data.loc[co_disp_data.loc[0, "timestamp"]: , "anomaly"].values

    co_disps = co_disp_data["score"].values
    if len(co_disps) != len(true_ans):
        print("the length of ans is not the same")
    best_F1, best_threshold, precision, recall = None, None, None, None
    start, end = np.mean(co_disps), np.max(co_disps)
    step, cur_threshold =  (end - start)/200, start
    for i in range(200):
        if i % 50 == 0:
            print(i, end=" ")
        predict_ans = [1 if d < cur_threshold else 0 for d in co_disps] if reverse else [1 if d > cur_threshold else 0 for d in co_disps]
        data = label_evaluation(predict_ans, true_ans)
        if best_F1 is None or data["F1-score"] >= best_F1:
            best_F1, best_threshold = data["F1-score"], cur_threshold
            precision, recall = data["precision"], data["recall"]
        cur_threshold += step
    print()

    co_disp_data["anomaly"] = [1 if d < best_threshold else 0 for d in co_disps] if reverse else [1 if d > best_threshold else 0 for d in co_disps]
    co_disp_data.to_csv(co_disp_file, index = False)
    return best_F1, best_threshold, precision, recall

def compute_F1_dir(string, batch, batch_size, reverse = False):
    print("start testing" + string)
    true_dir = "../contest_data/"
    predict_dir = "./contest_data/" + string + "/"
    perform = pd.read_csv(predict_dir + "performance-" + string + "-" + str(batch) + ".csv")
    perform.index = perform["file"]
    perform["precision"] = [None] * len(perform)
    perform["recall"] = [None] * len(perform)
    perform["best-F1"] = [None] * len(perform)
    perform["best-threshold"] = [None] * len(perform)
    file_list = sorted([p for p in os.listdir(true_dir) if os.path.isfile(true_dir + p)])
    file_list = file_list[batch_size * batch : min(batch_size * batch + batch_size, len(file_list))]
    for file in file_list:
        best_F1, best_threshold, precision, recall = compute_best_F1(true_dir + file, predict_dir + "test-" + string + file, reverse=reverse)
        perform.loc[file, "best-F1"] = best_F1
        perform.loc[file, "best-threshold"] = best_threshold
        perform.loc[file, "precision"] = precision
        perform.loc[file, "recall"] = recall
        perform.to_csv(predict_dir + "performance-" + string + "-" + str(batch) + ".csv", index = False)


