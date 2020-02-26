import numpy as np
import pandas as pd
from evaluation import label_evaluation
import os
def compute_best_F1(ans_file, co_disp_file ):
    ans_data = pd.read_csv(ans_file)
    co_disp_data = pd.read_csv(co_disp_file)

    ans_data.index = ans_data["timestamp"]
    true_ans = ans_data.loc[co_disp_data.loc[0, "timestamp"]: , "anomaly"].values

    co_disps = co_disp_data["score"].values
    if len(co_disps) != len(true_ans):
        print("the length of ans is not the same")

    best_F1, best_threshold = None, None
    step, cur_threshold =  max(co_disps)/1000, 0
    for i in range(1000):
        if i % 100 == 0:
            print(i, end=" ")
        predict_ans = [1 if d > cur_threshold else 0  for d in co_disps]
        data = label_evaluation(predict_ans, true_ans)
        if best_F1 is None or data["F1-score"] >= best_F1:
            best_F1, best_threshold = data["F1-score"], cur_threshold
        cur_threshold += step
    print()
    return best_F1, best_threshold

if __name__ == "__main__":
    string  = "6.3"
    print(string)
    true_dir = "../contest_data/"
    predict_dir = "./contest_data/" + string + "/"
    perform = pd.read_csv(predict_dir + "performance-" + string + ".csv")
    perform.index = perform["file"]
    perform["best-F1"] = [None] * len(perform)
    perform["best-threshold"] = [None] * len(perform)

    for file in os.listdir(true_dir):
        best_F1, best_threshold = compute_best_F1(true_dir + file, predict_dir + "test-"+ string + file)
        perform.loc[file, "best-F1"] = best_F1
        perform.loc[file, "best-threshold"] = best_threshold
        perform.to_csv(predict_dir + "performance-" + string + ".csv", index = False)


