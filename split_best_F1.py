import numpy as np
import pandas as pd
import os
from evaluation import get_range_proba
from evaluation import label_evaluation
from sklearn.metrics import f1_score, precision_score, recall_score
def compute_best_F1(ans_file, co_disp_file):
    # reverse 是True的话代表当前的方法的异常分数越小越可能是异常，否则是异常分越大越可能是异常
    ans_data = pd.read_csv(ans_file)
    co_disp_data = pd.read_csv(co_disp_file)

    median = np.median(ans_data["value"].values)
    ans_data.index = ans_data["timestamp"]
    ans_data = ans_data.loc[co_disp_data.loc[0, "timestamp"]:, :]
    true_ans = ans_data.loc[: , "anomaly"].values

    upper_indicator = ans_data["value"].values > median
    lower_indicator = ans_data["value"].values <= median
    co_disps = co_disp_data["score"].values
    if len(co_disps) != len(true_ans):
        print("the length of ans is not the same")
    upper_F1, upper_threshold, upper_p, upper_r = None, None, None, None
    lower_F1, lower_threshold, lower_p, lower_r = None, None, None, None
    start, end = np.mean(co_disps), np.max(co_disps)
    step, cur_threshold =  (end - start)/200, start
    for i in range(200):
        if i % 50 == 0:
            print(i, end=" ")
        predict_ans = [1 if d > cur_threshold else 0 for d in co_disps]
        predict_ans = get_range_proba(predict_ans, true_ans)

        F1_score = f1_score(true_ans[upper_indicator], predict_ans[upper_indicator])
        if upper_F1 is None or F1_score >= upper_F1:
            upper_F1, upper_threshold = F1_score, cur_threshold
            upper_p, upper_r = precision_score(true_ans[upper_indicator], predict_ans[upper_indicator]), recall_score(true_ans[upper_indicator], predict_ans[upper_indicator])
            co_disp_data["upper_anomaly"] = [1 if predict_ans[i] and upper_indicator[i] else 0 for i in range(len(predict_ans))]
        F1_score = f1_score(true_ans[lower_indicator], predict_ans[lower_indicator])
        if lower_F1 is None or F1_score >= lower_F1:
            lower_F1, lower_threshold = F1_score, cur_threshold
            lower_p, lower_r = precision_score(true_ans[lower_indicator], predict_ans[lower_indicator]), recall_score(true_ans[lower_indicator], predict_ans[lower_indicator])
            co_disp_data["lower_anomaly"] = [1 if predict_ans[i] and lower_indicator[i] else 0 for i in range(len(predict_ans))]
        cur_threshold += step
    print()
    co_disp_data.to_csv(co_disp_file, index = False)
    return upper_F1, upper_threshold, upper_p, upper_r, lower_F1, lower_threshold, lower_p, lower_r

if __name__ == "__main__":
    dir = "../3.5-不聚类/"
    true_dir = "../contest_data/"
    version = "6.1"
    back_up = pd.read_csv(dir + version + "/" + "performance-" + version + ".csv")
    perform = pd.DataFrame({"file": back_up["file"], "upper_F1": [None] * len(back_up), "upper_threshold": [None] * len(back_up),
                            "upper_p": [None] * len(back_up), "upper_r": [None] * len(back_up), "lower_F1": [None] * len(back_up),
                            "lower_threshold": [None] * len(back_up), "lower_p": [None] * len(back_up), "lower_r": [None] * len(back_up),})
    file_index = {}
    for i in range(len(back_up["file"])):
        file_index[back_up["file"][i]] = i
    for file in os.listdir(true_dir):
        all = compute_best_F1(true_dir + file, dir + version + "/test-" + version + file)
        perform.iloc[file_index[file], 1:] = all
        perform.to_csv(dir + version + "/" + "split-" + version + ".csv", index=False)