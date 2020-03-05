#coding=utf-8
import os
from utils import *
import time
from sklearn.ensemble import IsolationForest
from evaluation import label_evaluation
REPEAT_TIMES = 1


def RRCF_origin(use_src_dir, output):
    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir + p)])
    length = len(file_list)
    file_index = {file_list[i]:i for i in range(length)}
    perform = pd.DataFrame({"file": file_list, "storage":[0.0]*length, "train-time":[0.0]*length, "test-time":[0.0]*length})
    for file in file_list:

        train_f, train_tag, train_time, test_f, test_tag, test_time = preprocess(use_src_dir, file, 0.5, 0.5)
        print(file+" test begin.")

        for j in range(REPEAT_TIMES):

            print(str(j) + " times test. training ", end="")

            start = time.time()
            a = IsolationForest(n_estimators = 130, max_samples=1024, n_jobs=1)
            a.fit(X = train_f)
            end = time.time()
            tt = end - start
            perform.loc[file_index[file], "train-time"] += tt


            print("testing")
            start = time.time()
            score = a.decision_function(test_f)
            end = time.time()
            perform.loc[file_index[file], "test-time"] += end - start
            perform.loc[file_index[file], "storage"] += get_size(a)
            pd.DataFrame({"timestamp": test_time, "score":score}).to_csv(output + "test-IF"  + file, index= False)

        perform.iloc[file_index[file], 1:] /= REPEAT_TIMES
        perform.to_csv(output + "performance-IF.csv", index = False)


if __name__ == "__main__":
    use_src_dir = "../contest_data/"
    output_dir =  use_src_dir[1:] + "IF/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    RRCF_origin(use_src_dir, output_dir)

