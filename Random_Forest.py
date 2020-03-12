import os
import time
from utils import *
from sklearn.ensemble import RandomForestClassifier
import json
import pickle
from evaluation import label_evaluation
REPEAT_TIMES = 1

def Random_Forest(use_src_dir, output, type = "full", label_type = None):
    # read clusters' info

    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir+p)])
    length = len(file_list)
    file_index = {file_list[i]: i for i in range(length)}

    perform = pd.DataFrame({"file": file_list, "recall": [0.0] * length, "precision": [0.0] * length, "F1-score": [0.0] * length,
         "storage": [0.0] * length, "train-time": [0.0] * length, "test-time": [0.0] * length})

    for file in file_list:
        print(file + " test begin.")
        train_f, train_tag, train_time, test_f, test_tag, test_time = preprocess(use_src_dir, file, 0.5, 0.5)

        if type == "part":
            indices = pd.read_csv("./active/" + label_type + "/" + file)["indices"].values
            train_f, train_tag, train_time = train_f[indices], train_tag[indices], train_time[indices]

        for j in range(REPEAT_TIMES):

            print(str(j) + "times test. training ", end="")
            start = time.time()
            a = RandomForestClassifier(n_estimators= 70, max_features= None,n_jobs = 1)
            a.fit(X= train_f, y = train_tag)
            end = time.time()
            tt = end - start
            perform.loc[file_index[file], "train-time"] += tt

            perform.loc[file_index[file], "storage"] += get_size(a)

            print("testing")
            start = time.time()
            predict = a.predict(test_f)
            end = time.time()
            perform.loc[file_index[file], "test-time"] += end - start

            data = label_evaluation(predict, test_tag)
            pd.DataFrame({"timestamp": test_time, "anomaly": predict}).to_csv(output + "test-" + file, index=False)

            perform.loc[file_index[file], "F1-score"] += data["F1-score"]
            perform.loc[file_index[file], "recall"] += data["recall"]
            perform.loc[file_index[file], "precision"] += data["precision"]
    perform.iloc[:, 1:] /= REPEAT_TIMES
    name = "RF" if type == "part" else "full-RF"
    perform.to_csv(output + "performance-" + name + ".csv", index = False)


if __name__ == "__main__":
    use_src_dir = "../contest_data/"
    output_dir =  use_src_dir[1:] + "RF/"
    if not os.path.exists(output_dir + "/"):
        os.mkdir(output_dir + "/")
    Random_Forest(use_src_dir, output_dir, type = "part", label_type= "6.1")