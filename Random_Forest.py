import os
import warnings
import time
from utils import *
from sklearn.ensemble import RandomForestClassifier
import json
from evaluation import label_evaluation
REPEAT_TIMES = 5
def RandomForest_test(use_src_dir, output):
    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir + p)])
    length = len(file_list)
    file_index = {file_list[i]: i for i in range(length)}
    perform = pd.DataFrame(
        {"file": file_list, "recall": [0.0] * length, "precision": [0.0] * length, "F1-score": [0.0] * length,
         "storage": [0.0] * length, "time": [0.0] * length})
    for file in file_list:
        train_f, train_tag, train_time, test_f, test_tag, test_time = preprocess(use_src_dir, file, 0.5, 0.5)
        print(file + " test begin.")

        if st.PART_LABEL:
            train_f, train_tag, train_time = preprocess("./part_label/", file, 1.0, 0.0)[3:]

        best_performance = -1
        for j in range(REPEAT_TIMES):

            print(str(j) + " times test. training ", end="")

            start = time.time()
            a = RandomForestClassifier(n_estimators= 70, max_features= None)
            a.fit(X = train_f, y = train_tag)
            predict = a.predict(test_f)
            end = time.time()

            perform.loc[file_index[file], "time"] += end - start
            perform.loc[file_index[file], "storage"] += get_size(a)

            data = label_evaluation(predict, test_tag)

            if data["F1-score"] > best_performance:
                best_performance = data["F1-score"]
                pd.DataFrame({"timestamp": test_time, "anomaly": predict}).to_csv(output + "labels-" + file, index= False)
                plot_points(use_src_dir+file, output + "labels-" + st.STRING + file, "")

            perform.loc[file_index[file], "F1-score"] += data["F1-score"]
            perform.loc[file_index[file], "recall"] += data["recall"]
            perform.loc[file_index[file], "precision"] += data["precision"]
            print(data)
    perform.iloc[:, 1:] /= REPEAT_TIMES
    perform.to_csv(output + "performance.csv", index= False)

if __name__ == "__main__":
    use_src_dir = "../single-data/"
    RandomForest_test(use_src_dir, use_src_dir[1:])