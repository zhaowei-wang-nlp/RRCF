import os
from utils import *
import time
from collections import Counter
from rrcf import RRCF
from evaluation import label_evaluation
REPEAT_TIMES = 5


def RRCF_test(use_src_dir):
    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir + p)])
    length = len(file_list)
    file_dict = {file_list[i]:i for i in range(length)}
    perform = pd.DataFrame({"file": file_list, "recall": [0.0]*length, "precision":[0.0]*length, "F1-score":[0.0]*length,
    "storage":[0.0]*length, "time":[0.0]*length})

    for file in file_list:
        train_f, train_tag, test_f, test_tag = preprocess(use_src_dir, file, 0.6, 0.4)
        print(file+" test begin.")
        file_perform = pd.DataFrame({"times":list(range(REPEAT_TIMES)), "recall": [0.0]*REPEAT_TIMES,
        "precision":[0.0]*REPEAT_TIMES, "F1-score":[0.0]*REPEAT_TIMES,
                                     "storage":[0.0]*REPEAT_TIMES, "time":[0.0]*REPEAT_TIMES})
        for j in range(REPEAT_TIMES):
            print(str(j) + " times test. training ", end="")
            start = time.time()
            a = RRCF(tree_num = 70, tree_size = 1024, top = 0.002)
            a.fit(X = train_f)
            print("testing")
            codisp, predict = a.predict(test_f)
            #score_data = pd.DataFrame({"anomaly": test_tag, "anomaly_score": codisp})
            #score_data.to_csv("anomaly_score.csv", index=False)
            #print(123), exit()
            end = time.time()
            file_perform.loc[j, "time"] += end - start
            file_perform.loc[j, "storage"] += get_size(a)
            data = label_evaluation(predict, test_tag)
            file_perform.loc[j, "F1-score"] += data["F1-score"]
            file_perform.loc[j, "precision"] += data["precision"]
            file_perform.loc[j, "recall"] += data["recall"]
            print(data)
        perform.iloc[file_dict[file], 1:] = file_perform.iloc[:, 1:].sum()/REPEAT_TIMES
        file_perform.to_csv(file, index = False)
        break# TODO remember to delete this point
    perform.to_csv("performance.csv", index = False)


if __name__ == "__main__":
    use_src_dir = sys.argv[2] if len(sys.argv) > 2 else "../single-data/"
    RRCF_test(use_src_dir)

