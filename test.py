import os
from utils import *
import time
from collections import Counter
from rrcf import RRCF
from evaluation import label_evaluation
import pickle
REPEAT_TIMES = 1
import json
ts = json.load(open("../train_size.json"))
sim_data = pickle.load(open("similarity_dict.dat", "rb"))
def find_nearest(cluster: list) -> str:
    cluster = [c[:-4] for c in cluster if c[-4:] == ".csv"]
    sum = {c : 0 for c in cluster}
    for c1 in cluster:
        for c2 in cluster:
            sum[c1] += sim_data[c1][c2][2]

    min, res = None, None
    for c1 in cluster:
        if min is None or sum[c1] < min:
            min, res = sum[c1], c1
    return res + '.csv'

def select_anomaly_size(anomaly_size, file_name, data):
    if anomaly_size:
        return anomaly_size[file_name.split("-")[0]]
    else:
        c = Counter(data)
        top = (0.5 * c[1])/len(data)
        return top

def get_train_size(file_name):
    if file_name in ts:
        return ts[file_name]
    else:
        return 0.5

def RRCF_test(use_src_dir, output, anomaly_size):
    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir + p)])
    length = len(file_list)
    file_dict = {file_list[i]:i for i in range(length)}
    perform = pd.DataFrame({"file": file_list, "recall": [0.0]*length, "precision":[0.0]*length, "F1-score":[0.0]*length,
    "storage":[0.0]*length, "time":[0.0]*length})
    print(st.STRING)
    for file in file_list:
        file = "tps-10.0.210.35.csv"
        train_size = get_train_size(use_src_dir.split("/", 2)[2]+file)
        train_f, train_tag, train_time, test_f, test_tag, test_time = preprocess(use_src_dir, file, train_size, 1-train_size)
        print(file+" test begin.")
        file_perform = pd.DataFrame({"times":list(range(REPEAT_TIMES)), "recall": [0.0]*REPEAT_TIMES,
        "precision":[0.0]*REPEAT_TIMES, "F1-score":[0.0]*REPEAT_TIMES,
                                     "storage":[0.0]*REPEAT_TIMES, "time":[0.0]*REPEAT_TIMES})

        top = select_anomaly_size(anomaly_size, file, train_tag)

        for j in range(REPEAT_TIMES):
            print(str(j) + " times test. training ", end="")
            start = time.time()
            a = RRCF(tree_num = 70, tree_size = 1024, top = top)
            a.fit(X = train_f)
            print("testing")
            codisp, predict = a.predict(test_f)
            end = time.time()
            pd.DataFrame({"timestamp": test_time, "anomaly": predict}).to_csv(output + "labels-" + st.STRING + file, index= False)
            file_perform.loc[j, "time"] += end - start
            file_perform.loc[j, "storage"] += get_size(a)
            data = label_evaluation(predict, test_tag)
            file_perform.loc[j, "F1-score"] += data["F1-score"]
            file_perform.loc[j, "precision"] += data["precision"]
            file_perform.loc[j, "recall"] += data["recall"]
            print(data)
        input("input")
        perform.iloc[file_dict[file], 1:] = file_perform.iloc[:, 1:].sum()/REPEAT_TIMES
        break
        #file_perform.to_csv(output + st.STRING + file, index = False)
    #perform.to_csv(output + "performance-" + st.STRING + ".csv", index = False)


if __name__ == "__main__":
    anomaly_size = {"rplp": 0.006, "rspt": 0.0012, "sucp": 0.0025, "tps": 0.0026}
    use_src_dir = "../single-data/"
    if use_src_dir == "../NAB/":
        for f in os.listdir(use_src_dir):
            if os.path.isdir(use_src_dir + f):
                RRCF_test(use_src_dir+f+"/", (use_src_dir+f+"/")[1:], anomaly_size = None)
    elif use_src_dir == "../single-data/":
        RRCF_test(use_src_dir, use_src_dir[1:], anomaly_size=anomaly_size)

