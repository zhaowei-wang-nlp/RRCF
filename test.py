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



def RRCF_cluster_test(use_src_dir, output, anomaly_size):
    # read clusters' info
    clusters = json.loads(open("file_clusters.txt", "r"))
    clusters = {int(n):clusters[n] for n in clusters}

    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir+p)])
    length = len(file_list)
    file_index = {file_list[i]: i for i in range(length)}

    cluster_index = [0.0] * length
    for cn in clusters:
        for file in clusters[cn]:
            cluster_index[file_index[file]] = cn

    perform = pd.DataFrame({"file": file_list, "cluster": cluster_index, "recall": [0.0] * length, "precision": [0.0] * length, "F1-score": [0.0] * length,
    "storage": [0.0] * length, "time": [0.0]*length})
    print(st.STRING)

    for cnumber in clusters:
        c = clusters[cnumber]
        central = find_nearest(c)
        print(str(cnumber) + "cluster test begin.")
        train_f, train_tag, train_time = \
        preprocess(use_src_dir, central, 0.5, 0.5)[:3]

        f_dict, tag_dict, time_dict = {}, {}, {}
        best_perforamnce = {}
        for f in c:
            f_dict[f], tag_dict[f], time_dict[f] = preprocess(use_src_dir, f)[3:]
            best_perforamnce[f] = -1

        for j in range(REPEAT_TIMES):

            print(str(j) + "times test. training ", end="")
            top = select_anomaly_size(anomaly_size, central, train_tag)
            start = time.time()
            a = RRCF(tree_num= 70, tree_size= 1024, top = top)
            a.fit(X= train_f)
            end = time.time()
            perform.loc[file_index[central], "time"] += end - start
            perform.loc[file_index[central], "storage"] += get_size(a)

            print("testing")
            for f in c:
                start = time.time()
                codisp, predict = a.predict(f_dict[f])
                end = time.time()
                perform.loc[file_index[f], "time"] += end - start

                data = label_evaluation(predict, tag_dict[f])
                if data["F1-score"] > best_perforamnce[f]:
                    best_perforamnce[f] = data["F1-score"]
                    pd.DataFrame({"timestamp": time_dict[f], "anomaly": predict}).to_csv(output + "labels-" + st.STRING + f, index=False)
                    plot_points(use_src_dir+f, output+"labels-"+st.STRING+f, st.STRING)

                perform.loc[file_index[f], "F1-score"] += data["F1-score"]
                perform.loc[file_index[f], "recall"] += data["recall"]
                perform.loc[file_index[f], "precision"] += data["precision"]
                print(data)
    perform.iloc[:, 2:] /= REPEAT_TIMES
    perform.to_csv(output + "performace-" + st.STRING + ".csv", index = False)


def RRCF_test(use_src_dir, output, anomaly_size):
    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir + p)])
    length = len(file_list)
    file_index = {file_list[i]:i for i in range(length)}
    perform = pd.DataFrame({"file": file_list, "recall": [0.0]*length, "precision":[0.0]*length, "F1-score":[0.0]*length,
    "storage":[0.0]*length, "time":[0.0]*length})
    print(st.STRING)
    for file in file_list:
        train_f, train_tag, train_time, test_f, test_tag, test_time = preprocess(use_src_dir, file, 0.5, 0.5)
        print(file+" test begin.")

        top = select_anomaly_size(anomaly_size, file, train_tag)
        best_performance = -1
        for j in range(REPEAT_TIMES):

            print(str(j) + " times test. training ", end="")

            start = time.time()
            a = RRCF(tree_num = 70, tree_size = 1024, top = top)
            a.fit(X = train_f)
            print("testing")
            codisp, predict = a.predict(test_f)
            end = time.time()

            perform.loc[file_index[file], "time"] += end - start
            perform.loc[file_index[file], "storage"] += get_size(a)

            data = label_evaluation(predict, test_tag)
            if data["F1-score"] > best_performance:
                best_performance = data["F1-score"]
                pd.DataFrame({"timestamp": test_time, "anomaly": predict}).to_csv(output + "labels-" + st.STRING + file, index= False)
                plot_points(use_src_dir+file, output + "labels-" + st.STRING + file, st.STRING)

            perform.loc[file_index[file], "F1-score"] += data["F1-score"]
            perform.loc[file_index[file], "recall"] += data["recall"]
            perform.loc[file_index[file], "precision"] += data["precision"]
            print(data)
    perform.iloc[:, 1:] /= REPEAT_TIMES
    perform.to_csv(output + "performance-" + st.STRING + ".csv", index = False)


if __name__ == "__main__":
    anomaly_size = {"rplp": 0.006, "rspt": 0.0012, "sucp": 0.0025, "tps": 0.0026}
    use_src_dir = "../single-data/"
    used_method = RRCF_cluster_test if st.CLUSTER else RRCF_test
    used_method(use_src_dir, use_src_dir[1:], anomaly_size=anomaly_size)

