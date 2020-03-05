import os
import time
from utils import *
from sklearn.ensemble import RandomForestClassifier
import json
import pickle
from evaluation import label_evaluation
REPEAT_TIMES = 5
sim_data = pickle.load(open("./contest_data/similarity_dict.dat", "rb"))

def find_nearest(cluster: list) -> str:
    cluster = [c[:-4] for c in cluster if c[-4:] == ".csv"]
    sum = {c : 0 for c in cluster}
    for c1 in cluster:
        for c2 in cluster:
            sum[c1] += sim_data[c1][c2][0]

    max, res = None, None
    for c1 in cluster:
        if max is None or sum[c1] > max:
            max, res = sum[c1], c1
    return res + '.csv'
def Random_Forest(use_src_dir, output, type = "full", label_type = None):
    # read clusters' info
    clusters = json.load(open("./contest_data/file_clusters.txt", "r"))
    clusters = {int(n):clusters[n] for n in clusters}

    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir+p)])
    length = len(file_list)
    file_index = {file_list[i]: i for i in range(length)}

    cluster_index = [0.0] * length
    for cn in clusters:
        for file in clusters[cn]:
            cluster_index[file_index[file]] = cn

    perform = pd.DataFrame({"file": file_list, "cluster": cluster_index, "recall": [0.0] * length, "precision": [0.0] * length, "F1-score": [0.0] * length,
         "storage": [0.0] * length, "train-time": [0.0] * length, "test-time": [0.0] * length})

    for cnumber in clusters:
        c = clusters[cnumber]
        central = find_nearest(c)
        print(str(cnumber) + "cluster test begin.")
        train_f, train_tag, train_time = \
        preprocess(use_src_dir, central, 0.5, 0.5)[:3]

        if type == "part":
            indices = pd.read_csv("./active/" + label_type + "/" + central)["indices"].values
            train_f, train_tag, train_time = train_f[indices], train_tag[indices], train_time[indices]

        f_dict, tag_dict, time_dict = {}, {}, {}
        best_performance = {}
        for f in c:
            f_dict[f], tag_dict[f], time_dict[f] = preprocess(use_src_dir, f)[3:]
            best_performance[f] = -1

        for j in range(REPEAT_TIMES):

            print(str(j) + "times test. training ", end="")
            start = time.time()
            a = RandomForestClassifier(n_estimators= 70, max_features= None,n_jobs = 1)
            a.fit(X= train_f, y = train_tag)
            end = time.time()
            tt = end - start
            perform.loc[file_index[central], "train-time"] += tt

            perform.loc[file_index[central], "storage"] += get_size(a)

            print("testing")
            for f in c:
                start = time.time()
                predict = a.predict(f_dict[f])
                end = time.time()
                perform.loc[file_index[f], "test-time"] += end - start

                data = label_evaluation(predict, tag_dict[f])
                if data["F1-score"] > best_performance[f]:
                    best_performance[f] = data["F1-score"]
                    pd.DataFrame({"timestamp": time_dict[f], "anomaly": predict}).to_csv(output + "test-" + f, index=False)

                perform.loc[file_index[f], "F1-score"] += data["F1-score"]
                perform.loc[file_index[f], "recall"] += data["recall"]
                perform.loc[file_index[f], "precision"] += data["precision"]
                print(data)
    perform.iloc[:, 2:] /= REPEAT_TIMES
    perform.to_csv(output + "performance" + ".csv", index = False)


if __name__ == "__main__":
    use_src_dir = "../contest_data/"
    output_dir =  use_src_dir[1:] + "RF/"
    if not os.path.exists(output_dir + "/"):
        os.mkdir(output_dir + "/")
    Random_Forest(use_src_dir, output_dir, type = "part", label_type= "TOP_WEIGHT")