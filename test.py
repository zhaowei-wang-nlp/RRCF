import os
from utils import *
import time
from rrcf import RRCF
import pickle
REPEAT_TIMES = 2
import json
if st.CLUSTER:
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


def RRCF_cluster_test(use_src_dir, output):
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

    perform = pd.DataFrame({"file": file_list, "cluster": cluster_index,
         "storage": [0.0] * length, "train-time": [0.0] * length, "codisp-time": [0.0] * length, "test-time": [0.0] * length})

    if st.STRING != "":
        output += st.STRING + "/"

    print(st.STRING)

    for cnumber in clusters:
        c = clusters[cnumber]
        central = find_nearest(c)
        print(str(cnumber) + "cluster test begin.")
        train_f, train_tag, train_time = \
        preprocess(use_src_dir, central, 0.5, 0.5)[:3]

        f_dict, tag_dict, time_dict = {}, {}, {}
        train_codisp_flag = False
        for f in c:
            f_dict[f], tag_dict[f], time_dict[f] = preprocess(use_src_dir, f)[3:]

        for j in range(REPEAT_TIMES):

            print(str(j) + "times test. training ", end="")
            start = time.time()
            a = RRCF(tree_num= TREE_NUM, tree_size= TREE_SIZE)
            a.fit(X= train_f)
            end = time.time()
            tt = end - start
            perform.loc[file_index[central], "train-time"] += tt

            start = time.time()
            train_codisp = a.set_threshold()
            end = time.time()
            tt += end - start
            perform.loc[file_index[central], "codisp-time"] += tt
            if not train_codisp_flag:
                pd.DataFrame({"timestamp": train_time, "score": train_codisp}).to_csv(output + "train-" + st.STRING + central, index=False)
                train_codisp_flag = True

            if st.SELECT_POINT == "TOP":
                a.select_points_top(st.STRING + "/" + central, train_time)
            elif st.SELECT_POINT == "MID":
                a.select_points_mid(st.STRING + "/" + central, train_time)
            elif st.SELECT_POINT == "RANDOM":
                a.select_points_randomly(st.STRING + "/" + central, train_time)
            elif st.SELECT_POINT == "BUCKET":
                a.select_points_bucket(st.STRING + "/" + central, train_time)
            if st.FEEDBACK == "WEIGHT":
                a.update_tree_weight(st.STRING + "/" + central, train_tag)


            perform.loc[file_index[central], "storage"] += get_size(a)

            print("testing")
            for f in c:
                start = time.time()
                codisp = a.predict(f_dict[f])
                end = time.time()
                perform.loc[file_index[f], "test-time"] += end - start

                pd.DataFrame({"timestamp": time_dict[f], "score":codisp}).to_csv(output + "test-" + st.STRING + f, index=False)

    perform.iloc[:, 2:] /= REPEAT_TIMES
    perform.to_csv(output + "performance-" + st.STRING + ".csv", index = False)


def RRCF_test(use_src_dir, output):
    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir + p)])
    length = len(file_list)
    file_index = {file_list[i]:i for i in range(length)}
    perform = pd.DataFrame({"file": file_list, "storage":[0.0]*length, "train-time":[0.0]*length, "codisp-time":[0.0]*length,
                            "test-time":[0.0]*length})

    if st.STRING != "":
        output += st.STRING + "/"

    print(st.STRING)
    for file in file_list:
        train_f, train_tag, train_time, test_f, test_tag, test_time = preprocess(use_src_dir, file, 0.5, 0.5)
        print(file+" test begin.")

        for j in range(REPEAT_TIMES):

            print(str(j) + " times test. training ", end="")

            start = time.time()
            a = RRCF(tree_num = TREE_NUM, tree_size = TREE_SIZE)
            a.fit(X = train_f)
            end = time.time()
            tt = end - start
            perform.loc[file_index[file], "train-time"] += tt

            start = time.time()
            train_codisp = a.set_threshold()
            end = time.time()
            tt += end - start
            perform.loc[file_index[file], "codisp-time"] += tt


            print("testing")
            start = time.time()
            codisp = a.predict(test_f)
            end = time.time()
            perform.loc[file_index[file], "test-time"] += end - start
            perform.loc[file_index[file], "storage"] += get_size(a)
            pd.DataFrame({"timestamp": test_time, "score":codisp}).to_csv(output + "test-" + st.STRING + file, index= False)
            pd.DataFrame({"timestamp": train_time, "score": train_codisp}).to_csv(output + "train-" + st.STRING + file, index=False)

        perform.iloc[file_index[file], 1:] /= REPEAT_TIMES
        perform.to_csv(output + "performance-" + st.STRING + ".csv", index = False)


if __name__ == "__main__":
    use_src_dir = "../contest_data/"
    used_method = RRCF_cluster_test if st.CLUSTER else RRCF_test
    used_method(use_src_dir, use_src_dir[1:])

