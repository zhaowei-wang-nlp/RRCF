#coding=utf-8
import os
from utils import *
import time
from rrcf import RRCF
import pickle
from bestF1 import compute_F1_dir, compute_best_F1
REPEAT_TIMES = 1
import json
sim_data = None
def find_nearest(cluster):
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
            print(central, end = " ")
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
            elif st.FEEDBACK == "POINT":
                a.insert_more_normal(st.STRING + "/" + central, train_tag)
            elif st.FEEDBACK == "BOTH":
                a.update_tree_weight(st.STRING + "/" + central, train_tag)
                a.insert_more_normal(st.STRING + "/" + central, train_tag)


            perform.loc[file_index[central], "storage"] += get_size(a)

            print("testing")
            for f in c:
                print(f, end = " ")
                start = time.time()
                codisp = a.predict(f_dict[f])
                end = time.time()
                print(end - start)
                perform.loc[file_index[f], "test-time"] += end - start

                pd.DataFrame({"timestamp": time_dict[f], "score":codisp}).to_csv(output + "test-" + st.STRING + f, index=False)
    perform.iloc[:, 2:] /= REPEAT_TIMES
    perform.to_csv(output + "performance-" + st.STRING + ".csv", index = False)


def RRCF_test(use_src_dir, output, batch, batch_size):
    file_list = sorted([p for p in os.listdir(use_src_dir) if os.path.isfile(use_src_dir + p)])
    # TODO RECOVER
    file_list = ["da10a6.csv"]#["adb2fd.csv", "6a757d.csv", "42d661.csv", "da10a6.csv", "8723f0.csv", "6d1114.csv", "6efa3a.csv"]
    file_list = file_list[batch_size * batch: min(batch_size * batch + batch_size, len(file_list))]
    length = len(file_list)
    file_index = {file_list[i]:i for i in range(length)}
    perform = pd.DataFrame({"file": file_list, "storage":[0.0]*length, "train-time":[0.0]*length, "codisp-time":[0.0]*length,
                            "test-time":[0.0]*length, "best-F1":[0.0]*length, "precision":[0.0]*length, "recall":[0.0]*length,
                            "best-threshold":[0.0]*length})
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
            train_codisp = a.set_threshold(train_time)# TODO RECOVER
            end = time.time()
            tt += end - start
            perform.loc[file_index[file], "codisp-time"] += tt

            if st.SELECT_POINT == "TOP":
                a.select_points_top(st.STRING + "/" + file, train_time)
            elif st.SELECT_POINT == "MID":
                a.select_points_mid(st.STRING + "/" + file, train_time)
            elif st.SELECT_POINT == "RANDOM":
                a.select_points_randomly(st.STRING + "/" + file, train_time)
            elif st.SELECT_POINT == "BUCKET":
                a.select_points_bucket(st.STRING + "/" + file, train_time)
            if st.FEEDBACK == "WEIGHT":
                a.update_tree_weight(st.STRING + "/" + file, train_tag)
            elif st.FEEDBACK == "POINT":
                a.insert_more_normal(st.STRING + "/" + file, train_tag)
            elif st.FEEDBACK == "BOTH":
                a.update_tree_weight(st.STRING + "/" + file, train_tag)
                a.insert_more_normal(st.STRING + "/" + file, train_tag)

            print("testing")
            start = time.time()
            codisp = a.predict(test_f)
            end = time.time()
            perform.loc[file_index[file], "test-time"] += end - start
            perform.loc[file_index[file], "storage"] += get_size(a)
            pd.DataFrame({"timestamp": test_time, "score": codisp}).to_csv(output + str(j) + "test-week_diff_no_diff12-da10a6.csv" , index=False)
            pd.DataFrame({"timestamp": train_time, "score": train_codisp}).to_csv(output + str(j) + "train-week_diff_no_diff12-da10a6.csv" , index=False)
            best_F1, best_threshold, precision, recall = compute_best_F1(use_src_dir + file, output + str(j) + "test-week_diff_no_diff12-da10a6.csv" )
            #pd.DataFrame({"timestamp": test_time, "score":codisp}).to_csv(output + str(j) + "test-" + st.STRING + file, index= False)# TODO RECOVER
            #pd.DataFrame({"timestamp": train_time, "score": train_codisp}).to_csv(output + str(j) + "train-" + st.STRING  + file, index=False)# TODO RECOVER
            # best_F1, best_threshold, precision, recall = compute_best_F1(use_src_dir + file, output + str(j) + "test-" + st.STRING + file) # TODO RECOVER
            perform.loc[file_index[file], "best-F1"] += best_F1
            perform.loc[file_index[file], "best-threshold"] += best_threshold
            perform.loc[file_index[file], "precision"] += precision
            perform.loc[file_index[file], "recall"] += recall
        perform.iloc[file_index[file], 1:] /= REPEAT_TIMES
        #TODO RECOVER
        perform.to_csv(output + "performance-week_diff_no_diff12-da10a6.csv" + ".csv", index=False)
        # perform.to_csv(output + "performance-" + st.STRING + "-" + str(batch) + ".csv", index = False)


if __name__ == "__main__":
    version = int(10 * float(sys.argv[1])) if len(sys.argv) > 1 else int(10 * float("6.1"))
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    use_src_dir = "../contest_data/"
    if version  == 21:
        st.OUR_FEATURE = True
        st.STRING = "2.1"
    elif version == 22:
        st.OUR_FEATURE  = st.DATA_ANALYSIS = True
        st.STRING = "2.2"
    elif version == 32:
        st.OUR_FEATURE = st.DATA_ANALYSIS = st.FEATURE_SELECT = True
        st.STRING = "3.2"
    elif version == 33:
        st.OUR_FEATURE = st.DATA_ANALYSIS = st.CUT_SELECT = True
        st.STRING = "3.3"
    elif version == 34:
        st.OUR_FEATURE = st.DATA_ANALYSIS = st.CODISP_DEPTH = True
        st.STRING = "3.4"
    elif version == 35:
        st.OUR_FEATURE = st.DATA_ANALYSIS = st.FEATURE_SELECT = st.CUT_SELECT = st.CODISP_DEPTH = True
        st.STRING = "3.5"
    elif version == 36:
        st.OUR_FEATURE = st.DATA_ANALYSIS = st.FEATURE_SELECT = st.CUT_SELECT = st.CODISP_DEPTH = st.UPDATE_ANOMALY = True
        st.UPDATE_ALL = False
        st.STRING = "3.6"
    elif version >= 50 and version < 60:
        st.OUR_FEATURE = st.DATA_ANALYSIS = st.FEATURE_SELECT = st.CUT_SELECT = st.CODISP_DEPTH = st.UPDATE_ANOMALY = True
        st.UPDATE_ALL = False
        st.SELECT_POINT = "RANDOM"
        if version == 51:
            st.STRING, st.FEEDBACK = "5.1", "WEIGHT"
        elif version == 52:
            st.STRING, st.FEEDBACK = "5.2", "POINT"
        elif version == 53:
            st.STRING, st.FEEDBACK = "5.3", "BOTH"
    elif version >= 60:
        st.OUR_FEATURE = st.DATA_ANALYSIS = st.FEATURE_SELECT = st.CUT_SELECT = st.CODISP_DEPTH = st.UPDATE_ANOMALY = True
        st.UPDATE_ALL = False
        st.FEEDBACK = "WEIGHT"
        if version == 61:
            st.STRING, st.SELECT_POINT = "6.1", "TOP"
        elif version == 62:
            st.STRING, st.SELECT_POINT = "6.2", "MID"
        elif version == 63:
            st.STRING, st.SELECT_POINT = "6.3", "BUCKET"
    st.assert_parms()
    used_method = RRCF_cluster_test if st.CLUSTER else RRCF_test
    if st.CLUSTER:
        sim_data = pickle.load(open("./contest_data/similarity_dict.dat", "rb"))
    output_dir = use_src_dir[1:]
    if not os.path.exists(output_dir + st.STRING + "/"):
        os.mkdir(output_dir + st.STRING + "/")
    used_method(use_src_dir, output_dir, batch, batch_size)

