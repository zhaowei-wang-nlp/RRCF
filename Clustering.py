import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import pickle
import json
import matplotlib.pyplot as plt
def plot_cluster(cluster, output):
    plt.figure(figsize=(19.2, 13.4), dpi=100)
    for i in range(len(cluster)):
        kpi = cluster[i]
        plt.subplot( len(cluster), 1, i + 1)
        plt.title(kpi)
        data = pd.read_csv("../contest_data/" + kpi)
        data["time"] = pd.to_datetime(data["timestamp"], unit='s')
        plt.plot(data["time"], data["value"])
    plt.savefig(output + "cluster-" + str(kpi) + ".jpg")

def form_array(sim_data):

    index = {}
    for a in sorted(sim_data.keys()):
        if a not in index:
            index[a] = len(index)

    dis_mat = np.zeros((len(index),len(index)))
    for a in sim_data:
        for b in sim_data[a]:
            if a in index and b in index:
                dis_mat[index[a]][index[b]] = 1/sim_data[a][b][0]-1 if sim_data[a][b][0]!=0 else sys.float_info.max/10
    return dis_mat, index


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) > 1 else "./contest_data/similarity_dict.dat"
    data_dict = pickle.load(open(file_name, "rb"))
    dis_mat, name2index = form_array(data_dict)
    index2name = {name2index[name]:name for name in name2index}

    classes = [int(i) for i in DBSCAN(eps = 23, min_samples = 1).fit_predict(dis_mat)]
    clusters = {}
    for i in range(len(classes)):
        if classes[i] not in clusters:
            clusters[classes[i]] = []
        clusters[classes[i]].append(index2name[i]+".csv")
    print(classes)

    output = "./contest_cluster_pic/"
    for file in os.listdir(output):
        os.remove(output + file)
    for c in clusters:
        plot_cluster(clusters[c], output)

    with open("./contest_data/file_clusters.txt","w") as output:
        output.write(json.dumps(clusters, ensure_ascii=False) + '\n')