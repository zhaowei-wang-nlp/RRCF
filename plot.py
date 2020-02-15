import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_points(file, origin_result, our_result):
    data = pd.read_csv(file).iloc[60:, :]
    r1 = pd.read_csv(origin_result)
    r2 = pd.read_csv(our_result)

    data.index = data["timestamp"]
    data = data.loc[r1["timestamp"][0]:, :]
    data["time"] = pd.to_datetime(data["timestamp"], unit='s')
    data.index = range(len(data))
    plt.figure(figsize=(19.2, 13.4), dpi=100)
    plt.plot(data["time"].values, data["value"].values)
    plt.scatter(data[ data["anomaly"] == 1 ]["time"], data[ data["anomaly"] == 1 ]["value"], c = "red", s = 110, label = "True Anomalies")

    plt.scatter(data[r1["anomaly"] == 1]["time"], data[r1["anomaly"] == 1]["value"], c = "greenyellow", s = 80, label = "RRCF")
    plt.scatter(data[r2["anomaly"] == 1]["time"], data[r2["anomaly"] == 1]["value"], c = "black", s = 30, label = "our solution")
    plt.legend()
    a = "./JSYHpic/" + file.split("/")[-1] + ".jpg"
    plt.savefig("./JSYH_pic/" + file.split("/")[-1] + ".jpg")
    #plt.show()
if __name__ == "__main__":
    files = ["kpi.P1CCSVC.arsp.csv", "kpi.P8N-DPSQ.arsp.csv", "kpi.P8N-DPSQ.bsrt.csv"
             , "kpi.P8N-ELB.arsp.csv", "kpi.P8N-PAMT.arsp.csv", "kpi.P8N-PAMT.ssrt.csv",
             "kpi.P8N-RCC.arsp.csv", "kpi.P8N-RCC.ssrt.csv"]
    for file in files:
        plot_points("../JSYH_data/"+file,
                    "../rrcf-master/JSYH_data/labels-"+file,
                    "./JSYH_data/labels-our"+file)



