import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_points(file, origin_result, our_result):
    data = pd.read_csv(file).iloc[60:, :]
    r1 = pd.read_csv(origin_result)
    r2 = pd.read_csv(our_result)


    data["time"] = pd.to_datetime(data["timestamp"], unit='s')
    plt.figure(figsize=(19.2, 13.4), dpi=100)
    plt.plot(data["time"].values, data["value"].values)
    plt.scatter(data[ data["anomaly"] == 1 ]["time"], data[ data["anomaly"] == 1 ]["value"], c = "orange", s = 110, label = "True Anomalies")
    data.index = data["timestamp"]
    data = data.loc[r1["timestamp"][0]:, :]
    data.index = range(len(data))
    plt.scatter(data[data["anomaly"] == 1]["time"], data[data["anomaly"] == 1]["value"], c="red", s=110,
                label="True Anomalies")
    plt.scatter(data[r1["anomaly"] == 1]["time"], data[r1["anomaly"] == 1]["value"], c = "greenyellow", s = 80, label = "RRCF")
    plt.scatter(data[r2["anomaly"] == 1]["time"], data[r2["anomaly"] == 1]["value"], c = "black", s = 30, label = "3.2")
    plt.legend()

    plt.savefig("./contest_pic/" + file.split("/")[-1] + ".jpg")

if __name__ == "__main__":
    for file in os.listdir("../contest_data/"):
        plot_points("../contest_data/"+file,
                    "./contest_data/RRCF/test-RRCF"+file,
                    "./contest_data/FEATURE_SELECT/test-FEATURE_SELECT"+file)



